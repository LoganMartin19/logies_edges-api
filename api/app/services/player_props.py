# api/app/services/player_props.py
from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re
import math

from sqlalchemy.orm import Session

from ..models import PlayerOdds, Fixture, PlayerSeasonStats
from .apifootball import _get, BASE_URL, _get_meta, get_fixture_players, get_fixture
from .player_cache import get_team_season_players_cached
from .player_model import prob_over_xpoint5

# ---------------------------------------------------------------------
# Bet ID allowlist (API-Football odds -> bets[].id)
# ---------------------------------------------------------------------

BET_ID_MAP: Dict[int, Dict[str, Any]] = {
    # scorers
    92:  {"market": "anytime_goalscorer", "line": None},
    93:  {"market": "first_goalscorer",   "line": None},
    94:  {"market": "last_goalscorer",    "line": None},
    218: {"market": "anytime_goalscorer", "line": None},
    219: {"market": "first_goalscorer",   "line": None},
    226: {"market": "last_goalscorer",    "line": None},

    # cards
    102: {"market": "yellow", "line": 0.5},
    251: {"market": "yellow", "line": 0.5},
    103: {"market": "red",    "line": 0.5},

    # assists / score or assist
    212: {"market": "assists",          "line": None},
    255: {"market": "assists",          "line": None},
    256: {"market": "assists",          "line": None},
    257: {"market": "score_or_assist",  "line": None},
    258: {"market": "score_or_assist",  "line": None},
    259: {"market": "score_or_assist",  "line": None},

    # buckets (critical)
    215: {"market": "player_singles", "line": "from_value"},
    213: {"market": "player_triples", "line": "from_value"},
    # 214: {"market": "player_doubles", "line": "from_value"},

    # direct shots/SOT totals (some books provide direct, some via buckets)
    242: {"market": "sot",   "line": "from_value"},
    264: {"market": "sot",   "line": "from_value"},
    265: {"market": "shots", "line": "from_value"},

    240: {"market": "shots", "line": "from_value"},
    241: {"market": "shots", "line": "from_value"},

    269: {"market": "sot",   "line": "from_value"},
    270: {"market": "shots", "line": "from_value"},
    275: {"market": "sot",   "line": "from_value"},
    276: {"market": "shots", "line": "from_value"},

    # fouls / tackles
    266: {"market": "fouls",   "line": "from_value"},
    271: {"market": "fouls",   "line": "from_value"},
    277: {"market": "fouls",   "line": "from_value"},
    272: {"market": "tackles", "line": "from_value"},
    278: {"market": "tackles", "line": "from_value"},
}

# ---------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------

_LINE_RE = re.compile(r"(-?\d+(?:\.\d+)?)")  # finds 4.5 or -0.5 anywhere
_SPACE_RE = re.compile(r"\s+")
_DASH_LINE_RE = re.compile(r"\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*$")

NO_LINE_MARKETS = {
    "anytime_goalscorer",
    "first_goalscorer",
    "last_goalscorer",
    "assists",
    "score_or_assist",
}

BUCKET_MARKETS = {"player_singles", "player_doubles", "player_triples"}


def _safe_float(x, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x, default: Optional[int] = None) -> Optional[int]:
    if x is None:
        return default
    try:
        return int(x)
    except Exception:
        return default


def _parse_line(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).strip().replace("+", "")
    try:
        return float(s)
    except Exception:
        return None


def _clean_player_name(name: str) -> str:
    name = (name or "").strip()
    # remove trailing separators left by "Name - 0.5"
    name = re.sub(r"\s*[-–—]\s*$", "", name).strip()
    name = re.sub(r"\s{2,}", " ", name).strip()
    return name


def _norm_name(s: str) -> str:
    """
    Normalization that makes:
      "A. Bastoni" == "A Bastoni" == "a bastoni"
      and ignores dashes and extra spaces.
    """
    s = (s or "").strip().lower()
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace(".", " ")
    s = re.sub(r"[-–—]", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _split_player_and_line(value_str: str) -> Tuple[str, Optional[float]]:
    """
    Handles:
      "Alessandro Bastoni - 0.5"
      "Henrikh Mkhitaryan - 6.5"
      "Bastoni 4.5"
    """
    if not value_str:
        return "", None

    m = _LINE_RE.search(value_str)
    if not m:
        return _clean_player_name(value_str), None

    line = _parse_line(m.group(1))
    player = _LINE_RE.sub("", value_str, count=1).strip()
    player = _clean_player_name(player)
    return player, line


def _split_first_last(full: str) -> Tuple[Optional[str], Optional[str]]:
    """
    "A. Bastoni" => ("a", "bastoni")
    "Alessandro Bastoni" => ("alessandro", "bastoni")
    "Luis Henrique de Lima" => ("luis", "lima")  (best-effort: last token)
    """
    s = _norm_name(full)
    if not s:
        return None, None
    s = s.strip("-").strip()
    parts = [p for p in s.split(" ") if p]
    if len(parts) < 2:
        return (parts[0], None) if parts else (None, None)
    first = parts[0].replace(".", "")
    last = parts[-1].replace(".", "")
    return first or None, last or None


# ---------------------------------------------------------------------
# API call (odds)
# ---------------------------------------------------------------------

def fetch_player_odds_raw_for_fixture(db: Session, fixture_id: int) -> dict:
    fx: Fixture | None = db.query(Fixture).filter(Fixture.id == fixture_id).one_or_none()
    if not fx or not fx.provider_fixture_id:
        return {"ok": False, "error": "fixture missing provider_fixture_id"}

    provider_fixture_id = int(fx.provider_fixture_id)
    url = f"{BASE_URL}/odds"
    params = {"fixture": provider_fixture_id}
    return _get_meta(url, params)


def fetch_player_odds_for_fixture(provider_fixture_id: int) -> List[Dict[str, Any]]:
    url = f"{BASE_URL}/odds"
    payload = _get(url, {"fixture": provider_fixture_id}) or []
    return payload if isinstance(payload, list) else []


# ---------------------------------------------------------------------
# Fixture roster mapping (stable): name -> player_id using season-players cache
# ---------------------------------------------------------------------

def _build_fixture_player_index(db: Session, fixture_id: int) -> Dict[str, int]:
    """
    Uses cached season players (team-season roster) to build aliases for name->player_id.

    Builds:
      - raw provider display name: "A. Bastoni"
      - full: "Alessandro Bastoni"
      - initial variants: "A Bastoni" / "A. Bastoni"
      - last name only: "Bastoni" (only if unique in this fixture)
    """
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).one_or_none()
    if not fx or not fx.provider_fixture_id:
        return {}

    pfx = int(fx.provider_fixture_id)
    fjson = get_fixture(pfx) or {}
    core = (fjson.get("response") or [None])[0] or {}
    lg = core.get("league") or {}
    season = int(lg.get("season") or 0)

    teams = core.get("teams") or {}
    home_id = int(((teams.get("home") or {}).get("id")) or 0)
    away_id = int(((teams.get("away") or {}).get("id")) or 0)

    if not (season and home_id and away_id):
        return {}

    home_rows = get_team_season_players_cached(db, home_id, season) or []
    away_rows = get_team_season_players_cached(db, away_id, season) or []

    alias_to_id: Dict[str, int] = {}

    def _add_alias(key: str, pid: int):
        k = _norm_name(key)
        if not k:
            return
        alias_to_id.setdefault(k, pid)

    def _process(rows: List[dict]):
        for row in rows:
            pl = (row.get("player") or {}) or {}
            pid = int(pl.get("id") or 0)
            if not pid:
                continue

            raw_name = (pl.get("name") or "").strip()  # often "A. Bastoni"
            first = (pl.get("firstname") or "").strip()
            last = (pl.get("lastname") or "").strip()

            # 1) always alias provider display name
            if raw_name:
                _add_alias(raw_name, pid)

            # 2) firstname/lastname → full + initial aliases
            if first and last:
                _add_alias(f"{first} {last}", pid)
                fi = first[:1]
                if fi:
                    _add_alias(f"{fi} {last}", pid)
                    _add_alias(f"{fi}. {last}", pid)
                _add_alias(f"__LAST__:{last}", pid)

            # 3) fallback: split raw_name if firstname/lastname missing
            if (not first or not last) and raw_name:
                f, l = _split_first_last(raw_name)
                if f and l:
                    _add_alias(f"{f} {l}", pid)
                    _add_alias(f"{f}. {l}", pid)
                    _add_alias(f"__LAST__:{l}", pid)

    _process(home_rows)
    _process(away_rows)

    # last-name-only aliases only if unique in this fixture
    last_name_to_ids: Dict[str, set] = {}
    for k, pid in list(alias_to_id.items()):
        if k.startswith("__last__:"):
            ln = _norm_name(k.split(":", 1)[1])
            if ln:
                last_name_to_ids.setdefault(ln, set()).add(pid)

    for ln, ids in last_name_to_ids.items():
        if len(ids) == 1:
            alias_to_id[ln] = list(ids)[0]

    # Remove internal markers
    for k in [k for k in list(alias_to_id.keys()) if k.startswith("__last__:")]:
        alias_to_id.pop(k, None)

    return alias_to_id


def _resolve_player_id_from_alias(
    pid0: Optional[int],
    player_name: str,
    alias_map: Dict[str, int],
) -> Optional[int]:
    """
    Resolver for odds names using alias_map from season players cache.
    """
    if pid0:
        return int(pid0)
    if not player_name:
        return None

    nm = _norm_name(_clean_player_name(player_name))
    if nm in alias_map:
        return alias_map[nm]

    f, l = _split_first_last(player_name)
    if l:
        # try "A Bastoni" / "A. Bastoni"
        if f:
            fi = f[:1].lower()
            for cand in (f"{fi} {l}", f"{fi}. {l}"):
                pid = alias_map.get(_norm_name(cand))
                if pid:
                    return int(pid)

        # try unique last name
        pid = alias_map.get(_norm_name(l))
        if pid:
            return int(pid)

    return None


# ---------------------------------------------------------------------
# Player stats helpers (from cached PlayerSeasonStats.stats_json)
# ---------------------------------------------------------------------

def _get_latest_player_stats(db: Session, player_id: int, season: Optional[int] = None) -> Optional[dict]:
    q = db.query(PlayerSeasonStats).filter(PlayerSeasonStats.player_id == player_id)
    if season is not None:
        q = q.filter(PlayerSeasonStats.season == season)
    row = q.order_by(PlayerSeasonStats.updated_at.desc()).first()
    if not row or not row.stats_json:
        return None
    return row.stats_json


def _sum_minutes_and_stat(stats_json: Any, stat_path_candidates: List[Tuple[str, ...]]) -> Tuple[float, float]:
    minutes = 0.0
    total = 0.0

    stats_list = None
    if isinstance(stats_json, list):
        stats_list = stats_json
    elif isinstance(stats_json, dict):
        stats_list = stats_json.get("statistics") or stats_json.get("response") or stats_json.get("stats")

    if not isinstance(stats_list, list):
        return 0.0, 0.0

    for blk in stats_list:
        if not isinstance(blk, dict):
            continue
        s = blk.get("statistics") if "statistics" in blk else blk

        m = None
        try:
            m = (s.get("games") or {}).get("minutes")
        except Exception:
            m = None
        if m is None:
            m = s.get("minutes")
        m = _safe_float(m, 0.0) or 0.0
        minutes += m

        found = None
        for path in stat_path_candidates:
            cur = s
            ok = True
            for key in path:
                if not isinstance(cur, dict) or key not in cur:
                    ok = False
                    break
                cur = cur[key]
            if ok:
                found = cur
                break

        if found is None:
            continue

        if isinstance(found, dict):
            found = found.get("total") or found.get("value")

        total += (_safe_float(found, 0.0) or 0.0)

    return minutes, total


def _per90_from_cached(db: Session, player_id: int, season: Optional[int], stat_key: str) -> Optional[float]:
    payload = _get_latest_player_stats(db, player_id, season=season)
    if payload is None:
        return None

    STAT_PATHS: Dict[str, List[Tuple[str, ...]]] = {
        "shots": [("shots", "total"), ("shots_total",)],
        "sot":   [("shots", "on"), ("shots_on",)],

        "fouls": [("fouls", "committed"), ("fouls_committed",)],
        "tackles": [("tackles", "total"), ("tackles_total",)],
        "interceptions": [("tackles", "interceptions"), ("interceptions",)],

        "passes": [("passes", "total"), ("passes_total",)],
        "key_passes": [("passes", "key"), ("key_passes",)],
    }

    paths = STAT_PATHS.get(stat_key)
    if not paths:
        return None

    mins, tot = _sum_minutes_and_stat(payload, paths)
    if mins <= 0:
        return None

    return float((tot / mins) * 90.0)


def _expected_minutes_from_cached(db: Session, player_id: int, season: Optional[int]) -> int:
    payload = _get_latest_player_stats(db, player_id, season=season)
    if payload is None:
        return 80

    stats_list = payload if isinstance(payload, list) else payload.get("statistics") or payload.get("response") or []
    if not isinstance(stats_list, list):
        return 80

    mins_total = 0.0
    apps_total = 0.0
    for blk in stats_list:
        if not isinstance(blk, dict):
            continue
        s = blk.get("statistics") if "statistics" in blk else blk
        games = s.get("games") or {}
        mins_total += (_safe_float(games.get("minutes"), 0.0) or 0.0)
        apps_total += (_safe_float(games.get("appearences") or games.get("appearances"), 0.0) or 0.0)

    if apps_total > 0 and mins_total > 0:
        avg = mins_total / apps_total
        return int(max(45, min(avg, 95)))

    return 80


# ---------------------------------------------------------------------
# Bucket inference (your existing logic) — unchanged
# ---------------------------------------------------------------------

def _player_position_from_cached(db: Session, player_id: int, season: Optional[int]) -> Optional[str]:
    payload = _get_latest_player_stats(db, player_id, season=season)
    if payload is None:
        return None

    stats_list = payload if isinstance(payload, list) else payload.get("statistics") or payload.get("response") or []
    if not isinstance(stats_list, list):
        return None

    for blk in stats_list:
        if not isinstance(blk, dict):
            continue
        s = blk.get("statistics") if "statistics" in blk else blk
        games = s.get("games") or {}
        pos = (games.get("position") or "").strip().lower()
        if not pos:
            continue

        if "goal" in pos:
            return "goalkeeper"
        if "def" in pos:
            return "defender"
        if "mid" in pos:
            return "midfielder"
        if "att" in pos or "forw" in pos or "strik" in pos:
            return "attacker"

        return pos

    return None


def _bucket_priors(position: Optional[str], line: float) -> Dict[str, float]:
    pos = (position or "").strip().lower()
    w = {
        "shots": 0.20,
        "sot": 0.12,
        "fouls": 0.20,
        "tackles": 0.18,
        "passes": 0.18,
        "interceptions": 0.12,
        "key_passes": 0.10,
    }

    if pos == "defender":
        w["tackles"] += 0.22
        w["interceptions"] += 0.10
        w["fouls"] += 0.10
        w["shots"] -= 0.12
        w["sot"] -= 0.06
        w["key_passes"] -= 0.04
    elif pos == "midfielder":
        w["passes"] += 0.18
        w["key_passes"] += 0.08
        w["tackles"] += 0.06
        w["shots"] += 0.02
        w["sot"] += 0.01
    elif pos == "attacker":
        w["shots"] += 0.26
        w["sot"] += 0.14
        w["fouls"] -= 0.06
        w["tackles"] -= 0.10
        w["passes"] -= 0.06
        w["interceptions"] -= 0.06
    elif pos == "goalkeeper":
        w["shots"] -= 0.20
        w["sot"] -= 0.10
        w["passes"] += 0.10
        w["tackles"] -= 0.10
        w["fouls"] -= 0.10

    if line >= 35:
        w["passes"] += 0.35
        w["shots"] -= 0.10
        w["sot"] -= 0.06
        w["fouls"] -= 0.04
        w["tackles"] -= 0.05
    elif line >= 15:
        w["passes"] += 0.18
        w["tackles"] += 0.06
        w["interceptions"] += 0.04
        w["shots"] -= 0.06
        w["sot"] -= 0.04
    elif line >= 3:
        w["tackles"] += 0.10
        w["fouls"] += 0.10
        w["shots"] += 0.02
    else:
        w["shots"] += 0.10
        w["sot"] += 0.06

    for k in list(w.keys()):
        w[k] = max(0.001, float(w[k]))
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def _infer_bucket_stat(
    db: Session,
    fixture: Fixture,
    player_id: Optional[int],
    player_name: str,
    line: float,
    price: float,
) -> Optional[str]:
    if not player_id:
        return None

    season = fixture.kickoff_utc.year if fixture.kickoff_utc else None
    expected_minutes = _expected_minutes_from_cached(db, player_id, season=season)

    implied = 1.0 / float(price) if price and price > 0 else None
    if implied is None:
        return None

    x_half = float(line)
    if x_half < -0.5:
        return None

    candidates = ["shots", "sot", "fouls", "tackles", "passes", "interceptions", "key_passes"]

    pos = _player_position_from_cached(db, player_id, season=season)
    pri = _bucket_priors(pos, x_half)

    best_key = None
    best_score = 1e9

    for stat_key in candidates:
        per90 = _per90_from_cached(db, player_id, season=season, stat_key=stat_key)
        if per90 is None:
            continue

        p_model = prob_over_xpoint5(per90=per90, expected_minutes=expected_minutes, x_half=x_half)
        p_model = max(1e-6, min(1 - 1e-6, float(p_model)))

        fit_err = abs(p_model - float(implied))
        prior = float(pri.get(stat_key, 1e-6))
        prior_penalty = -math.log(max(prior, 1e-9))

        score = (fit_err * 1.0) + (prior_penalty * 0.12)

        if score < best_score:
            best_score = score
            best_key = stat_key

    if best_key is None:
        return None

    if price < 10:
        best_fit = 1e9
        for stat_key in candidates:
            per90 = _per90_from_cached(db, player_id, season=season, stat_key=stat_key)
            if per90 is None:
                continue
            p_model = prob_over_xpoint5(per90=per90, expected_minutes=expected_minutes, x_half=x_half)
            p_model = max(1e-6, min(1 - 1e-6, float(p_model)))
            best_fit = min(best_fit, abs(p_model - float(implied)))
        if best_fit > 0.22:
            return None

    return best_key


# ---------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------

def _extract_player_rows(db: Session, fixture: Fixture, api_response: List[dict]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(api_response, list):
        return rows

    # ✅ stable alias map for this fixture (season players cache)
    alias_map: Dict[str, int] = _build_fixture_player_index(db, fixture.id)

    for fx_block in api_response:
        bookmakers = fx_block.get("bookmakers") or fx_block.get("bookmaker") or []
        if isinstance(bookmakers, dict):
            bookmakers = [bookmakers]
        if not isinstance(bookmakers, list):
            continue

        for bm in bookmakers:
            bookmaker = bm.get("name") or bm.get("key") or "Unknown"
            bets = bm.get("bets") or []
            if not isinstance(bets, list):
                continue

            for bet in bets:
                bet_id = _safe_int(bet.get("id"))
                cfg = BET_ID_MAP.get(bet_id)
                if not cfg:
                    continue

                canonical_market = cfg["market"]
                raw_market = (bet.get("name") or "").strip()

                values = bet.get("values") or []
                if not isinstance(values, list):
                    continue

                for v in values:
                    price = _safe_float(v.get("odd") or v.get("price"))
                    if not price or price <= 0:
                        continue

                    pinfo = v.get("player") or {}
                    embedded = (v.get("value") or v.get("participant") or v.get("player_name") or "").strip()
                    nested_name = (pinfo.get("name") or "").strip()

                    if nested_name:
                        player_name = _clean_player_name(nested_name)
                        line = _parse_line(v.get("handicap") or v.get("line"))
                        if line is None and embedded:
                            _, line_guess = _split_player_and_line(embedded)
                            line = line_guess
                    else:
                        player_name, line = _split_player_and_line(embedded)

                    player_name = _clean_player_name(player_name)
                    if not player_name:
                        continue

                    pid0 = _safe_int(pinfo.get("id") or v.get("id") or v.get("player_id"))
                    player_id = _resolve_player_id_from_alias(pid0, player_name, alias_map)

                    cfg_line = cfg.get("line")

                    if canonical_market in NO_LINE_MARKETS or cfg_line is None:
                        line = None
                    elif cfg_line == 0.5:
                        line = 0.5
                    else:
                        if line is None:
                            line = _parse_line(v.get("handicap") or v.get("line"))
                        if line is None:
                            line = 0.0

                    inferred_from_bucket = None

                    if canonical_market in BUCKET_MARKETS and line is not None:
                        inferred = _infer_bucket_stat(
                            db=db,
                            fixture=fixture,
                            player_id=player_id,
                            player_name=player_name,
                            line=float(line),
                            price=float(price),
                        )
                        if inferred:
                            inferred_from_bucket = canonical_market
                            canonical_market = inferred

                    rows.append(
                        {
                            "player_id": player_id,
                            "player_name": player_name,
                            "market": canonical_market,
                            "line": (float(line) if line is not None else None),
                            "bookmaker": bookmaker,
                            "price": float(price),
                            "raw_market": raw_market,
                            "bucket": inferred_from_bucket,
                            "bet_id": bet_id,
                        }
                    )

    return rows


# ---------------------------------------------------------------------
# Ingest / upsert
# ---------------------------------------------------------------------

def ingest_player_odds_for_fixture(db: Session, fixture_id: int) -> int:
    fx: Fixture | None = db.query(Fixture).filter(Fixture.id == fixture_id).one_or_none()
    if not fx or not fx.provider_fixture_id:
        return 0

    provider_fixture_id = int(fx.provider_fixture_id)
    api_blocks = fetch_player_odds_for_fixture(provider_fixture_id)

    rows = _extract_player_rows(db, fx, api_blocks)
    if not rows:
        return 0

    now = datetime.utcnow()
    upserts = 0

    for r in rows:
        existing = (
            db.query(PlayerOdds)
            .filter(
                PlayerOdds.fixture_id == fixture_id,
                PlayerOdds.player_name == r["player_name"],
                PlayerOdds.market == r["market"],
                PlayerOdds.bookmaker == r["bookmaker"],
            )
        )

        if r["line"] is None:
            existing = existing.filter(PlayerOdds.line.is_(None))
        else:
            existing = existing.filter(PlayerOdds.line == float(r["line"]))

        existing = existing.one_or_none()

        # Upgrade NULL player_id rows when we can resolve it
        if existing:
            changed = False
            if r.get("player_id") and not existing.player_id:
                existing.player_id = int(r["player_id"])
                changed = True
            if float(existing.price) != float(r["price"]):
                existing.price = float(r["price"])
                changed = True
            existing.last_seen = now
            if changed:
                db.add(existing)
            upserts += 1
        else:
            db.add(
                PlayerOdds(
                    fixture_id=fixture_id,
                    player_id=int(r["player_id"]) if r.get("player_id") else None,
                    player_name=r["player_name"],
                    market=r["market"],
                    line=r["line"],
                    bookmaker=r["bookmaker"],
                    price=r["price"],
                    last_seen=now,
                )
            )
            upserts += 1

    db.commit()
    print(
        f"[player_props] upserted={upserts} fixture_id={fixture_id} provider_fixture_id={provider_fixture_id}"
    )
    return upserts