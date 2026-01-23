# api/app/services/player_props.py
from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re
import math

from sqlalchemy.orm import Session

from ..models import PlayerOdds, Fixture, PlayerSeasonStats
from .apifootball import _get, BASE_URL, _get_meta, get_fixture_players
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
# Fixture roster mapping: name -> player_id (with variants)
# ---------------------------------------------------------------------

def _name_variants(full_name: str) -> List[str]:
    """
    Produce reasonable match keys for a roster name.
    Example: "Alessandro Bastoni" ->
      ["alessandro bastoni", "a bastoni", "a. bastoni", "bastoni"]
    """
    nm = _norm_name(_clean_player_name(full_name))
    if not nm:
        return []

    parts = [p for p in nm.split(" ") if p]
    if not parts:
        return [nm]

    first = parts[0]
    last = parts[-1]

    out = {nm, last}

    # first initial + last
    if first:
        out.add(f"{first[0]} {last}")
        out.add(f"{first[0]}. {last}")

    # first + last (even if middle names exist)
    if len(parts) >= 2:
        out.add(f"{parts[0]} {parts[-1]}")

    return list(out)


def _fixture_player_name_map(provider_fixture_id: int) -> Dict[str, int]:
    """
    Call API-Football /fixtures/players and build a map {normalized_name_variant: player_id}.
    This solves cases where odds payload doesn't include player.id and only provides a name.
    """
    try:
        j = get_fixture_players(int(provider_fixture_id)) or {}
        resp = j.get("response") or []
        if not isinstance(resp, list):
            return {}
    except Exception as e:
        print(f"[player_props] get_fixture_players error: {e}")
        return {}

    out: Dict[str, int] = {}
    for team_block in resp:
        players = (team_block or {}).get("players") or []
        if not isinstance(players, list):
            continue
        for p in players:
            pinfo = (p or {}).get("player") or {}
            pid = _safe_int(pinfo.get("id"))
            name = _clean_player_name((pinfo.get("name") or "").strip())
            if not pid or not name:
                continue

            for v in _name_variants(name):
                out[v] = int(pid)

    return out


def _first_initial_and_last(nm: str) -> Tuple[str, str]:
    parts = [p for p in _norm_name(nm).split(" ") if p]
    if not parts:
        return "", ""
    first = parts[0]
    last = parts[-1]
    return (first[0] if first else ""), last


def _resolve_player_id(player_id: Optional[int], player_name: str, name_map: Dict[str, int]) -> Optional[int]:
    """
    Resolution strategy:
      1) keep provided id
      2) exact match on normalized name
      3) try variants implicitly (since name_map already contains variants)
      4) fallback: surname + first initial match
    """
    if player_id:
        return player_id
    if not player_name:
        return None

    nm = _norm_name(_clean_player_name(player_name))
    if nm in name_map:
        return name_map[nm]

    # fallback: initial + last
    ini, last = _first_initial_and_last(player_name)
    if ini and last:
        k1 = f"{ini} {last}"
        k2 = f"{ini}. {last}"
        if k1 in name_map:
            return name_map[k1]
        if k2 in name_map:
            return name_map[k2]

    # fallback: surname only
    if last and last in name_map:
        return name_map[last]

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

    provider_fixture_id = int(fixture.provider_fixture_id) if fixture.provider_fixture_id else None
    name_map: Dict[str, int] = _fixture_player_name_map(provider_fixture_id) if provider_fixture_id else {}

    # Optional debug: prove roster has Bastoni
    # print("[player_props] roster size:", len(name_map))
    # print("[player_props] roster keys w/ bastoni:", [k for k in name_map.keys() if "bastoni" in k][:10])

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
                    player_id = _resolve_player_id(pid0, player_name, name_map)

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

    def _base_q():
        q = db.query(PlayerOdds).filter(
            PlayerOdds.fixture_id == fixture_id,
            PlayerOdds.market == r["market"],
            PlayerOdds.bookmaker == r["bookmaker"],
        )
        # line nullable-safe (explicit, avoids any weird float/NULL behaviour)
        if r["line"] is None:
            q = q.filter(PlayerOdds.line.is_(None))
        else:
            q = q.filter(PlayerOdds.line == float(r["line"]))
        return q

    for r in rows:
        pid = r.get("player_id")
        pname = r["player_name"]

        existing_pid = None
        existing_name = None

        # 1) Prefer PID-key match when we have a resolved player_id
        if pid:
            existing_pid = _base_q().filter(PlayerOdds.player_id == int(pid)).one_or_none()

        # 2) Fallback to name-key match
        existing_name = _base_q().filter(PlayerOdds.player_name == pname).one_or_none()

        # 3) If both exist and are different rows, keep PID row, delete legacy name row
        if existing_pid and existing_name and existing_pid.id != existing_name.id:
            # update pid row with latest info
            changed = False
            if float(existing_pid.price) != float(r["price"]):
                existing_pid.price = float(r["price"])
                changed = True

            existing_pid.last_seen = now

            # optional: unify the displayed name to latest provider name
            # (or keep existing_pid.player_name if you want it stable)
            if pname and existing_pid.player_name != pname:
                existing_pid.player_name = pname
                changed = True

            if changed:
                db.add(existing_pid)

            # delete the legacy row (often the NULL player_id row)
            db.delete(existing_name)

            upserts += 1
            continue

        # 4) If we have a PID match, update it
        if existing_pid:
            changed = False

            # keep the name fresh
            if pname and existing_pid.player_name != pname:
                existing_pid.player_name = pname
                changed = True

            if float(existing_pid.price) != float(r["price"]):
                existing_pid.price = float(r["price"])
                changed = True

            existing_pid.last_seen = now

            if changed:
                db.add(existing_pid)

            upserts += 1
            continue

        # 5) Else if name row exists, update it and upgrade player_id if we can
        if existing_name:
            changed = False

            if pid and not existing_name.player_id:
                existing_name.player_id = int(pid)
                changed = True

            if float(existing_name.price) != float(r["price"]):
                existing_name.price = float(r["price"])
                changed = True

            existing_name.last_seen = now

            if changed:
                db.add(existing_name)

            upserts += 1
            continue

        # 6) Else insert new row
        db.add(
            PlayerOdds(
                fixture_id=fixture_id,
                player_id=int(pid) if pid else None,
                player_name=pname,
                market=r["market"],
                line=(float(r["line"]) if r["line"] is not None else None),
                bookmaker=r["bookmaker"],
                price=float(r["price"]),
                last_seen=now,
            )
        )
        upserts += 1

    db.commit()
    print(
        f"[player_props] upserted={upserts} fixture_id={fixture_id} provider_fixture_id={provider_fixture_id}"
    )
    return upserts