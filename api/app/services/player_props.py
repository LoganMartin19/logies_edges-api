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
# We ingest ONLY these IDs for player markets.
#
# NOTE: As you said, the vast majority of player shots / SOT etc are coming
# from Player Singles (215) and Player Triples (213) buckets — so those must
# be included and handled carefully.
# ---------------------------------------------------------------------

BET_ID_MAP: Dict[int, Dict[str, Any]] = {
    # scorers
    92:  {"market": "anytime_goalscorer", "line": None},
    93:  {"market": "first_goalscorer",   "line": None},
    94:  {"market": "last_goalscorer",    "line": None},
    218: {"market": "anytime_goalscorer", "line": None},  # away variant
    219: {"market": "first_goalscorer",   "line": None},  # away variant
    226: {"market": "last_goalscorer",    "line": None},  # away variant

    # cards
    102: {"market": "yellow", "line": 0.5},  # player to be booked
    251: {"market": "yellow", "line": 0.5},  # duplicate variant
    103: {"market": "red",    "line": 0.5},  # player to be sent off

    # assists / score or assist
    212: {"market": "assists",          "line": None},
    255: {"market": "assists",          "line": None},  # home variant
    256: {"market": "assists",          "line": None},  # away variant
    257: {"market": "score_or_assist",  "line": None},
    258: {"market": "score_or_assist",  "line": None},  # home variant
    259: {"market": "score_or_assist",  "line": None},  # away variant

    # buckets (critical)
    215: {"market": "player_singles", "line": "from_value"},
    213: {"market": "player_triples", "line": "from_value"},
    # optional if you want
    # 214: {"market": "player_doubles", "line": "from_value"},

    # shots / SOT totals (some books provide direct, some via buckets)
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
    """
    Normalize provider 'handicap'/'value'/'line' into float.
    Accepts '2+', '1.5', 1, None, etc.
    """
    if raw is None:
        return None
    s = str(raw).strip().replace("+", "")
    try:
        return float(s)
    except Exception:
        return None


def _split_player_and_line(value_str: str) -> Tuple[str, Optional[float]]:
    """
    Bucket markets often look like: "Alessandro Bastoni 4.5" or "Bastoni -0.5".
    We pull first numeric token as line and remove it from name.
    """
    if not value_str:
        return "", None

    m = _LINE_RE.search(value_str)
    if not m:
        return value_str.strip(), None

    line = _parse_line(m.group(1))
    player = _LINE_RE.sub("", value_str, count=1).strip()
    player = re.sub(r"\s{2,}", " ", player).strip()
    return player, line


def _norm_name(s: str) -> str:
    """
    Normalize names so odds strings match fixture roster strings:
    - lowercase
    - collapse spaces
    - remove periods
    """
    s = (s or "").strip().lower()
    s = s.replace(".", " ")
    s = re.sub(r"\s{2,}", " ", s)
    return s


# ---------------------------------------------------------------------
# API call (odds)
# ---------------------------------------------------------------------

def fetch_player_odds_raw_for_fixture(db: Session, fixture_id: int) -> dict:
    """
    Debug helper: returns the *raw* provider response (with status/errors),
    for this fixture's odds call.
    """
    fx: Fixture | None = db.query(Fixture).filter(Fixture.id == fixture_id).one_or_none()
    if not fx or not fx.provider_fixture_id:
        return {"ok": False, "error": "fixture missing provider_fixture_id"}

    provider_fixture_id = int(fx.provider_fixture_id)
    url = f"{BASE_URL}/odds"
    params = {"fixture": provider_fixture_id}  # ✅ no type=player
    return _get_meta(url, params)


def fetch_player_odds_for_fixture(provider_fixture_id: int) -> List[Dict[str, Any]]:
    """
    Fetch odds for a fixture (no type param), then extract player markets by bet.id allowlist.
    """
    url = f"{BASE_URL}/odds"
    payload = _get(url, {"fixture": provider_fixture_id}) or []
    return payload if isinstance(payload, list) else []


# ---------------------------------------------------------------------
# Fixture roster mapping: name -> player_id
# ---------------------------------------------------------------------

def _fixture_player_name_map(provider_fixture_id: int) -> Dict[str, int]:
    """
    Call API-Football /fixtures/players and build a map {normalized_name: player_id}.
    This solves cases where odds payload doesn't include player.id and only provides a name.
    """
    try:
        j = get_fixture_players(int(provider_fixture_id)) or {}
        resp = j.get("response") or []
        if not isinstance(resp, list):
            return {}
    except Exception:
        return {}

    out: Dict[str, int] = {}
    for team_block in resp:
        players = team_block.get("players") or []
        if not isinstance(players, list):
            continue
        for p in players:
            pinfo = p.get("player") or {}
            pid = _safe_int(pinfo.get("id"))
            name = (pinfo.get("name") or "").strip()
            if pid and name:
                out[_norm_name(name)] = int(pid)
    return out


def _resolve_player_id(player_id: Optional[int], player_name: str, name_map: Dict[str, int]) -> Optional[int]:
    if player_id:
        return player_id
    if not player_name:
        return None
    return name_map.get(_norm_name(player_name))


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

    per90 = (tot / mins) * 90.0
    return float(per90) if per90 >= 0 else None


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


def _player_position_from_cached(db: Session, player_id: int, season: Optional[int]) -> Optional[str]:
    """
    Best-effort infer position from cached PlayerSeasonStats.
    Returns: goalkeeper/defender/midfielder/attacker or None.
    """
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
    """
    Priors for bucket meaning, based on position + line magnitude.
    These are "soft" weights used to prevent silly inferences.
    """
    pos = (position or "").strip().lower()

    # baseline (roughly even-ish)
    w = {
        "shots": 0.20,
        "sot": 0.12,
        "fouls": 0.20,
        "tackles": 0.18,
        "passes": 0.18,
        "interceptions": 0.12,
        "key_passes": 0.10,
    }

    # position nudges
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
        # buckets for keepers are basically never meaningful for these props
        w["shots"] -= 0.20
        w["sot"] -= 0.10
        w["passes"] += 0.10  # some keeper pass markets exist, but rare here
        w["tackles"] -= 0.10
        w["fouls"] -= 0.10

    # line magnitude nudges (very rough):
    # - high lines are more plausible for passes
    # - medium for tackles/fouls
    # - low for shots/sot
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
        # line 0.5 / 1.5 / 2.5 etc
        w["shots"] += 0.10
        w["sot"] += 0.06

    # clamp to positive
    for k in list(w.keys()):
        w[k] = max(0.001, float(w[k]))

    # normalize to sum=1
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
    """
    Infer which stat a bucket market corresponds to by matching model probability
    to bookmaker implied probability, plus position/line priors.
    """
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

        # primary fit term: absolute probability error
        fit_err = abs(p_model - float(implied))

        # prior penalty: lower is better
        # (if prior is tiny, penalty is bigger)
        prior = float(pri.get(stat_key, 1e-6))
        prior_penalty = -math.log(max(prior, 1e-9))  # 0..large

        # combine: fit dominates, prior gently nudges
        # tuneable weights
        score = (fit_err * 1.0) + (prior_penalty * 0.12)

        if score < best_score:
            best_score = score
            best_key = stat_key

    # sanity gate:
    if best_key is None:
        return None

    # also block if fit is miles off (don’t force priors to guess)
    # approximate “fit” by removing priors influence:
    # if we have a good best_score but fit itself likely huge, avoid
    # (simple version: implied price not too long, but still far)
    if price < 10:
        # find best candidate fit only
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

    # build fixture roster map once (used only if odds values lack player.id)
    provider_fixture_id = int(fixture.provider_fixture_id) if fixture.provider_fixture_id else None
    name_map: Dict[str, int] = _fixture_player_name_map(provider_fixture_id) if provider_fixture_id else {}

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
                    continue  # ✅ only ingest known player bet IDs

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

                    # resolve player + line
                    if nested_name:
                        player_name = nested_name
                        line = _parse_line(v.get("handicap") or v.get("line"))
                        if line is None and embedded:
                            _, line_guess = _split_player_and_line(embedded)
                            line = line_guess
                    else:
                        player_name, line = _split_player_and_line(embedded)

                    if not player_name:
                        continue

                    # resolve player_id:
                    pid0 = _safe_int(pinfo.get("id") or v.get("id") or v.get("player_id"))
                    player_id = _resolve_player_id(pid0, player_name, name_map)

                    # apply config line policy
                    cfg_line = cfg.get("line")

                    if canonical_market in NO_LINE_MARKETS or cfg_line is None:
                        line = None
                    elif cfg_line == 0.5:
                        line = 0.5
                    else:
                        # "from_value" -> keep parsed line; fallback to 0.0
                        if line is None:
                            line = _parse_line(v.get("handicap") or v.get("line"))
                        if line is None:
                            line = 0.0

                    inferred_from_bucket = None

                    # bucket inference: singles/triples/doubles often hide shots/SOT/fouls/tackles lines
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
                            canonical_market = inferred  # e.g. "shots"/"sot"/"fouls"/...

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
                PlayerOdds.player_id == r.get("player_id"),
                PlayerOdds.player_name == r["player_name"],
                PlayerOdds.market == r["market"],
                PlayerOdds.line == r["line"],
                PlayerOdds.bookmaker == r["bookmaker"],
            )
            .one_or_none()
        )

        if existing:
            if float(existing.price) != float(r["price"]):
                existing.price = float(r["price"])
            existing.last_seen = now
            db.add(existing)
            upserts += 1
        else:
            db.add(
                PlayerOdds(
                    fixture_id=fixture_id,
                    player_id=r.get("player_id"),
                    player_name=r["player_name"],
                    market=r["market"],
                    line=r["line"],  # can be None
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