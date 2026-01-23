# api/app/services/player_props.py
from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re
import math

from sqlalchemy.orm import Session

from ..models import PlayerOdds, Fixture, PlayerSeasonStats
from .apifootball import _get, BASE_URL, _get_meta
from .player_model import prob_over_xpoint5, fair_odds as _fair_odds


# ---------------------------------------------------------------------
# Market mapping / normalization
# ---------------------------------------------------------------------

# canonical internal keys (what we store in PlayerOdds.market)
# NOTE: we keep these lowercase and stable.
MARKET_ALIASES: Dict[str, str] = {
    # shots
    "shots on target": "sot",
    "player shots on target": "sot",
    "sot": "sot",

    "shots": "shots",
    "player shots": "shots",

    # fouls
    "fouls": "fouls",
    "player fouls": "fouls",

    # cards
    "yellow cards": "yellow",
    "player to be booked": "yellow",
    "red cards": "red",

    # passing / assists
    "player assists": "assists",
    "assists": "assists",

    # scorers
    "anytime goal scorer": "anytime_goalscorer",
    "first goal scorer": "first_goalscorer",
    "last goal scorer": "last_goalscorer",
}

BUCKET_MARKETS: Dict[str, str] = {
    "player singles": "player_singles",
    "player doubles": "player_doubles",
    "player triples": "player_triples",
}

# finds 4.5 or -0.5 anywhere (handles "Bastoni -0.5", "Bastoni 4.5", etc)
_LINE_RE = re.compile(r"(-?\d+(?:\.\d+)?)")

# some props are "line-less" (line is meaningless)
NO_LINE_MARKETS = {
    "anytime_goalscorer",
    "first_goalscorer",
    "last_goalscorer",
    "assists",  # depends on provider; treat as line-less for now
}


def _canon_market(name: str | None) -> Optional[str]:
    """
    Make this defensive:
      - case-insensitive
      - tolerate extra whitespace
      - accept partial matches for common variants
    """
    if not name:
        return None

    s = " ".join(str(name).strip().split()).lower()  # normalize whitespace + case

    # exact hits
    if s in MARKET_ALIASES:
        return MARKET_ALIASES[s]
    if s in BUCKET_MARKETS:
        return BUCKET_MARKETS[s]

    # fuzzy matches (API-Football can vary)
    if "shots on target" in s:
        return "sot"
    if s.endswith("shots") or "player shots" in s:
        return "shots"
    if "foul" in s:
        return "fouls"
    if "to be booked" in s or "yellow" in s:
        return "yellow"
    if "anytime goal" in s:
        return "anytime_goalscorer"
    if "first goal" in s:
        return "first_goalscorer"
    if "last goal" in s:
        return "last_goalscorer"

    return None


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
    Typical API-Football 'values' for bucket markets look like:
      "Alessandro Bastoni 4.5"
      "Alessandro Bastoni -0.5"

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


# ---------------------------------------------------------------------
# API call (player odds)
# ---------------------------------------------------------------------
def fetch_player_odds_raw_for_fixture(db: Session, fixture_id: int) -> dict:
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).one_or_none()
    if not fx or not fx.provider_fixture_id:
        return {"error": "fixture missing provider_fixture_id"}

    provider_fixture_id = int(fx.provider_fixture_id)
    url = f"{BASE_URL}/odds"
    params = {"fixture": provider_fixture_id, "type": "player"}
    return _get_meta(url, params)
    
def fetch_player_odds_for_fixture(provider_fixture_id: int) -> List[Dict[str, Any]]:
    url = f"{BASE_URL}/odds"
    params = {"fixture": provider_fixture_id, "type": "player"}
    payload = _get(url, params) or {}

    if isinstance(payload, list):
        return payload

    resp = payload.get("response") or []
    return resp if isinstance(resp, list) else []


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
        "drawn_fouls": [("fouls", "drawn"), ("fouls_drawn",)],

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
    """
    Best-effort expected minutes for the match.
    If you have season totals: avg minutes per appearance.
    If not: default 80.
    """
    payload = _get_latest_player_stats(db, player_id, season=season)
    if payload is None:
        return 80

    # walk blocks to find games.minutes and games.appearences if present
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
    to the bookmaker implied probability.
    """
    if not player_id:
        return None

    season = fixture.kickoff_utc.year if fixture.kickoff_utc else None
    expected_minutes = _expected_minutes_from_cached(db, player_id, season=season)

    candidates = ["shots", "sot", "fouls", "tackles", "passes", "interceptions", "key_passes"]

    implied = 1.0 / float(price) if price and price > 0 else None
    if implied is None:
        return None

    best_key = None
    best_err = 1e9

    x_half = float(line)
    if x_half < -0.5:
        return None

    for stat_key in candidates:
        per90 = _per90_from_cached(db, player_id, season=season, stat_key=stat_key)
        if per90 is None:
            continue

        p_model = prob_over_xpoint5(per90=per90, expected_minutes=expected_minutes, x_half=x_half)
        p_model = max(1e-6, min(1 - 1e-6, float(p_model)))

        # compare probability distance (more stable than odds distance)
        err = abs(p_model - implied)

        if err < best_err:
            best_err = err
            best_key = stat_key

    # confidence gate (tune later)
    if best_key is None:
        return None

    # if we're still miles away, don't guess
    if best_err > 0.22 and price < 10:
        return None

    return best_key


# ---------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------

def _extract_player_rows(db: Session, fixture: Fixture, api_response: List[dict]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(api_response, list):
        return rows

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
                raw_market = (bet.get("name") or "").strip()
                market = _canon_market(raw_market)
                if not market:
                    continue

                values = bet.get("values") or []
                if not isinstance(values, list):
                    continue

                for v in values:
                    price = _safe_float(v.get("odd") or v.get("price"))
                    if not price or price <= 0:
                        continue

                    pinfo = v.get("player") or {}
                    player_id = _safe_int(pinfo.get("id") or v.get("id") or v.get("player_id"))

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

                    if line is None:
                        line = _parse_line(v.get("handicap") or v.get("line"))

                    # keep None for line-less markets
                    if market in NO_LINE_MARKETS:
                        line = None

                    canonical_market = market
                    inferred_from_bucket = None

                    # infer bucket stats -> replace market with inferred canonical stat key
                    if market in {"player_singles", "player_doubles", "player_triples"} and line is not None:
                        inferred = _infer_bucket_stat(
                            db=db,
                            fixture=fixture,
                            player_id=player_id,
                            player_name=player_name,
                            line=float(line),
                            price=float(price),
                        )
                        if inferred:
                            inferred_from_bucket = market
                            canonical_market = inferred  # e.g. "shots" / "fouls" / etc

                    rows.append(
                        {
                            "player_id": player_id,
                            "player_name": player_name,
                            "market": canonical_market,
                            "line": (float(line) if line is not None else None),
                            "bookmaker": bookmaker,
                            "price": float(price),
                            "raw_market": raw_market,
                            "bucket": inferred_from_bucket,  # helps debug/audit inference
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
                    line=r["line"],               # can be None
                    bookmaker=r["bookmaker"],
                    price=r["price"],
                    last_seen=now,
                )
            )
            upserts += 1

    db.commit()
    print(f"[player_props] upserted={upserts} fixture_id={fixture_id} provider_fixture_id={provider_fixture_id}")
    return upserts