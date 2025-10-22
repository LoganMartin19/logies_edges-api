# api/app/services/apifootball.py
from __future__ import annotations
import os, time, requests
from typing import Dict, Tuple, List, Optional
from datetime import date, datetime, timezone

# ---------------- Config ----------------
BASE_URL = os.getenv("FOOTBALL_API_URL", "https://v3.football.api-sports.io")
API_URL  = f"{BASE_URL}/fixtures"
ODDS_URL = f"{BASE_URL}/odds"

# If you're using RapidAPI proxy, keep these headers. If you use direct API-Football,
# swap to: {"x-apisports-key": API_KEY}
API_KEY  = os.getenv("FOOTBALL_API_KEY")
API_HOST = os.getenv("RAPIDAPI_HOST", "v3.football.api-sports.io")
HEADERS = {
    "X-RapidAPI-Key": API_KEY or "",
    "X-RapidAPI-Host": API_HOST,
    "Accept": "application/json",
}

# ---------------- League IDs ----------------
LEAGUE_MAP = {
    # England
    "EPL": 39, "Premier League": 39,
    "CHAMP": 40, "Championship": 40,
    "LG1": 41, "League One": 41,
    "LG2": 42, "League Two": 42,
    "EFL_CUP": 185, "EFL Cup": 185,
    "EFL_TROPHY": 46, "EFL Trophy": 46,

    # Scotland
    "SCO_PREM": 179, "Scottish Premiership": 179,
    "SCO_CHAMP": 180, "Scottish Championship": 180,
    "SCO1": 181, "SCO2": 182,

    # Europe
    "LA_LIGA": 140, "La Liga": 140,
    "BUNDES": 78, "Bundesliga": 78,
    "BUNDES2": 79, "2. Bundesliga": 79,
    "SERIE_A": 135, "Serie A": 135,
    "SERIE_B": 136, "Serie B": 136,
    "LIGUE1": 61, "Ligue 1": 61,

    "UCL": 2, "UEFA Champions League": 2,
    "UEL": 3, "UEFA Europa League": 3,
    "UECL": 848, "UEFA Europa Conference League": 848,
    "UWCL": 525, "UEFA Women's Champions League": 525,

    # USA
    "MLS": 253, "Major League Soccer": 253,

    #Brazil
    "BR_SERIE_A": 71, "Brasileirão Série A": 71,
}

def canonicalize_comp(league_obj: dict) -> str:
    """
    Map an API-Football league object to our internal competition key.
    - Prefer exact ID matches in LEAGUE_MAP (e.g., 39 -> "EPL").
    - Disambiguate generic names like "Championship" by country.
    - Fallback to provider league name if unknown.
    """
    if not isinstance(league_obj, dict):
        return ""
    lid = league_obj.get("id")
    name = (league_obj.get("name") or "").strip()
    country = (league_obj.get("country") or "").strip()

    # Exact ID match against our map values
    for key, val in LEAGUE_MAP.items():
        if val == lid:
            return key

    # Disambiguate common name collisions
    if name == "Championship":
        if country == "England":
            return "CHAMP"
        if country == "Scotland":
            return "SCO_CHAMP"

    # Fallback to the provider name
    return name

# ---------------- Diagnostics ----------------
LAST_HTTP: Dict[str, Optional[object]] = {
    "url": None,
    "params": None,
    "status": None,             # int or "cached"/"error"
    "error": None,              # provider/server error string
    "response_count": None,     # len(response) list
    "cached": False,
    "ratelimit_remaining": None,
    "ratelimit_reset": None,
    "timestamp": None,          # ISO8601
}

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time()))

# ---------------- Caching + HTTP ----------------
_CACHE: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Tuple[float, List[dict]]] = {}
_CACHE_TTL = 60.0

def _cache_key(url: str, params: dict) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    return (url, tuple(sorted((k, str(v)) for k, v in (params or {}).items())))

def _sleep_for_429(r, fallback: float) -> float:
    ra = r.headers.get("Retry-After")
    if ra:
        try:
            return max(1.0, float(ra))
        except Exception:
            pass
    reset = (
        r.headers.get("X-RateLimit-Requests-Reset")
        or r.headers.get("X-RateLimit-Reset")
        or r.headers.get("x-ratelimit-requests-reset")
        or r.headers.get("x-ratelimit-reset")
    )
    if reset:
        try:
            reset_epoch = float(reset)
            delta = reset_epoch - time.time()
            return max(1.0, min(delta + 1.0, 600.0))
        except Exception:
            pass
    return min(max(fallback, 1.0), 120.0)

def _parse_iso_utc(s: str) -> Optional[datetime]:
    """
    Robust ISO parser -> timezone-aware UTC datetime.
    Accepts '2025-09-29T19:45:00Z' or with an offset, or naive (assumed UTC).
    Returns None if parsing fails.
    """
    if not s:
        return None
    try:
        # API often returns 'Z' for UTC — make it an explicit offset
        s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    # make aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _get(url: str, params: dict, retries: int = 4, backoff: float = 0.75) -> List[dict]:
    """
    Resilient GET that:
      - uses a 60s in-proc cache
      - handles 429 with reset/backoff
      - retries 5xx with backoff
      - ALWAYS returns a list ( [] on error/empty/null )
      - records LAST_HTTP for diagnostics
    """
    key = _cache_key(url, params or {})
    now = time.time()

    # Cached?
    if key in _CACHE:
        ts, data = _CACHE[key]
        if now - ts < _CACHE_TTL:
            LAST_HTTP.update({
                "url": url,
                "params": dict(params or {}),
                "status": "cached",
                "error": None,
                "response_count": len(data) if isinstance(data, list) else 0,
                "cached": True,
                "ratelimit_remaining": None,
                "ratelimit_reset": None,
                "timestamp": _now_iso(),
            })
            return data

    wait = backoff
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=20)
            status = r.status_code

            # Rate limited
            if status == 429 and attempt < retries - 1:
                LAST_HTTP.update({
                    "url": url,
                    "params": dict(params or {}),
                    "status": status,
                    "error": "429 rate_limited",
                    "response_count": 0,
                    "cached": False,
                    "ratelimit_remaining": (
                        r.headers.get("X-RateLimit-Requests-Remaining")
                        or r.headers.get("X-RateLimit-Remaining")
                    ),
                    "ratelimit_reset": (
                        r.headers.get("X-RateLimit-Requests-Reset")
                        or r.headers.get("X-RateLimit-Reset")
                    ),
                    "timestamp": _now_iso(),
                })
                time.sleep(_sleep_for_429(r, wait))
                wait = min(wait * 2, 5.0)
                continue

            # Retry 5xx
            if 500 <= status < 600 and attempt < retries - 1:
                time.sleep(wait)
                wait = min(wait * 2, 5.0)
                continue

            r.raise_for_status()

            # Parse JSON
            try:
                j = r.json() or {}
            except Exception:
                j = {}

            provider_err = None
            if isinstance(j.get("errors"), dict) and j["errors"]:
                provider_err = "; ".join(f"{k}: {v}" for k, v in j["errors"].items())
            elif isinstance(j.get("errors"), list) and j["errors"]:
                provider_err = "; ".join(map(str, j["errors"]))
            elif isinstance(j.get("message"), str) and j["message"]:
                provider_err = j["message"]

            resp = j.get("response")
            data: List[dict] = resp if isinstance(resp, list) else []

            LAST_HTTP.update({
                "url": url,
                "params": dict(params or {}),
                "status": status,
                "error": provider_err,
                "response_count": len(data),
                "cached": False,
                "ratelimit_remaining": (
                    r.headers.get("X-RateLimit-Requests-Remaining")
                    or r.headers.get("X-RateLimit-Remaining")
                ),
                "ratelimit_reset": (
                    r.headers.get("X-RateLimit-Requests-Reset")
                    or r.headers.get("X-RateLimit-Reset")
                ),
                "timestamp": _now_iso(),
            })

            if provider_err:
                return []

            _CACHE[key] = (time.time(), data)
            return data

        except Exception as e:
            if attempt == retries - 1:
                LAST_HTTP.update({
                    "url": url,
                    "params": dict(params or {}),
                    "status": getattr(e, "response", None).status_code if hasattr(e, "response") and e.response else "error",
                    "error": repr(e),
                    "response_count": 0,
                    "cached": False,
                    "ratelimit_remaining": None,
                    "ratelimit_reset": None,
                    "timestamp": _now_iso(),
                })
                break

    _CACHE[key] = (time.time(), [])
    return []

def _get_all_pages(url: str, params: dict) -> List[dict]:
    """
    Try a single fetch without 'page'. If JSON has paging.total > 1,
    continue with page=2..N (politely spaced).
    """
    out: List[dict] = list(_get(url, dict(params or {})))
    try:
        r0 = requests.get(url, headers=HEADERS, params=dict(params or {}), timeout=20)
        paging = (r0.json() or {}).get("paging", {})
        total_pages = int(paging.get("total") or 1)
    except Exception:
        total_pages = 1

    if total_pages <= 1:
        return out

    for page in range(2, total_pages + 1):
        chunk = _get(url, {**(params or {}), "page": page})
        if isinstance(chunk, list):
            out.extend(chunk)
        time.sleep(0.12)  # be polite
    return out

# ---------------- Fixtures + Odds ----------------
def fetch_fixtures(day: date, leagues: List[str]) -> List[dict]:
    wanted_ids = {LEAGUE_MAP[c] for c in leagues if c in LEAGUE_MAP}
    all_for_day = _get_all_pages(API_URL, {"date": day.isoformat()})
    if not wanted_ids:
        return all_for_day
    return [fx for fx in all_for_day if fx.get("league", {}).get("id") in wanted_ids]

def fetch_fixture_by_id(fixture_id: int) -> List[dict]:
    return _get(API_URL, {"id": int(fixture_id)})

def fetch_fixtures_by_date(date_str: Optional[str] = None, league_id: Optional[int] = None) -> List[dict]:
    params = {}
    if date_str:
        params["date"] = date_str
    if league_id is not None:
        params["league"] = int(league_id)
    return _get_all_pages(API_URL, params)

def fetch_fixtures_by_league_and_date(league_id: int, day: date) -> List[dict]:
    return _get_all_pages(API_URL, {"date": day.isoformat(), "league": int(league_id)})

def fetch_fixtures_all_pages_for_date(day: date) -> List[dict]:
    return _get_all_pages(API_URL, {"date": day.isoformat()})

def fetch_odds_for_fixture(fixture_id: int) -> List[dict]:
    return _get(ODDS_URL, {"fixture": fixture_id})

# ---------------- Standings / Strength ----------------
def get_standings_for_league(league_id: int, season: int = 2025):
    """
    Returns:
      strength_map: {team -> strength score (simple rank-based proxy)}
      table: standings rows with position/points/W/D/L/GF/GA/form/strength
    """
    res = _get(f"{BASE_URL}/standings", {"league": league_id, "season": season})
    if not res:
        return {}, []
    try:
        standings = res[0]["league"]["standings"][0]
    except Exception:
        return {}, []

    strength_map: Dict[str, float] = {}
    num_teams = max(1, len(standings))
    for i, entry in enumerate(standings):
        team = entry["team"]["name"]
        strength_score = 1.0 - (i / (num_teams - 1)) * 0.95 if num_teams > 1 else 1.0
        strength_map[team] = round(strength_score, 3)

    table = [{
        "position": e["rank"],
        "team": e["team"]["name"],
        "points": e["points"],
        "played": e["all"]["played"],
        "win": e["all"]["win"],
        "draw": e["all"]["draw"],
        "lose": e["all"]["lose"],
        "gf": e["all"]["goals"]["for"],
        "ga": e["all"]["goals"]["against"],
        "form": e.get("form"),
        "strength": strength_map[e["team"]["name"]],
    } for e in standings]

    return strength_map, table

# ---------------- Core JSON passthroughs used elsewhere ----------------
def get_fixture(fixture_id: int):
    """Raw fixture details by provider fixture ID (JSON passthrough)."""
    return requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params={"id": fixture_id}).json()

def get_players(team_id: int, league_id: Optional[int], season: int):
    """
    Player-level stats (cards, shots, fouls etc).
    If league_id is None, API-Football returns all competitions for that season.
    """
    params = {"team": team_id, "season": season}
    if league_id is not None:
        params["league"] = league_id
    return requests.get(f"{BASE_URL}/players", headers=HEADERS, params=params).json()

def get_predictions(fixture_id: int):
    return requests.get(f"{BASE_URL}/predictions", headers=HEADERS, params={"fixture": fixture_id}).json()

def get_lineups(fixture_id: int):
    return requests.get(f"{BASE_URL}/fixtures/lineups", headers=HEADERS, params={"fixture": fixture_id}).json()

def get_h2h(team1: int, team2: int):
    return requests.get(f"{BASE_URL}/fixtures/headtohead", headers=HEADERS, params={"h2h": f"{team1}-{team2}"}).json()

def get_team_stats(team_id: int, league_id: int, season: int):
    return requests.get(
        f"{BASE_URL}/teams/statistics",
        headers=HEADERS,
        params={"team": team_id, "league": league_id, "season": season},
    ).json()

def get_top_scorers(league: int, season: int):
    return requests.get(f"{BASE_URL}/players/topscorers", headers=HEADERS, params={"league": league, "season": season}).json()

def get_injuries(team_id: int, league_id: int, season: int):
    return requests.get(
        f"{BASE_URL}/injuries",
        headers=HEADERS,
        params={"team": team_id, "league": league_id, "season": season},
    ).json()

def get_events(fixture_id: int):
    # Using same BASE_URL+headers so it stays consistent with the rest
    return requests.get(f"{BASE_URL}/fixtures/events", headers=HEADERS, params={"fixture": fixture_id}).json()

# ---------------- Form / Recent fixtures ----------------
def get_team_recent_results(
    team_id: int,
    season: int = 2025,
    limit: int = 5,
    *,
    before_iso: Optional[str] = None,
    league_id: Optional[int] = None,
) -> List[dict]:
    """
    Last N finished fixtures for a team (with score, opponent, result).
    Optional:
      - before_iso: only include matches strictly before this ISO8601 timestamp
      - league_id: restrict to a specific league id
    Returns a list of normalized dicts incl. comp_key (our internal league key).
    """
    params = {"team": team_id, "season": season, "status": "FT"}
    if league_id is not None:
        params["league"] = league_id

    matches = _get(f"{BASE_URL}/fixtures", params) or []

    # optional cutoff
    cutoff = _parse_iso_utc(before_iso) if before_iso else None

    cleaned: List[dict] = []
    for m in matches:
        fx = m.get("fixture", {}) or {}
        lg = m.get("league", {}) or {}
        teams = m.get("teams", {}) or {}
        goals = m.get("goals", {}) or {}

        dt = _parse_iso_utc(fx.get("date"))
        if cutoff and (dt is None or not (dt < cutoff)):
            continue

        if league_id is not None and lg.get("id") != league_id:
            continue

        home = teams.get("home", {}).get("name")
        away = teams.get("away", {}).get("name")
        is_home = teams.get("home", {}).get("id") == team_id

        gh = goals.get("home") or 0
        ga = goals.get("away") or 0
        gf, ga_ = (gh, ga) if is_home else (ga, gh)
        result = "win" if gf > ga_ else ("loss" if gf < ga_ else "draw")

        cleaned.append({
            "date": (dt.isoformat().replace("+00:00", "Z") if dt else fx.get("date")),
            "opponent": away if is_home else home,
            "is_home": is_home,
            "score": f"{gh}-{ga}",
            "result": result,
            "fixture_id": fx.get("id"),
            "league_id": lg.get("id"),
            "league_name": lg.get("name"),
            "league_country": lg.get("country"),
            # 🔑 internal key for filtering/joins (e.g. UCL, EPL, CHAMP, etc.)
            "comp_key": canonicalize_comp(lg),
        })

    cleaned.sort(key=lambda x: x["date"] or "", reverse=True)
    return cleaned[:limit]

# ---------------- Player props (cards, shots, fouls, etc.) ----------------
def get_player_stats(team_id: int, league_id: int, season: int = 2025) -> List[dict]:
    """
    Player-level stats (cards, shots, fouls etc) for a team in league/season.
    Uses paging aggregator to fetch all players.
    """
    return _get_all_pages(f"{BASE_URL}/players", {"team": team_id, "league": league_id, "season": season})

def get_fixture_fouls_drawn(fixture_id: int, team_id: int) -> float:
    """
    Fouls drawn by team_id in THIS fixture.
    We count 'Foul' events committed by the OPPONENT (team != team_id).
    Returns raw count (per-match ≈ per90 at team level).
    """
    try:
        j = requests.get(
            f"{BASE_URL}/fixtures/events",
            headers=HEADERS,
            params={"fixture": int(fixture_id)},
            timeout=20
        ).json() or {}
        events = j.get("response") or []
        if not isinstance(events, list):
            return 0.0
    except Exception as e:
        print(f"[get_fixture_fouls_drawn] error: {e}")
        return 0.0

    drawn = 0
    for ev in events:
        # team.id is the team associated with the event (for 'Foul', this is the committing team)
        ev_team_id = (ev.get("team") or {}).get("id")
        ev_type = ev.get("type")
        if ev_type == "Foul" and ev_team_id and ev_team_id != team_id:
            drawn += 1

    return float(drawn)

# api/app/routes/football_extra.py
from fastapi import APIRouter, Query, HTTPException, Depends
from sqlalchemy.orm import Session
from ..db import get_db
from ..models import Fixture, PlayerOdds
from ..services.apifootball import (
    get_predictions,
    get_lineups,
    get_h2h,
    get_team_stats,
    get_top_scorers,
    get_players,
    get_injuries,
    get_events,
    get_fixture,
    get_player_stats,
    get_fixture_fouls_drawn,   # 🔑 new helper
)
from ..services.player_odds import ingest_player_odds
from ..services.player_model import prob_over_xpoint5, prob_card, fair_odds, edge

router = APIRouter(prefix="/football", tags=["football"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_provider_fixture_id(db: Session, fixture_id: int) -> int:
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx or not fx.provider_fixture_id:
        raise HTTPException(status_code=404, detail="Fixture not found")
    return int(fx.provider_fixture_id)


def _get_player_props_data(fixture_id: int, db: Session) -> dict:
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx or not fx.provider_fixture_id:
        raise HTTPException(status_code=404, detail="Fixture not found")

    pfx = int(fx.provider_fixture_id)
    fx_json = get_fixture(pfx)
    if not fx_json.get("response"):
        raise HTTPException(status_code=404, detail="Fixture detail not found")
    fr = fx_json["response"][0]

    league_id = fr["league"]["id"]
    season    = int(fr["league"]["season"])
    home_id   = fr["teams"]["home"]["id"]
    away_id   = fr["teams"]["away"]["id"]
    league_name_lc = (fr["league"]["name"] or "").strip().lower()

    def _pick_block(stat_blocks: list[dict]) -> dict | None:
        for s in stat_blocks or []:
            lg = (s.get("league", {}) or {})
            if int(lg.get("id") or 0) == int(league_id):
                return s
        for s in stat_blocks or []:
            lg = (s.get("league", {}) or {})
            if (lg.get("name") or "").strip().lower() == league_name_lc:
                return s
        return None

    def _flatten(items: list[dict]) -> list[dict]:
        out = []
        for row in items or []:
            player = row.get("player", {}) or {}
            stats  = row.get("statistics") or []
            s = _pick_block(stats)
            if not s:
                continue

            games = s.get("games", {}) or {}
            shots = s.get("shots", {}) or {}
            cards = s.get("cards", {}) or {}
            fouls = s.get("fouls", {}) or {}

            name = player.get("name") or "—"
            mins = int(games.get("minutes") or 0)

            sh_total = int(shots.get("total") or 0)
            sh_on    = int(shots.get("on") or 0)
            sot_pct  = round((sh_on / sh_total * 100.0), 1) if sh_total else 0.0

            fouls_comm = int((fouls.get("committed") or 0) or 0)

            per90 = (lambda v: round((v * 90.0) / mins, 2) if mins else 0.0)

            out.append({
                "id": player.get("id"),
                "name": name,
                "photo": player.get("photo"),
                "pos": games.get("position") or player.get("position") or "?",
                "minutes": mins,
                "shots": sh_total,
                "shots_on": sh_on,
                "sot_pct": sot_pct,
                "yellow": int(cards.get("yellow") or 0),
                "red":    int(cards.get("red") or 0),
                "fouls_committed": fouls_comm,
                "shots_per90": per90(sh_total),
                "fouls_committed_per90": per90(fouls_comm),
            })
        out.sort(key=lambda r: (r["minutes"], r["shots"], r["yellow"]), reverse=True)
        return out

    def normalize_team(team_id: int) -> list[dict]:
        rows = get_player_stats(team_id, league_id, season)
        flat = _flatten(rows if isinstance(rows, list) else [])
        if any(r["minutes"] or r["shots"] or r["yellow"] for r in flat):
            return flat

        prev = get_player_stats(team_id, league_id, season - 1)
        flat_prev = _flatten(prev if isinstance(prev, list) else [])
        if any(r["minutes"] or r["shots"] or r["yellow"] for r in flat_prev):
            return flat_prev

        from ..services.apifootball import _get_all_pages, BASE_URL
        all_comp = _get_all_pages(f"{BASE_URL}/players", {"team": team_id, "season": season})
        return _flatten(all_comp)

    return {
        "league_id": league_id,
        "season": season,
        "home_team": fx.home_team,
        "away_team": fx.away_team,
        "home": normalize_team(home_id),
        "away": normalize_team(away_id),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/player-props/fair")
def player_props_fair(
    fixture_id: int,
    team: str | None = Query(None, description="home|away"),
    markets: str | None = Query(None, description="CSV of markets"),
    min_prob: float = Query(0.0, ge=0.0, le=1.0),
    minutes: int | None = Query(None, ge=10, le=120),
    opponent_adj: bool = Query(True, description="apply opponent fouls-drawn bump"),
    ref_adj: bool = Query(True, description="apply referee cards bump (if known)"),
    ref_factor_override: float | None = Query(None, ge=0.5, le=1.5),
    db: Session = Depends(get_db),
):
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx:
        raise HTTPException(status_code=404, detail="Fixture not found")

    try:
        pfx = int(fx.provider_fixture_id)
    except Exception:
        pfx = None

    league_id = season = home_pid = away_pid = None
    referee_name = (fx.referee or "").strip()

    if pfx:
        try:
            fjson = get_fixture(pfx)
            core = (fjson.get("response") or [None])[0] or {}
            lg = core.get("league") or {}
            league_id = lg.get("id")
            season = lg.get("season")
            teams = core.get("teams") or {}
            home_pid = (teams.get("home") or {}).get("id")
            away_pid = (teams.get("away") or {}).get("id")
            if not referee_name:
                referee_name = (core.get("fixture") or {}).get("referee") or ""
        except Exception:
            pass

    # 🔑 Opponent context from events API
    home_drawn90 = get_fixture_fouls_drawn(pfx, home_pid) if pfx and home_pid else 0.0
    away_drawn90 = get_fixture_fouls_drawn(pfx, away_pid) if pfx and away_pid else 0.0

    LEAGUE_FOULS_DRAWN_AVG = 10.0

    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def opponent_fouls_factor(opponent_drawn90: float) -> float:
        if not opponent_adj:
            return 1.0
        if opponent_drawn90 <= 0:
            return 1.0
        return clamp(opponent_drawn90 / LEAGUE_FOULS_DRAWN_AVG, 0.80, 1.25)

    def referee_cards_factor() -> float:
        if not ref_adj:
            return 1.0
        if ref_factor_override:
            return ref_factor_override
        return 1.0

    stats_data = _get_player_props_data(fixture_id, db)

    stored_odds = db.query(PlayerOdds).filter(PlayerOdds.fixture_id == fixture_id).all()
    odds_map: dict[tuple[int, str, float], dict] = {}
    for o in stored_odds:
        key = (int(o.player_id), (o.market or "").lower(), float(o.line or 0.0))
        best = odds_map.get(key)
        if not best or float(o.price) > best["price"]:
            odds_map[key] = {"bookmaker": o.bookmaker, "price": float(o.price)}

    team_norm = (team or "").strip().lower()
    want_team = team_norm in {"home", "away"}
    market_set = set(m.strip().lower() for m in (markets or "").split(",") if m and m.strip())

    out = {"fixture_id": fixture_id, "props": []}

    for side in ("home", "away"):
        if want_team and side != team_norm:
            continue

        roster = stats_data.get(side, []) or []
        opp_drawn90 = away_drawn90 if side == "home" else home_drawn90
        fouls_ctx = opponent_fouls_factor(opp_drawn90)
        ref_ctx = referee_cards_factor()

        for pl in roster:
            mins_played = int(pl.get("minutes") or 0)
            m_used = minutes or (80 if mins_played >= 600 else 30)

            shots_per90 = float(pl.get("shots_per90") or 0.0)
            fouls90 = float(pl.get("fouls_committed_per90") or 0.0)

            if mins_played > 0:
                sot_per90 = (float(pl.get("shots_on") or 0.0) * 90.0) / mins_played
                cards_per90 = (float(pl.get("yellow") or 0.0) * 90.0) / mins_played
            else:
                sot_per90 = 0.0
                cards_per90 = 0.0

            p_shots15 = prob_over_xpoint5(shots_per90, m_used, 1.5)
            p_sot05 = prob_over_xpoint5(sot_per90, m_used, 0.5)
            p_fouls05 = prob_over_xpoint5(fouls90, m_used, 0.5, opponent_factor=fouls_ctx)
            p_card = prob_card(cards_per90, m_used, ref_factor=ref_ctx, opponent_factor=fouls_ctx)

            markets_calc = [
                ("shots_over_1.5", 1.5, p_shots15, fair_odds(p_shots15)),
                ("sot_over_0.5", 0.5, p_sot05, fair_odds(p_sot05)),
                ("fouls_over_0.5", 0.5, p_fouls05, fair_odds(p_fouls05)),
                ("to_be_booked", 0.5, p_card, fair_odds(p_card)),
            ]

            for market, line, prob, fair in markets_calc:
                if prob < min_prob:
                    continue
                if market_set and market not in market_set:
                    continue

                key = (int(pl["id"]), market, float(line))
                bm = odds_map.get(key)

                out["props"].append({
                    "player_id": int(pl["id"]),
                    "player": pl.get("name") or "",
                    "team_side": side,
                    "market": market,
                    "line": float(line),
                    "proj_minutes": int(m_used),
                    "prob": float(prob),
                    "fair_odds": float(fair) if fair else None,
                    "best_price": bm["price"] if bm else None,
                    "bookmaker": bm["bookmaker"] if bm else None,
                    "edge": edge(prob, bm["price"]) if bm and fair else None,
                    "per90_shots": round(shots_per90, 2),
                    "per90_sot": round(sot_per90, 2),
                    "per90_fouls": round(fouls90, 2),
                    "cards_per90": round(cards_per90, 2),
                    "opp_fouls_drawn_per90": round(opp_drawn90, 2),
                    "opponent_factor": round(fouls_ctx, 3),
                    "ref_factor": round(ref_ctx, 3),
                })

    out["props"].sort(key=lambda r: (float(r.get("edge") or 0.0), float(r["prob"])), reverse=True)
    return out

def get_fixture_players(fixture_id: int):
    """
    API-Football: /fixtures/players?fixture={id}
    Returns { response: [ { team:{...}, players:[ { player:{...}, statistics:[...] } ] } ] }
    """
    url = f"{BASE_URL}/fixtures/players"
    r = requests.get(url, headers=HEADERS, params={"fixture": int(fixture_id)}, timeout=30)
    r.raise_for_status()
    return r.json()

# --- Opponent context averages (events-first, stats fallback) ----------------

def _events_fouls_committed_in_fixture(fixture_id: int, team_id: int) -> int:
    """Count fouls COMMITTED by team_id in a single fixture using events."""
    try:
        j = requests.get(
            f"{BASE_URL}/fixtures/events",
            headers=HEADERS,
            params={"fixture": int(fixture_id)},
            timeout=20
        ).json() or {}
        events = j.get("response") or []
        if not isinstance(events, list):
            return 0
    except Exception:
        return 0

    committed = 0
    for ev in events:
        if ev.get("type") == "Foul":
            ev_team_id = (ev.get("team") or {}).get("id")
            if ev_team_id and ev_team_id == team_id:  # committing team
                committed += 1
    return committed


def get_team_fouls_committed_avg(team_id: int, season: int, league_id: int | None = None, lookback: int = 5) -> float:
    """
    Average fouls COMMITTED per match by team_id over last `lookback` fixtures (events-first).
    Fallback: /teams/statistics fouls.committed.total / fixtures.played.total
    """
    # 1) Try events over last N finished fixtures
    recent = get_team_recent_results(team_id, season=season, limit=lookback, league_id=league_id) or []
    ev_counts = []
    for fx in recent:
        fid = fx.get("fixture_id")
        if not fid:
            continue
        c = _events_fouls_committed_in_fixture(int(fid), team_id)
        # Some comps have no events; skip zeros ONLY if there are no events at all:
        ev_counts.append(c)

    # If we got at least one non-None entry, compute average (zeros are legit if events existed)
    if ev_counts and any(x is not None for x in ev_counts):
        denom = max(1, len(ev_counts))
        return float(sum(ev_counts)) / float(denom)

    # 2) Fallback to team stats if events were empty or unavailable
    try:
        if league_id is not None:
            stats = get_team_stats(team_id, league_id, season) or {}
            r = stats.get("response") or {}
            fouls = (r.get("fouls") or {})
            committed_total = int(((fouls.get("committed") or {}).get("total")) or 0)
            played = int(((r.get("fixtures") or {}).get("played") or {}).get("total") or 0)
            return (committed_total / played) if played else 0.0
    except Exception:
        pass

    return 0.0


def get_team_fouls_drawn_avg(team_id: int, season: int, league_id: int | None = None, lookback: int = 5) -> float:
    """
    Average fouls DRAWN per match by team_id over last `lookback` fixtures.
    For each fixture: fouls drawn by team A == fouls COMMITTED by the opponent.
    We compute it directly from events by counting fouls where event.team.id != team_id.
    Fallback: if events missing, approximate using opponent-committed via team stats.
    """
    # 1) Events-based over last N fixtures
    recent = get_team_recent_results(team_id, season=season, limit=lookback, league_id=league_id) or []
    drawn_counts = []
    for fx in recent:
        fid = fx.get("fixture_id")
        if not fid:
            continue
        try:
            j = requests.get(
                f"{BASE_URL}/fixtures/events",
                headers=HEADERS,
                params={"fixture": int(fid)},
                timeout=20
            ).json() or {}
            events = j.get("response") or []
            if not isinstance(events, list):
                continue
        except Exception:
            continue

        drawn = 0
        for ev in events:
            if ev.get("type") == "Foul":
                ev_team_id = (ev.get("team") or {}).get("id")
                # opponent committed -> our team drew
                if ev_team_id and ev_team_id != team_id:
                    drawn += 1
        drawn_counts.append(drawn)

    if drawn_counts and any(x is not None for x in drawn_counts):
        return float(sum(drawn_counts)) / float(len(drawn_counts))

    # 2) Fallback: approximate with our opponents' committed via stats is messy without opponent IDs here,
    # so fall back to our own team-stats if provider exposes 'drawn'. If not, return 0.
    try:
        if league_id is not None:
            stats = get_team_stats(team_id, league_id, season) or {}
            r = stats.get("response") or {}
            fouls = (r.get("fouls") or {})
            drawn_total = int(((fouls.get("drawn") or {}).get("total")) or 0)
            played = int(((r.get("fixtures") or {}).get("played") or {}).get("total") or 0)
            return (drawn_total / played) if played else 0.0
    except Exception:
        pass

    return 0.0

# --- NEW: single-fixture statistics (shots, SoT, fouls, xG, etc.) ---
def get_fixture_statistics(fixture_id: int):
    """Raw fixture stats (shots, SoT, fouls, xG, etc.) for a single fixture."""
    return requests.get(
        f"{BASE_URL}/fixtures/statistics",
        headers=HEADERS,
        params={"fixture": int(fixture_id)},
        timeout=20,
    ).json()

def _read_stat(stats_list, *aliases) -> float:
    """
    Find a numeric value in API-Football statistics array by type aliases.
    Handles numbers and strings like '52%'.
    """
    if not isinstance(stats_list, list):
        return 0.0
    al = [a.lower() for a in aliases]
    for item in stats_list:
        t = (item.get("type") or "").lower()
        if any(a in t for a in al):
            v = item.get("value")
            try:
                return float(str(v).strip("%"))
            except Exception:
                return 0.0
    return 0.0

# --- NEW: rolling averages from recent finished fixtures' statistics ---
def get_team_shots_against_avgs(
    team_id: int,
    *,
    season: int,
    league_id: int | None = None,
    lookback: int = 5,
) -> dict:
    """
    Rolling averages of shots conceded and SoT conceded over last N finished fixtures.
    We:
      1) get last N finished fixtures for team_id (optionally filtered to league_id),
      2) call /fixtures/statistics for each fixture,
      3) read the OPPONENT's 'Total Shots' & 'Shots on Goal' (i.e., shots conceded).
    """
    recent = get_team_recent_results(team_id, season=season, limit=lookback, league_id=league_id) or []
    rows = []
    tot_sh = 0.0
    tot_sot = 0.0
    counted = 0

    for fx in recent:
        fid = fx.get("fixture_id")
        if not fid:
            continue

        j = get_fixture_statistics(int(fid)) or {}
        resp = j.get("response") or []
        if not isinstance(resp, list) or len(resp) < 2:
            # some comps won’t have statistics
            continue

        # Opponent = the statistics block whose team.id != our team_id
        opp_row = next((r for r in resp if (r.get("team") or {}).get("id") != team_id), None)
        if not opp_row:
            continue

        stats = opp_row.get("statistics") or []
        opp_total_shots = _read_stat(stats, "total shots", "shots total")
        opp_sot         = _read_stat(stats, "shots on goal", "shots on target")

        rows.append({
            "fixture_id": fid,
            "date": fx.get("date"),
            "opponent": fx.get("opponent"),
            "is_home": fx.get("is_home"),
            "shots_against": opp_total_shots,
            "sot_against": opp_sot,
        })
        tot_sh  += opp_total_shots
        tot_sot += opp_sot
        counted += 1

    return {
        "matches_counted": counted,
        "shots_against_per_match": (tot_sh / counted) if counted else 0.0,
        "sot_against_per_match":   (tot_sot / counted) if counted else 0.0,
        "fixtures": rows,
    }

def get_team_xg_avgs(
    team_id: int,
    *,
    season: int,
    league_id: int | None = None,
    lookback: int = 5,
) -> dict:
    """
    Rolling averages of xG for and xG against over last N finished fixtures, using /fixtures/statistics.
    We read our own xG from our block, and xG against from opponent's block.
    """
    recent = get_team_recent_results(team_id, season=season, limit=lookback, league_id=league_id) or []
    rows = []
    tot_xg_for = 0.0
    tot_xg_against = 0.0
    counted = 0

    for fx in recent:
        fid = fx.get("fixture_id")
        if not fid:
            continue

        j = get_fixture_statistics(int(fid)) or {}
        resp = j.get("response") or []
        if not isinstance(resp, list) or len(resp) < 2:
            continue

        our_row = next((r for r in resp if (r.get("team") or {}).get("id") == team_id), None)
        opp_row = next((r for r in resp if (r.get("team") or {}).get("id") != team_id), None)
        if not our_row or not opp_row:
            continue

        our_stats = our_row.get("statistics") or []
        opp_stats = opp_row.get("statistics") or []

        xg_for     = _read_stat(our_stats, "expected_goals", "xg")
        xg_against = _read_stat(opp_stats, "expected_goals", "xg")

        rows.append({
            "fixture_id": fid,
            "date": fx.get("date"),
            "opponent": fx.get("opponent"),
            "is_home": fx.get("is_home"),
            "xg_for": xg_for,
            "xg_against": xg_against,
        })
        tot_xg_for     += xg_for
        tot_xg_against += xg_against
        counted += 1

    return {
        "matches_counted": counted,
        "xg_for_per_match": (tot_xg_for / counted) if counted else 0.0,
        "xg_against_per_match": (tot_xg_against / counted) if counted else 0.0,
        "fixtures": rows,
    }

def get_team_fouls_from_statistics_avg(
    team_id: int,
    *,
    season: int,
    league_id: int | None = None,
    lookback: int = 5,
) -> dict:
    """
    Rolling averages of fouls committed and fouls drawn using /fixtures/statistics only.
    'Fouls' in opponent's block = fouls *they* committed => fouls we drew.
    """
    recent = get_team_recent_results(team_id, season=season, limit=lookback, league_id=league_id) or []
    rows = []
    tot_comm = 0.0
    tot_drawn = 0.0
    counted = 0

    for fx in recent:
        fid = fx.get("fixture_id")
        if not fid:
            continue

        j = get_fixture_statistics(int(fid)) or {}
        resp = j.get("response") or []
        if not isinstance(resp, list) or len(resp) < 2:
            continue

        our_row = next((r for r in resp if (r.get("team") or {}).get("id") == team_id), None)
        opp_row = next((r for r in resp if (r.get("team") or {}).get("id") != team_id), None)
        if not our_row or not opp_row:
            continue

        our_fouls = _read_stat(our_row.get("statistics") or [], "fouls")
        opp_fouls = _read_stat(opp_row.get("statistics") or [], "fouls")

        # our_fouls = committed by us; opp_fouls = committed by them -> we drew
        rows.append({
            "fixture_id": fid,
            "date": fx.get("date"),
            "opponent": fx.get("opponent"),
            "is_home": fx.get("is_home"),
            "fouls_committed": our_fouls,
            "fouls_drawn": opp_fouls,
        })
        tot_comm  += our_fouls
        tot_drawn += opp_fouls
        counted   += 1

    return {
        "matches_counted": counted,
        "fouls_committed_per_match": (tot_comm / counted) if counted else 0.0,
        "fouls_drawn_per_match":     (tot_drawn / counted) if counted else 0.0,
        "fixtures": rows,
    }