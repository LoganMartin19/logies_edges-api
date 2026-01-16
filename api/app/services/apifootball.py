# api/app/services/apifootball.py
from __future__ import annotations
import os
import time
import requests
from typing import Dict, Tuple, List, Optional
from datetime import date, datetime, timezone

from ..db import SessionLocal
from .team_stats_cache import get_team_season_stats_cached 


# ---------------- Config ----------------
BASE_URL = os.getenv("FOOTBALL_API_URL", "https://v3.football.api-sports.io")
API_URL = f"{BASE_URL}/fixtures"
ODDS_URL = f"{BASE_URL}/odds"

# If you're using RapidAPI proxy, keep these headers. If you use direct API-Football,
# swap to: {"x-apisports-key": API_KEY}
API_KEY = os.getenv("FOOTBALL_API_KEY")
API_HOST = os.getenv("RAPIDAPI_HOST", "v3.football.api-sports.io")
HEADERS = {
    "X-RapidAPI-Key": API_KEY or "",
    "X-RapidAPI-Host": API_HOST,
    "Accept": "application/json",
}

# ---------------- League IDs ----------------
LEAGUE_MAP = {
    # England 🇬🇧
    "EPL": 39, "Premier League": 39,
    "CHAMP": 40, "Championship": 40,
    "LG1": 41, "League One": 41,
    "LG2": 42, "League Two": 42,
    "ENG_FA": 45, "FA Cup": 45,
    "EFL_CUP": 48, "EFL Cup": 48,
    "EFL_TROPHY": 46, "EFL Trophy": 46,
    "NAT_LEAGUE": 49, "National League": 43,
    "NAT_NORTH": 50, "National League North": 50,
    "NAT_SOUTH": 51, "National League South": 51,

    # Scotland 🏴
    "SCO_PREM": 179, "Scottish Premiership": 179,
    "SCO_CHAMP": 180, "Scottish Championship": 180,
    "SCO1": 183, "SCO2": 184,
    "SCO_SC": 181, "Scottish Cup": 181,
    "SCO_LC": 185, "Scottish League Cup": 185,
    "SCO_CHAL": 182, "Scottish Challenge Cup": 182,

    # Spain 🇪🇸
    "LA_LIGA": 140, "La Liga": 140,
    "LA_LIGA2": 141, "La Liga 2": 141,
    "ESP_CDR": 143, "Copa del Rey": 143,

    # Germany 🇩🇪
    "BUNDES": 78, "Bundesliga": 78,
    "BUNDES2": 79, "2. Bundesliga": 79,
    "GER_POKAL": 81, "DFB-Pokal": 81,

    # Italy 🇮🇹
    "SERIE_A": 135, "Serie A": 135,
    "SERIE_B": 136, "Serie B": 136,
    "ITA_COPPA": 137, "Coppa Italia": 137,

    # France 🇫🇷
    "LIGUE1": 61, "Ligue 1": 61,
    "LIGUE2": 62, "Ligue 2": 62,
    "FRA_CDF": 66, "Coupe de France": 66,

    # Portugal 🇵🇹
    "POR_LIGA": 94, "Primeira Liga": 94,
    "POR_TACA": 95, "Taça de Portugal": 95,

    # Netherlands 🇳🇱
    "NED_ERED": 88, "Eredivisie": 88,
    "NED_EERST": 89, "Eerste Divisie": 89,
    "NED_KNVB": 90, "KNVB Beker": 90,

    # Belgium 🇧🇪
    "BEL_PRO": 144, "Pro League": 144,
    "BEL_CUP": 145, "Belgian Cup": 145,

    # Norway 🇳🇴
    "NOR_ELI": 103, "Eliteserien": 103,
    "NOR_CUP": 104, "Norwegian Cup": 104,

    # Denmark 🇩🇰
    "DEN_SL": 119, "Superliga": 119,
    "DEN_CUP": 63, "DBU Pokalen": 63,

    # Sweden 🇸🇪
    "SWE_ALLS": 113, "Allsvenskan": 113,
    "SWE_SUPER": 114, "Superettan": 114,
    "SWE_CUP": 115, "Svenska Cupen": 115,

    # Argentina 🇦🇷
    "ARG_LP": 128, "Liga Profesional": 128,
    "ARG_CDL": 130, "Copa de la Liga Profesional": 130,
    "ARG_CUP": 131, "Copa Argentina": 131,

    # Brazil 🇧🇷
    "BR_SERIE_A": 71, "Brasileirão Série A": 71,
    "BR_SERIE_B": 72, "Brasileirão Série B": 72,

    # USA 🇺🇸
    "MLS": 253, "Major League Soccer": 253,

    # Continental 🇪🇺
    "UCL": 2, "UEFA Champions League": 2,
    "UEL": 3, "UEFA Europa League": 3,
    "UECL": 848, "UEFA Europa Conference League": 848,
    "UWCL": 525, "UEFA Women's Champions League": 525,
    "WCQ_EUR": 32, "FIFA World Cup Qualifiers - Europe": 32,
    "AFCON": 6, "Africa Cup of Nations": 6,

    #Rest of the World 🌍
    "AUS_A_LEAGUE": 188, "A-League": 188,
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

# ---------------- In-proc HTTP cache (short-lived) ----------------
_CACHE: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Tuple[float, List[dict]]] = {}
_CACHE_TTL = 36000.0


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
      - uses a short in-proc cache
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
                _CACHE[key] = (time.time(), [])
                return []

            _CACHE[key] = (time.time(), data)
            return data

        except Exception as e:
            if attempt == retries - 1:
                LAST_HTTP.update({
                    "url": url,
                    "params": dict(params or {}),
                    "status": getattr(e, "response", None).status_code
                    if hasattr(e, "response") and e.response
                    else "error",
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
    """
    Raw fixture details by provider fixture ID (JSON passthrough),
    backed by FixtureDetailCache in Postgres.
    """
    from .fixture_cache import get_fixture_detail_cached  # 👈 local import

    db = SessionLocal()
    try:
        return get_fixture_detail_cached(db, int(fixture_id))
    finally:
        db.close()


def get_players(team_id: int, league_id: Optional[int], season: int):
    """
    Player-level stats (cards, shots, fouls etc).
    If league_id is None, API-Football returns all competitions for that season.
    """
    params = {"team": team_id, "season": season}
    if league_id is not None:
        params["league"] = league_id
    return requests.get(
        f"{BASE_URL}/players",
        headers=HEADERS,
        params=params,
        timeout=20,
    ).json()


def get_predictions(fixture_id: int):
    return requests.get(
        f"{BASE_URL}/predictions",
        headers=HEADERS,
        params={"fixture": int(fixture_id)},
        timeout=20,
    ).json()


def get_lineups(fixture_id: int):
    return requests.get(
        f"{BASE_URL}/fixtures/lineups",
        headers=HEADERS,
        params={"fixture": int(fixture_id)},
        timeout=20,
    ).json()


def get_h2h(team1: int, team2: int):
    return requests.get(
        f"{BASE_URL}/fixtures/headtohead",
        headers=HEADERS,
        params={"h2h": f"{team1}-{team2}"},
        timeout=20,
    ).json()




def get_team_stats(team_id: int, league_id: int, season: int, *, refresh: bool = False):
    """
    Team-level season statistics, DB-backed via TeamSeasonStats.

    Returns the same shape as the raw API:
        {
          "get": "...",
          "parameters": {...},
          "errors": {...},
          "results": ...,
          "response": { ... }
        }
    so existing callers using stats.get("response") keep working.
    """
    db = SessionLocal()
    try:
        return get_team_season_stats_cached(
            db,
            team_id=int(team_id),
            league_id=int(league_id),
            season=int(season),
            refresh=refresh,
        )
    finally:
        db.close()


def get_top_scorers(league: int, season: int):
    return requests.get(
        f"{BASE_URL}/players/topscorers",
        headers=HEADERS,
        params={"league": league, "season": season},
        timeout=20,
    ).json()


def get_injuries(team_id: int, league_id: int, season: int):
    return requests.get(
        f"{BASE_URL}/injuries",
        headers=HEADERS,
        params={"team": team_id, "league": league_id, "season": season},
        timeout=20,
    ).json()


def get_events(fixture_id: int):
    """
    Raw events JSON, backed by FixtureEventsCache (short TTL ~5 minutes).
    """
    from .fixture_cache import get_fixture_events_cached  # 👈 local import

    db = SessionLocal()
    try:
        return get_fixture_events_cached(db, int(fixture_id))
    finally:
        db.close()

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
            "comp_key": canonicalize_comp(lg),
        })

    cleaned.sort(key=lambda x: x["date"] or "", reverse=True)
    return cleaned[:limit]

# ---------------- Player props helpers ----------------
def get_player_stats(team_id: int, league_id: int, season: int = 2025) -> List[dict]:
    """
    Player-level stats (cards, shots, fouls etc) for a team in league/season.
    Uses paging aggregator to fetch all players.
    """
    return _get_all_pages(
        f"{BASE_URL}/players",
        {"team": team_id, "league": league_id, "season": season},
    )


def get_fixture_fouls_drawn(fixture_id: int, team_id: int) -> float:
    """
    Fouls drawn by team_id in THIS fixture.
    We count 'Foul' events committed by the OPPONENT (team != team_id).
    Uses cached events via get_events().
    """
    try:
        j = get_events(int(fixture_id)) or {}
        events = j.get("response") or []
        if not isinstance(events, list):
            return 0.0
    except Exception as e:
        print(f"[get_fixture_fouls_drawn] error: {e}")
        return 0.0

    drawn = 0
    for ev in events:
        ev_team_id = (ev.get("team") or {}).get("id")
        ev_type = ev.get("type")
        if ev_type == "Foul" and ev_team_id and ev_team_id != team_id:
            drawn += 1

    return float(drawn)


def get_fixture_players(fixture_id: int):
    """
    API-Football: /fixtures/players?fixture={id}
    Returns { response: [ { team:{...}, players:[ { player:{...}, statistics:[...] } ] } ] }
    """
    url = f"{BASE_URL}/fixtures/players"
    r = requests.get(
        url,
        headers=HEADERS,
        params={"fixture": int(fixture_id)},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

# --- Opponent context averages (events-first, stats fallback) ----------------
def _events_fouls_committed_in_fixture(fixture_id: int, team_id: int) -> int:
    """Count fouls COMMITTED by team_id in a single fixture using events (cached)."""
    try:
        j = get_events(int(fixture_id)) or {}
        events = j.get("response") or []
        if not isinstance(events, list):
            return 0
    except Exception:
        return 0

    committed = 0
    for ev in events:
        if ev.get("type") == "Foul":
            ev_team_id = (ev.get("team") or {}).get("id")
            if ev_team_id and ev_team_id == team_id:
                committed += 1
    return committed


def get_team_fouls_committed_avg(
    team_id: int,
    season: int,
    league_id: int | None = None,
    lookback: int = 5,
) -> float:
    """
    Average fouls COMMITTED per match by team_id over last `lookback` fixtures (events-first).
    Fallback: /teams/statistics fouls.committed.total / fixtures.played.total
    """
    recent = get_team_recent_results(
        team_id, season=season, limit=lookback, league_id=league_id
    ) or []
    ev_counts: List[int] = []

    for fx in recent:
        fid = fx.get("fixture_id")
        if not fid:
            continue
        c = _events_fouls_committed_in_fixture(int(fid), team_id)
        ev_counts.append(c)

    if ev_counts and any(x is not None for x in ev_counts):
        denom = max(1, len(ev_counts))
        return float(sum(ev_counts)) / float(denom)

    try:
        if league_id is not None:
            stats = get_team_stats(team_id, league_id, season) or {}
            r = stats.get("response") or {}
            fouls = (r.get("fouls") or {}) or {}
            committed_total = int((fouls.get("committed") or {}).get("total") or 0)
            played = int(
                ((r.get("fixtures") or {}).get("played") or {}).get("total") or 0
            )
            return (committed_total / played) if played else 0.0
    except Exception:
        pass

    return 0.0


def get_team_fouls_drawn_avg(
    team_id: int,
    season: int,
    league_id: int | None = None,
    lookback: int = 5,
) -> float:
    """
    Average fouls DRAWN per match by team_id over last `lookback` fixtures.
    For each fixture: fouls drawn by team A == fouls COMMITTED by the opponent.
    We compute it directly from events by counting fouls where event.team.id != team_id.
    Fallback: /teams/statistics fouls.drawn.total if available.
    """
    recent = get_team_recent_results(
        team_id, season=season, limit=lookback, league_id=league_id
    ) or []
    drawn_counts: List[int] = []

    for fx in recent:
        fid = fx.get("fixture_id")
        if not fid:
            continue
        try:
            j = get_events(int(fid)) or {}
            events = j.get("response") or []
            if not isinstance(events, list):
                continue
        except Exception:
            continue

        drawn = 0
        for ev in events:
            if ev.get("type") == "Foul":
                ev_team_id = (ev.get("team") or {}).get("id")
                if ev_team_id and ev_team_id != team_id:
                    drawn += 1
        drawn_counts.append(drawn)

    if drawn_counts and any(x is not None for x in drawn_counts):
        return float(sum(drawn_counts)) / float(len(drawn_counts))

    try:
        if league_id is not None:
            stats = get_team_stats(team_id, league_id, season) or {}
            r = stats.get("response") or {}
            fouls = (r.get("fouls") or {}) or {}
            drawn_total = int((fouls.get("drawn") or {}).get("total") or 0)
            played = int(
                ((r.get("fixtures") or {}).get("played") or {}).get("total") or 0
            )
            return (drawn_total / played) if played else 0.0
    except Exception:
        pass

    return 0.0

# --- single-fixture statistics (shots, SoT, fouls, xG, etc.) ---
def get_fixture_statistics(fixture_id: int):
    """
    Raw fixture stats (shots, SoT, fouls, xG, etc.) for a single fixture,
    backed by FixtureStatsCache.
    """
    from .fixture_cache import get_fixture_stats_cached  # 👈 local import

    db = SessionLocal()
    try:
        return get_fixture_stats_cached(db, int(fixture_id))
    finally:
        db.close()


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

# --- rolling averages from recent finished fixtures' statistics ---
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
      2) call get_fixture_statistics (DB-backed),
      3) read the OPPONENT's 'Total Shots' & 'Shots on Goal' (i.e., shots conceded).
    """
    recent = get_team_recent_results(
        team_id, season=season, limit=lookback, league_id=league_id
    ) or []
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
            continue

        opp_row = next(
            (r for r in resp if (r.get("team") or {}).get("id") != team_id),
            None,
        )
        if not opp_row:
            continue

        stats = opp_row.get("statistics") or []
        opp_total_shots = _read_stat(stats, "total shots", "shots total")
        opp_sot = _read_stat(stats, "shots on goal", "shots on target")

        rows.append({
            "fixture_id": fid,
            "date": fx.get("date"),
            "opponent": fx.get("opponent"),
            "is_home": fx.get("is_home"),
            "shots_against": opp_total_shots,
            "sot_against": opp_sot,
        })
        tot_sh += opp_total_shots
        tot_sot += opp_sot
        counted += 1

    return {
        "matches_counted": counted,
        "shots_against_per_match": (tot_sh / counted) if counted else 0.0,
        "sot_against_per_match": (tot_sot / counted) if counted else 0.0,
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
    Rolling averages of xG for and xG against over last N finished fixtures, using get_fixture_statistics (DB-backed).
    We read our own xG from our block, and xG against from opponent's block.
    """
    recent = get_team_recent_results(
        team_id, season=season, limit=lookback, league_id=league_id
    ) or []
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

        our_row = next(
            (r for r in resp if (r.get("team") or {}).get("id") == team_id),
            None,
        )
        opp_row = next(
            (r for r in resp if (r.get("team") or {}).get("id") != team_id),
            None,
        )
        if not our_row or not opp_row:
            continue

        our_stats = our_row.get("statistics") or []
        opp_stats = opp_row.get("statistics") or []

        xg_for = _read_stat(our_stats, "expected_goals", "xg")
        xg_against = _read_stat(opp_stats, "expected_goals", "xg")

        rows.append({
            "fixture_id": fid,
            "date": fx.get("date"),
            "opponent": fx.get("opponent"),
            "is_home": fx.get("is_home"),
            "xg_for": xg_for,
            "xg_against": xg_against,
        })
        tot_xg_for += xg_for
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
    Rolling averages of fouls committed and fouls drawn using get_fixture_statistics only.
    'Fouls' in opponent's block = fouls *they* committed => fouls we drew.
    """
    recent = get_team_recent_results(
        team_id, season=season, limit=lookback, league_id=league_id
    ) or []
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

        our_row = next(
            (r for r in resp if (r.get("team") or {}).get("id") == team_id),
            None,
        )
        opp_row = next(
            (r for r in resp if (r.get("team") or {}).get("id") != team_id),
            None,
        )
        if not our_row or not opp_row:
            continue

        our_fouls = _read_stat(our_row.get("statistics") or [], "fouls")
        opp_fouls = _read_stat(opp_row.get("statistics") or [], "fouls")

        rows.append({
            "fixture_id": fid,
            "date": fx.get("date"),
            "opponent": fx.get("opponent"),
            "is_home": fx.get("is_home"),
            "fouls_committed": our_fouls,
            "fouls_drawn": opp_fouls,
        })
        tot_comm += our_fouls
        tot_drawn += opp_fouls
        counted += 1

    return {
        "matches_counted": counted,
        "fouls_committed_per_match": (tot_comm / counted) if counted else 0.0,
        "fouls_drawn_per_match": (tot_drawn / counted) if counted else 0.0,
        "fixtures": rows,
    }