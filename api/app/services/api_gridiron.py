# api/app/services/api_gridiron.py
from __future__ import annotations
import os
import time
from datetime import date, datetime, timezone
from typing import Dict, Tuple, List, Any

import requests

# --- Endpoints & auth ---------------------------------------------------------

API_BASE = os.getenv("GRIDIRON_API_BASE", "https://v1.american-football.api-sports.io")

# NOTE: API_BASE already includes /v1 — endpoints are just /games and /odds
GRIDIRON_FIXTURES_URL = os.getenv("GRIDIRON_FIXTURES_URL", f"{API_BASE}/games")
GRIDIRON_ODDS_URL     = os.getenv("GRIDIRON_ODDS_URL",     f"{API_BASE}/odds")

# Support BOTH direct API-Sports and RapidAPI proxy (pick whichever env you have set)
GRIDIRON_API_KEY = os.getenv("GRIDIRON_API_KEY", "")            # direct API-Sports key
RAPIDAPI_KEY     = os.getenv("RAPIDAPI_KEY", "")                # RapidAPI key (optional)
RAPIDAPI_HOST    = os.getenv("RAPIDAPI_HOST", "v1.american-football.api-sports.io")

if RAPIDAPI_KEY:
    # RapidAPI proxy mode (mirrors your soccer config)
    HEADERS = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST,
        "Accept": "application/json",
    }
else:
    # Direct API-Sports mode
    HEADERS = {
        "x-apisports-key": GRIDIRON_API_KEY,
        "Accept": "application/json",
    }

# --- League map (provider league IDs) ----------------------------------------
# Accept multiple aliases and normalize
LEAGUE_MAP = {
    "NFL":   1,
    "NCAA":  2,   # API shows "NCAA" for college football
    "CFB":   2,   # common alias
    "NCAAF": 2,   # common alias
}

def _league_id(name: str | None) -> int | None:
    if not name:
        return None
    return LEAGUE_MAP.get(str(name).strip().upper())

# --- Light cache + last HTTP debug -------------------------------------------

_CACHE: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Tuple[float, List[dict]]] = {}
_CACHE_TTL = 60

LAST_HTTP: Dict[str, Any] = {
    "url": None,
    "params": None,
    "status": None,
    "error": None,
    "response_count": None,
    "cached": False,
    "timestamp": None,
}

def _ck(url: str, params: dict) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    return (url, tuple(sorted((k, str(v)) for k, v in (params or {}).items())))

def _stamp_last(url: str, params: dict, status: int | None, error: str | None, resp_len: int, cached: bool) -> None:
    LAST_HTTP.update({
        "url": url,
        "params": params,
        "status": status,
        "error": error,
        "response_count": resp_len,
        "cached": cached,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

def _get(url: str, params: dict, retries: int = 4, backoff: float = 0.75) -> List[dict]:
    key = _ck(url, params or {})
    now = time.time()

    # Serve from cache
    if key in _CACHE and now - _CACHE[key][0] < _CACHE_TTL:
        data = _CACHE[key][1]
        _stamp_last(url, params, status=200, error=None, resp_len=len(data), cached=True)
        return data

    wait = backoff
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=20)
            if r.status_code in (429, 500, 502, 503, 504) and i < retries - 1:
                time.sleep(wait)
                wait = min(wait * 2, 5.0)
                continue

            r.raise_for_status()
            j = r.json() or {}
            resp = j.get("response") if isinstance(j, dict) else j
            data: List[dict] = resp if isinstance(resp, list) else []

            _CACHE[key] = (time.time(), data)
            _stamp_last(url, params, status=r.status_code, error=None, resp_len=len(data), cached=False)
            return data

        except Exception as e:
            if i == retries - 1:
                _CACHE[key] = (time.time(), [])
                status = getattr(getattr(e, "response", None), "status_code", None)
                _stamp_last(url, params, status=status, error=str(e), resp_len=0, cached=False)
                return []
            time.sleep(wait)
            wait = min(wait * 2, 5.0)

    _stamp_last(url, params, status=None, error="unknown", resp_len=0, cached=False)
    return []

# --- Public fetchers ----------------------------------------------------------

def _normalize_leagues(leagues) -> list[str]:
    """
    Accept 'NCAA', 'NFL,NCAA' or a list-like object and return a proper list of league names.
    """
    if leagues is None:
        return []
    if isinstance(leagues, str):
        return [s.strip() for s in leagues.split(",") if s.strip()]
    try:
        return [str(s).strip() for s in leagues if str(s).strip()]
    except TypeError:
        return [str(leagues).strip()]


# api/app/services/api_gridiron.py
# ...imports & header setup unchanged...

def fetch_fixtures(day: date, leagues: List[str]) -> List[dict]:
    """
    Fetch all games on a given day and league(s), e.g. ['NFL','NCAA'].
    Some plans restrict 'season' – we first call WITHOUT the season param,
    then gracefully retry with a couple of candidates.
    """
    results: List[dict] = []
    # normalized league ids
    norm_keys = [k for k in (leagues or []) if _league_id(k)]

    for key in norm_keys:
        lid = _league_id(key)
        if not lid:
            continue

        # 1) Try WITHOUT season (best for mixed plans)
        base_params = {"date": day.isoformat(), "league": lid}
        data = _get(GRIDIRON_FIXTURES_URL, base_params)

        # 2) If empty, try with the date’s year (may work on paid plans)
        if not data:
            data = _get(GRIDIRON_FIXTURES_URL, {**base_params, "season": day.year})

        # 3) If still empty, try a safe fallback season allowed by free plan
        if not data:
            for fallback in (2023, 2022, 2021):
                data = _get(GRIDIRON_FIXTURES_URL, {**base_params, "season": fallback})
                if data:
                    break

        if data:
            results.extend(data)

    return results


def fetch_odds_for_game(game_id: int) -> list[dict]:
    """Fetch odds for one game."""
    return _get(GRIDIRON_ODDS_URL, {"game": int(game_id)})

# --- Team games fetcher (used by form) ---------------------------------------

# --- Team games fetcher (free-plan friendly) ---------------------------------
from typing import Optional
from datetime import datetime, timedelta, timezone, date as _date

def _iter_days(start: _date, end: _date):
    d = start
    one = timedelta(days=1)
    while d <= end:
        yield d
        d += one

def fetch_team_games(
    team_id: int,
    season: Optional[int | str] = None,
    league: Optional[str | int] = None,   # "NFL" / "NCAA" or id
    status: str = "FT",
    last_n: int = 10,
) -> List[dict]:
    """
    Returns up to last_n finished games for a team.
    Strategy:
      1) Try direct /games with season+status (fast).
      2) If empty (plan-gated), try /games with status only (no season).
      3) If still empty, loop recent dates (free-plan safe) and filter by team+league.
         Short-circuit when we’ve collected last_n.
    """
    params: Dict[str, Any] = {"team": int(team_id)}
    if status:
        params["status"] = status
    if league is not None:
        params["league"] = _league_id(league) if isinstance(league, str) else int(league)
    if season is not None:
        try:
            params["season"] = int(season)
        except Exception:
            pass

    # 1) direct
    rows = _get(GRIDIRON_FIXTURES_URL, params) or []
    if rows:
        return rows

    # 2) without season
    params.pop("season", None)
    rows = _get(GRIDIRON_FIXTURES_URL, params) or []
    if rows:
        return rows

    # 3) date window fallback (collect until last_n)
    # Use a 365-day lookback ending today; adjust if you want.
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=365)
    results: List[dict] = []
    wanted_league_ids = None
    if league is not None:
        wanted_league_ids = {_league_id(league)} if isinstance(league, str) else {int(league)}

    for d in reversed(list(_iter_days(start, today))):  # newest first
        day_rows = fetch_fixtures(d, [league] if league is not None else ["NFL", "NCAA"]) or []
        for p in day_rows:
            tm = (p.get("teams") or {})
            hid = (tm.get("home") or {}).get("id")
            aid = (tm.get("away") or {}).get("id")
            if hid != team_id and aid != team_id:
                continue
            if status and str(p.get("status") or "").upper() not in {"FT", "FINISHED", "ENDED"}:
                # API-Sports sometimes keeps 'status' nested; we’re lenient here
                pass  # most fetch_fixtures payloads don’t include status; filtering later in consumer is fine
            if wanted_league_ids:
                lg = p.get("league") or {}
                if int(lg.get("id") or 0) not in wanted_league_ids:
                    continue
            results.append(p)
            if len(results) >= last_n:
                break
        if len(results) >= last_n:
            break

    return results