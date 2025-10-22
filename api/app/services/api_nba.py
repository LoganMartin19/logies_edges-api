# api/app/services/api_nba.py
from __future__ import annotations
import os, time
from datetime import date, datetime, timezone
from typing import Dict, Tuple, List, Any, Optional
import requests

API_BASE = os.getenv("NBA_API_BASE", "https://v2.nba.api-sports.io")
NBA_GAMES_URL = os.getenv("NBA_GAMES_URL", f"{API_BASE}/games")
NBA_ODDS_URL  = os.getenv("NBA_ODDS_URL",  f"{API_BASE}/odds")

# Key handling similar to your other services
API_KEY       = os.getenv("NBA_API_KEY") or os.getenv("GRIDIRON_API_KEY") or os.getenv("FOOTBALL_API_KEY", "")
RAPIDAPI_KEY  = os.getenv("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST", "v2.nba.api-sports.io")

if RAPIDAPI_KEY:
    HEADERS = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST,
        "Accept": "application/json",
    }
else:
    HEADERS = {"x-apisports-key": API_KEY, "Accept": "application/json"}

# tiny cache + last-http, same style as ice
_CACHE: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Tuple[float, List[dict]]] = {}
_CACHE_TTL = 60
LAST_HTTP: Dict[str, Any] = {
    "url": None, "params": None, "status": None, "error": None,
    "response_count": None, "cached": False, "timestamp": None
}

def _ck(url: str, params: dict) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    return (url, tuple(sorted((k, str(v)) for k, v in (params or {}).items())))

def _stamp_last(url: str, params: dict, status: int | None, error: str | None,
                resp_len: int, cached: bool) -> None:
    LAST_HTTP.update({
        "url": url, "params": params, "status": status, "error": error,
        "response_count": resp_len, "cached": cached,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

def _get(url: str, params: dict, retries: int = 4, backoff: float = 0.75) -> List[dict]:
    key = _ck(url, params or {})
    now = time.time()
    if key in _CACHE and now - _CACHE[key][0] < _CACHE_TTL:
        data = _CACHE[key][1]
        _stamp_last(url, params, 200, None, len(data), True)
        return data

    wait = backoff
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=20)
            if r.status_code in (429, 500, 502, 503, 504) and i < retries - 1:
                time.sleep(wait); wait = min(wait * 2, 5.0); continue
            r.raise_for_status()
            j = r.json() or {}
            # v2.nba.api-sports.io returns a dict with "response" list
            resp = j.get("response") if isinstance(j, dict) else j
            data: List[dict] = resp if isinstance(resp, list) else []
            _CACHE[key] = (time.time(), data)
            _stamp_last(url, params, r.status_code, None, len(data), False)
            return data
        except Exception as e:
            if i == retries - 1:
                _CACHE[key] = (time.time(), [])
                status = getattr(getattr(e, "response", None), "status_code", None)
                _stamp_last(url, params, status, str(e), 0, False)
                return []
            time.sleep(wait); wait = min(wait * 2, 5.0)
    _stamp_last(url, params, None, "unknown", 0, False)
    return []

# ---------------- helpers ----------------

def _norm_league_name(lg: Any) -> str:
    """
    NBA API sometimes returns 'league' as a string ("standard")
    or a dict. Normalize to lowercase string safely.
    """
    if isinstance(lg, dict):
        return str(lg.get("name") or lg.get("key") or lg.get("slug") or lg.get("id") or "").strip().lower()
    return str(lg or "").strip().lower()

def _is_nba_league(lg: Any) -> bool:
    """True for NBA proper: 'standard' (API key for NBA) or 'nba'."""
    s = _norm_league_name(lg)
    return s in {"standard", "nba"}

# ---------------- primary fetchers ----------------

def fetch_fixtures(day: date, leagues: List[str] | None = None) -> List[dict]:
    """
    Fetch games for a date. By default, returns NBA-only (standard/nba).
    If 'leagues' provided, it must include 'standard' or 'nba' to pass the filter.
    """
    rows = _get(NBA_GAMES_URL, {"date": day.isoformat()}) or []
    if not rows:
        return []

    allowed: Optional[set[str]] = None
    if leagues:
        allowed = {str(s).strip().lower() for s in leagues if str(s).strip()}

    out: List[dict] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        lg = r.get("league")
        lg_ok = _is_nba_league(lg)
        if allowed is not None:
            # require league key to be in caller-allowed set
            if _norm_league_name(lg) not in allowed:
                continue
        else:
            # default: NBA only
            if not lg_ok:
                continue
        out.append(r)

    return out

def fetch_odds_for_game(game_id: int) -> List[dict]:
    """
    Odds endpoint can be sparse on some plans. Keep tolerant.
    """
    return _get(NBA_ODDS_URL, {"game": int(game_id)}) or []