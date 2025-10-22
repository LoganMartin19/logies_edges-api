# api/app/services/api_ice.py
from __future__ import annotations
import os, time
from datetime import date, datetime, timezone, timedelta
from typing import Dict, Tuple, List, Any, Optional
import requests

API_BASE = os.getenv("ICE_API_BASE", "https://v1.hockey.api-sports.io")
ICE_FIXTURES_URL = os.getenv("ICE_FIXTURES_URL", f"{API_BASE}/games")
ICE_ODDS_URL     = os.getenv("ICE_ODDS_URL",     f"{API_BASE}/odds")

# Support RapidAPI proxy (optional) or direct key
ICE_API_KEY   = os.getenv("ICE_API_KEY") or os.getenv("GRIDIRON_API_KEY") or os.getenv("FOOTBALL_API_KEY", "")
RAPIDAPI_KEY  = os.getenv("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST", "v1.hockey.api-sports.io")

if RAPIDAPI_KEY:
    HEADERS = {"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": RAPIDAPI_HOST, "Accept": "application/json"}
else:
    HEADERS = {"x-apisports-key": ICE_API_KEY, "Accept": "application/json"}

# League map
LEAGUE_MAP: Dict[str, int] = {
    "NHL": 57,   # we will still fetch by date and filter to id==57, to avoid season-gating
}

# tiny cache for status/debug
_CACHE: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Tuple[float, List[dict]]] = {}
_CACHE_TTL = 60
LAST_HTTP: Dict[str, Any] = {"url": None, "params": None, "status": None, "error": None, "response_count": None, "cached": False, "timestamp": None}

def _ck(url: str, params: dict) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    return (url, tuple(sorted((k, str(v)) for k, v in (params or {}).items())))

def _stamp_last(url: str, params: dict, status: int | None, error: str | None, resp_len: int, cached: bool) -> None:
    LAST_HTTP.update({"url": url, "params": params, "status": status, "error": error, "response_count": resp_len, "cached": cached, "timestamp": datetime.now(timezone.utc).isoformat()})

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

# ---- helpers ----------------------------------------------------------------

def _is_finished(row: dict) -> bool:
    """
    Return True if a hockey game is finished.
    API-Sports hockey exposes status under row["status"] (dict with short/long).
    """
    st = row.get("status") or (row.get("game") or {}).get("status") or {}
    short = (st.get("short") if isinstance(st, dict) else st) or ""
    long  = (st.get("long")  if isinstance(st, dict) else "")
    s = str(short).upper()
    l = str(long).upper()
    return (
        s in {"FT", "FINAL", "AOT", "AP", "ENDED"} or
        "FINAL" in l or "ENDED" in l or "AFTER" in l
    )

# ---- primary endpoints -------------------------------------------------------

def fetch_fixtures(day: date, leagues: List[str]) -> List[dict]:
    """
    Fetch by DATE ONLY (to bypass season lock), then filter locally to NHL (id==57).
    This is used by your ingest; it intentionally normalizes minimally.
    """
    rows = _get(ICE_FIXTURES_URL, {"date": day.isoformat()}) or []
    wanted_ids = {LEAGUE_MAP.get(k) for k in (leagues or []) if LEAGUE_MAP.get(k)}
    if not wanted_ids:
        return []
    out = []
    for p in rows:
        lg = p.get("league") or {}
        if int(lg.get("id") or 0) in wanted_ids:
            out.append({
                "id": p.get("id"),
                "league": lg,
                "teams": {"home": (p.get("teams") or {}).get("home"), "away": (p.get("teams") or {}).get("away")},
                "game": {"date": {
                    "date": (p.get("date") or "")[:10],
                    "time": (p.get("time") or "00:00"),
                    "timezone": (p.get("timezone") or "UTC"),
                    "timestamp": p.get("timestamp"),
                }},
            })
    return out

def fetch_odds_for_game(game_id: int) -> List[dict]:
    return _get(ICE_ODDS_URL, {"game": int(game_id)})

def get_fixture(game_id: int) -> dict:
    """
    Fetch a single hockey game by provider id.
    Returns {"response": [ ... ]} (mirrors API-Football).
    """
    rows = _get(ICE_FIXTURES_URL, {"id": int(game_id)}) or []
    return {"response": rows}

# --- Team games fetcher (free-plan friendly) ---------------------------------

def _iter_days(start: date, end: date):
    d = start
    one = timedelta(days=1)
    while d <= end:
        yield d
        d += one

def fetch_team_games(
    team_id: int,
    season: Optional[int | str] = None,
    league: Optional[str | int] = "NHL",  # default to NHL
    status: str = "FT",
    last_n: int = 10,
) -> List[dict]:
    """
    Returns up to last_n finished NHL games for a team.
    Plan-gate safe: falls back to date-loop + local filtering when 'season' is blocked.
    Ensures we only return FINISHED games in all branches.
    """
    params: Dict[str, Any] = {"team": int(team_id)}
    if status:
        params["status"] = status
    if league is not None:
        if isinstance(league, str):
            lid = LEAGUE_MAP.get(league.upper())
            if lid:
                params["league"] = lid
        else:
            params["league"] = int(league)
    if season is not None:
        try:
            params["season"] = int(season)
        except Exception:
            pass

    # 1) direct (may be season-gated on free plan)
    rows = _get(ICE_FIXTURES_URL, params) or []
    rows = [r for r in rows if _is_finished(r)]
    if rows:
        return rows[:last_n]

    # 2) without season
    params.pop("season", None)
    rows = _get(ICE_FIXTURES_URL, params) or []
    rows = [r for r in rows if _is_finished(r)]
    if rows:
        return rows[:last_n]

    # 3) date window fallback — use RAW /games?date=YYYY-MM-DD (not fetch_fixtures)
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=365)

    wanted_ids = set()
    if league is not None:
        if isinstance(league, str):
            lx = LEAGUE_MAP.get(league.upper())
            if lx:
                wanted_ids.add(lx)
        else:
            wanted_ids.add(int(league))
    if not wanted_ids:
        wanted_ids = {LEAGUE_MAP["NHL"]}

    collected: List[dict] = []
    for d in reversed(list(_iter_days(start, today))):  # newest first
        # raw call keeps status/scores
        day_rows = _get(ICE_FIXTURES_URL, {"date": d.isoformat()}) or []
        for p in day_rows:
            lg = p.get("league") or {}
            if int(lg.get("id") or 0) not in wanted_ids:
                continue
            if not _is_finished(p):
                continue
            tm = (p.get("teams") or {})
            hid = (tm.get("home") or {}).get("id")
            aid = (tm.get("away") or {}).get("id")
            if hid != team_id and aid != team_id:
                continue
            collected.append(p)
            if len(collected) >= last_n:
                break
        if len(collected) >= last_n:
            break

    return collected