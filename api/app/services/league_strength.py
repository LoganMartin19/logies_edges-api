# api/app/services/league_strength.py

from typing import Dict, Optional, Union
from fastapi import APIRouter, Query, Depends
from sqlalchemy.orm import Session  # kept for signature compatibility; not used
from datetime import datetime

from .apifootball import LEAGUE_MAP, get_standings_for_league
from ..db import get_db  # kept for signature compatibility (FastAPI Depends)

router = APIRouter(prefix="/api/fixtures", tags=["league-strength"])

# cache: comp_key -> { normalized_team_name: strength }
_cached_strengths: Dict[str, Dict[str, float]] = {}

# --- Normalization helpers ---------------------------------------------------

def normalize_team_name(name: str) -> str:
    return (
        (name or "")
        .lower()
        .replace(" fc", "")
        .replace(" afc", "")
        .replace(".", "")
        .replace("'", "")
        .strip()
    )

_ALIAS = {
    "championship": "CHAMP",
    "league one": "LG1",
    "league two": "LG2",
    "premier league": "EPL",
    "scottish championship": "SCO_CHAMP",
    "scottish premiership": "SCO_PREM",
    "challenge cup": "SCO_CHAL",
    "bundesliga": "BUNDES",
    "2. bundesliga": "BUNDES2",
    "serie b": "SERIE_B",
    "serie a": "SERIE_A",
    "ligue 1": "LIGUE1",
    "ligue 2": "LIGUE2",
    "la liga": "LA_LIGA",
    "ucl": "UCL",
    "uel": "UEL",
    "uecl": "UECL",
    "argentina primera división": "ARG_LP",  # common alias
}

def _resolve_league_key(key: Union[str, int]) -> Optional[str]:
    """
    Accept internal key (EPL), alias ('Premier League'), or numeric id.
    Returns internal key used in LEAGUE_MAP (e.g., 'EPL'), or None.
    """
    if isinstance(key, int):
        for k, v in LEAGUE_MAP.items():
            if v == key:
                return k
        return None

    s = (key or "").strip()
    if not s:
        return None

    # already internal?
    if s in LEAGUE_MAP:
        return s

    # numeric string?
    if s.isdigit():
        num = int(s)
        for k, v in LEAGUE_MAP.items():
            if v == num:
                return k
        return None

    # alias?
    alias = _ALIAS.get(s.lower())
    if alias and alias in LEAGUE_MAP:
        return alias

    return None

def _league_id_from_key(internal_key: str) -> Optional[int]:
    return LEAGUE_MAP.get(internal_key)

# --- API-first table/strength fetch ------------------------------------------

def _current_season() -> int:
    now = datetime.utcnow()
    return now.year if now.month >= 7 else now.year - 1

def _fetch_strength_map_api(internal_comp_key: str, season: Optional[int] = None) -> Dict[str, float]:
    """
    Pull strengths from API-Football standings via helper.
    Returns normalized team name -> strength in [0..1].
    """
    league_id = _league_id_from_key(internal_comp_key)
    if league_id is None:
        return {}

    season = season or _current_season()
    strength_map, _table = get_standings_for_league(league_id=league_id, season=season)
    if not strength_map:
        return {}

    # Normalize keys
    return {normalize_team_name(k): float(v) for k, v in strength_map.items()}

def _get_cached_or_fetch(internal_comp_key: str, season: Optional[int] = None) -> Dict[str, float]:
    cache_key = f"{internal_comp_key}:{season or _current_season()}"
    m = _cached_strengths.get(cache_key)
    if m is None:
        m = _fetch_strength_map_api(internal_comp_key, season=season)
        _cached_strengths[cache_key] = m
    return m

def _fuzzy_lookup(name_norm: str, strength_map: Dict[str, float]) -> Optional[float]:
    """
    Fallback matcher for near-name mismatches (e.g., accents or suffixes).
    """
    if name_norm in strength_map:
        return strength_map[name_norm]

    # token overlap scoring
    tokens = set(name_norm.split())
    best_key = None
    best_overlap = 0
    for k in strength_map.keys():
        ktokens = set(k.split())
        overlap = len(tokens & ktokens)
        if overlap > best_overlap or (overlap == best_overlap and best_key is None):
            best_key = k
            best_overlap = overlap

    if best_key and best_overlap > 0:
        return strength_map.get(best_key)

    # simple prefix symmetry
    for k in strength_map.keys():
        if name_norm.startswith(k) or k.startswith(name_norm):
            return strength_map[k]

    return None

# --- Public functions used by edge/explain -----------------------------------

def get_team_strength(team_name: str, league_key: Union[str, int], db: Session | None = None) -> float:
    """
    API-first: return league-relative strength in [0,1], ~1 = top, ~0 = bottom.
    (db is unused; kept for signature compatibility.)
    """
    internal = _resolve_league_key(league_key)
    if not internal:
        return 0.50

    smap = _get_cached_or_fetch(internal)
    if not smap:
        return 0.50

    name_norm = normalize_team_name(team_name)
    val = smap.get(name_norm)
    if val is None:
        val = _fuzzy_lookup(name_norm, smap)

    try:
        return float(val) if val is not None else 0.50
    except Exception:
        return 0.50

# --- Endpoints ---------------------------------------------------------------

@router.get("/league/table")
def league_table(
    league: str = Query(..., description="Internal key (EPL/UCL), alias, or numeric league id"),
    season: int | None = Query(None, description="Season year, defaults to current"),
    db: Session = Depends(get_db),  # kept to mirror previous signature
):
    internal = _resolve_league_key(league)
    if not internal:
        return {"source": "api", "error": f"Unknown league '{league}'", "table": [], "strength_map": {}}

    league_id = _league_id_from_key(internal)
    if not league_id:
        return {"source": "api", "error": f"No league id for '{internal}'", "table": [], "strength_map": {}}

    season = season or _current_season()
    strength_map, table = get_standings_for_league(league_id=league_id, season=season)
    return {
        "source": "api",
        "league_id": league_id,
        "season": season,
        "table": table or [],
        "strength_map": strength_map or {},
    }

@router.get("/league-strength")
def league_strength_endpoint(
    comp: str = Query(..., description="Internal league key OR alias OR numeric id"),
    season: int | None = Query(None, description="Season year, defaults to current"),
    db: Session = Depends(get_db),  # kept to mirror previous signature
):
    """
    API-first: returns {team -> strength} from API standings, not DB.
    """
    internal = _resolve_league_key(comp)
    if not internal:
        return {}

    smap = _get_cached_or_fetch(internal, season=season)
    # Return denormalized (pretty) names? Historically we returned raw names.
    # Here we keep normalized keys stable for matching; FE can map if needed.
    return smap