# api/app/services/league_strength.py
from typing import Dict, Optional, Union
from fastapi import APIRouter, Query, Depends
from sqlalchemy.orm import Session
from datetime import datetime

from .apifootball import LEAGUE_MAP, get_standings_for_league
from ..db import get_db
from ..models import TeamForm  # DB model with 'comp', 'team', 'strength'

router = APIRouter(prefix="/api/fixtures", tags=["league-strength"])
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
    "argentina primera división": "ARG_LP",  # ✅ add Argentina alias
}

def _resolve_league_key(key: Union[str, int]) -> Optional[str]:
    """
    Accept internal key (EPL), alias ('Premier League'), or numeric id.
    Returns internal key used in DB (e.g., 'EPL'), or None.
    """
    if isinstance(key, int):
        # reverse-lookup internal key from LEAGUE_MAP values
        for k, v in LEAGUE_MAP.items():
            if v == key:
                return k
        return None
    s = (key or "").strip()
    if not s:
        return None
    # direct internal key
    if s in LEAGUE_MAP:
        return s
    # numeric to internal
    if s.isdigit():
        num = int(s)
        for k, v in LEAGUE_MAP.items():
            if v == num:
                return k
        return None
    # alias to internal
    alias = _ALIAS.get(s.lower())
    if alias and alias in LEAGUE_MAP:
        return alias
    return None

def _league_id_from_key(internal_key: str) -> Optional[int]:
    return LEAGUE_MAP.get(internal_key)

# --- Strength logic ----------------------------------------------------------

def _build_strength_cache_for_comp(db: Session, internal_comp_key: str) -> Dict[str, float]:
    """
    Try DB first (TeamForm.strength). If empty, fall back to API standings.
    Returns a normalized-name -> strength map.
    """
    rows = db.query(TeamForm).filter(
        TeamForm.comp == internal_comp_key,
        TeamForm.strength.isnot(None)
    ).all()

    if rows:
        return {normalize_team_name(r.team): float(r.strength) for r in rows}

    # Fallback: API league table
    league_id = _league_id_from_key(internal_comp_key)
    if league_id is None:
        return {}

    now = datetime.utcnow()
    season = now.year if now.month >= 7 else now.year - 1
    strength_map, _table = get_standings_for_league(league_id=league_id, season=season)  # your helper returns (map, table)
    # Normalize keys; guard against None
    normed = {normalize_team_name(k): float(v) for k, v in (strength_map or {}).items()}
    return normed

def _fuzzy_lookup(name_norm: str, strength_map: Dict[str, float]) -> Optional[float]:
    """
    Fallback matcher for cases like 'Belgrano Córdoba' vs 'Belgrano'.
    Try startswith / contains on tokens.
    """
    if name_norm in strength_map:
        return strength_map[name_norm]

    # token-based contain
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
    # prefix fallback (common for city suffixes)
    for k in strength_map.keys():
        if name_norm.startswith(k) or k.startswith(name_norm):
            return strength_map[k]

    return None

def get_team_strength(team_name: str, league_key: Union[str, int], db: Session) -> float:
    """
    Return league-relative strength in [0,1], where ~1 = top, ~0 = bottom.
    """
    internal = _resolve_league_key(league_key)
    if not internal:
        return 0.50

    cache_key = f"comp:{internal}"
    strength_map = _cached_strengths.get(cache_key)
    if strength_map is None:
        strength_map = _build_strength_cache_for_comp(db, internal)
        _cached_strengths[cache_key] = strength_map

    if not strength_map:
        return 0.50

    name_norm = normalize_team_name(team_name)
    val = strength_map.get(name_norm)
    if val is None:
        val = _fuzzy_lookup(name_norm, strength_map)

    return float(val) if isinstance(val, (int, float)) else 0.50

def get_strengths_for_comp(db: Session, comp: str) -> Dict[str, float]:
    rows = db.query(TeamForm).filter(
        TeamForm.comp == comp,
        TeamForm.strength.isnot(None)
    ).all()
    return {r.team: float(r.strength) for r in rows}

# --- API-first league table (unchanged public endpoint) ----------------------

@router.get("/league/table")
def league_table(
    league: str = Query(..., description="Internal key (EPL/UCL), alias, or numeric league id"),
    season: int | None = Query(None, description="Season year, defaults to current"),
    db: Session = Depends(get_db),
):
    # Keep existing behavior but normalize the key for consistency.
    internal = _resolve_league_key(league)
    if not internal:
        return {"source": "api", "error": f"Unknown league '{league}'", "table": [], "strength_map": {}}

    league_id = _league_id_from_key(internal)
    if not league_id:
        return {"source": "api", "error": f"No league id for '{internal}'", "table": [], "strength_map": {}}

    if season is None:
        now = datetime.utcnow()
        season = now.year if now.month >= 7 else now.year - 1

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
    comp: str = Query(..., description="Internal league key, e.g., EPL, CHAMP, SCO_CHAMP"),
    db: Session = Depends(get_db),
):
    return get_strengths_for_comp(db, comp)