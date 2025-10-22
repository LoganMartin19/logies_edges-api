# api/app/services/league_strength.py

from typing import Dict, List, Optional, Union
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
}

def _resolve_league_id(league: Union[str, int]) -> Optional[int]:
    """
    Accepts internal key (EPL), alias (Premier League), or numeric ID.
    Returns API-Football league_id or None if not resolvable.
    """
    if isinstance(league, int):
        return league
    if isinstance(league, str):
        s = league.strip()
        if s.isdigit():
            return int(s)
        if s in LEAGUE_MAP:
            return LEAGUE_MAP[s]
        k = _ALIAS.get(s.lower())
        if k and k in LEAGUE_MAP:
            return LEAGUE_MAP[k]
    return None

# --- Strength logic (unchanged) ----------------------------------------------

def get_team_strength(team_name: str, league_key: str, db: Session) -> float:
    league_key_lower = str(league_key).lower()
    reverse_map = {str(v).lower(): k for k, v in LEAGUE_MAP.items()}

    if league_key_lower in _ALIAS:
        league_key = _ALIAS[league_key_lower]
    elif league_key_lower in reverse_map:
        league_key = reverse_map[league_key_lower]
    elif league_key not in LEAGUE_MAP:
        return 0.50

    norm_input = normalize_team_name(team_name)

    cache_key = f"comp:{league_key}"
    strength_map = _cached_strengths.get(cache_key)
    if strength_map is None:
        rows = db.query(TeamForm).filter(
            TeamForm.comp == league_key,
            TeamForm.strength.isnot(None)
        ).all()
        strength_map = {normalize_team_name(r.team): r.strength for r in rows}
        _cached_strengths[cache_key] = strength_map

    return strength_map.get(norm_input, 0.50)

def get_strengths_for_comp(db: Session, comp: str) -> Dict[str, float]:
    rows = db.query(TeamForm).filter(
        TeamForm.comp == comp,
        TeamForm.strength.isnot(None)
    ).all()
    return {r.team: r.strength for r in rows}

# --- API-first league table --------------------------------------------------

def get_league_table_api_first(
    league: Union[str, int],
    season: Optional[int] = None,
    db: Optional[Session] = None,
) -> Dict[str, object]:
    """
    API-first table/strengths.
    Returns {"source": "api", "league_id": int, "season": int, "table": [...], "strength_map": {...}}
    """
    league_id = _resolve_league_id(league)
    if league_id is None:
        return {"source": "api", "error": f"Unknown league '{league}'", "table": [], "strength_map": {}}

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

# --- Endpoints ---------------------------------------------------------------

@router.get("/league/table")
def league_table(
    league: str = Query(..., description="Internal key (EPL/UCL), alias, or numeric league id"),
    season: int | None = Query(None, description="Season year, defaults to current"),
    db: Session = Depends(get_db),
):
    """
    Fetch league table (API-first).
    Backward compatible with frontend calls.
    """
    out = get_league_table_api_first(league=league, season=season, db=db)
    return out if "table" in out else {"table": [], **out}

@router.get("/league-strength")
def league_strength_endpoint(
    comp: str = Query(..., description="Internal league key, e.g., EPL, CHAMP, SCO_CHAMP"),
    db: Session = Depends(get_db),
):
    return get_strengths_for_comp(db, comp)