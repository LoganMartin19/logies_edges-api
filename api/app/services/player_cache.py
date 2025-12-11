# api/app/services/player_cache.py
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from fastapi import HTTPException  # (still unused but fine to keep)

from ..models import TeamSeasonPlayers, FixturePlayersCache
from ..services.apifootball import BASE_URL, _get_all_pages, get_fixture_players

FRESH_FOR = timedelta(hours=12)


def _now():
    return datetime.utcnow()


# ---------- TEAM-SEASON PLAYERS (from /players?team=..&season=.., all pages) ----------

def get_team_season_players_cached(
    db: Session,
    team_id: int,
    season: int,
    refresh: bool = False,
) -> list[dict]:
    """
    Returns a flattened list of players for this team+season.
    Reads from cache when fresh; otherwise fetches and updates cache.
    If refresh=True, always refetch from provider (but won't overwrite DB with empty).
    NEVER writes a row if team_id or season are falsy, or if payload is empty.
    """
    team_id = int(team_id or 0)
    season = int(season or 0)
    if team_id <= 0 or season <= 0:
        # don't hit DB with invalid keys
        return []

    row = (
        db.query(TeamSeasonPlayers)
        .filter(TeamSeasonPlayers.team_id == team_id, TeamSeasonPlayers.season == season)
        .one_or_none()
    )

    # If we have a fresh cache and we're not forcing refresh, just return it
    if (
        not refresh
        and row
        and row.updated_at
        and (_now() - row.updated_at) < FRESH_FOR
    ):
        return row.players_json or []

    # fetch all pages (no league filter) and flatten
    try:
        items = _get_all_pages(
            f"{BASE_URL}/players",
            {"team": team_id, "season": season},
        ) or []
    except Exception:
        items = []

    # Provider returned nothing? Don't clobber an existing cache
    if not items:
        return (row.players_json or []) if row else []

    if row is None:
        row = TeamSeasonPlayers(
            team_id=team_id,
            season=season,
            players_json=items,
            updated_at=_now(),
        )
        db.add(row)
    else:
        row.players_json = items
        row.updated_at = _now()

    db.commit()
    return items


# ---------- FIXTURE PLAYERS (from /fixtures/players?fixture=ID) ----------

def get_fixture_players_cached(
    db: Session,
    provider_fixture_id: int,
    refresh: bool = False,
) -> dict:
    """
    Returns the fixture-players payload for an API-Football fixture id.
    Reads from cache when fresh; otherwise fetches and updates cache.
    If refresh=True, always refetch from provider (but won't overwrite DB with empty).
    Uses your FixturePlayersCache(fixture_provider_id, payload, updated_at).
    """
    fid = int(provider_fixture_id or 0)
    if fid <= 0:
        return {}

    row = (
        db.query(FixturePlayersCache)
        .filter(FixturePlayersCache.fixture_provider_id == fid)
        .one_or_none()
    )

    # Use cache if fresh and we're not forcing refresh
    if (
        not refresh
        and row
        and row.updated_at
        and (_now() - row.updated_at) < FRESH_FOR
    ):
        return row.payload or {}

    # Fetch from provider
    try:
        data = get_fixture_players(fid) or {}
    except Exception:
        data = {}

    # Provider returned nothing or no response array -> keep existing cache
    if not (isinstance(data, dict) and (data.get("response") or [])):
        return (row.payload or {}) if row else {}

    if row is None:
        row = FixturePlayersCache(
            fixture_provider_id=fid,
            payload=data,
            updated_at=_now(),
        )
        db.add(row)
    else:
        row.payload = data
        row.updated_at = _now()

    db.commit()
    return data