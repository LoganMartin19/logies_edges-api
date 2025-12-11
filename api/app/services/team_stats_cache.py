# api/app/services/team_stats_cache.py
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import requests
from sqlalchemy.orm import Session

from ..models import TeamSeasonStats

# Local API config – we *don’t* import from apifootball to avoid circulars
BASE_URL = os.getenv("FOOTBALL_API_URL", "https://v3.football.api-sports.io")
API_KEY = os.getenv("FOOTBALL_API_KEY")
API_HOST = os.getenv("RAPIDAPI_HOST", "v3.football.api-sports.io")

HEADERS = {
    "X-RapidAPI-Key": API_KEY or "",
    "X-RapidAPI-Host": API_HOST,
    "Accept": "application/json",
}


def get_team_season_stats_cached(
    db: Session,
    team_id: int,
    league_id: int,
    season: int,
    *,
    refresh: bool = False,
) -> dict:
    """
    DB-backed cache for API-Football:
        /teams/statistics?team={team_id}&league={league_id}&season={season}

    - Stores the *full JSON body* in TeamSeasonStats.stats_json
    - Returns that same JSON (matching requests.get(...).json())
    """
    row: Optional[TeamSeasonStats] = (
        db.query(TeamSeasonStats)
        .filter(
            TeamSeasonStats.team_id == team_id,
            TeamSeasonStats.league_id == league_id,
            TeamSeasonStats.season == season,
        )
        .one_or_none()
    )

    # ✅ Use cached copy if present and no explicit refresh
    if row and not refresh and row.stats_json:
        return row.stats_json

    # 🔄 Fetch fresh from provider
    params = {"team": int(team_id), "league": int(league_id), "season": int(season)}
    r = requests.get(
        f"{BASE_URL}/teams/statistics",
        headers=HEADERS,
        params=params,
        timeout=20,
    )
    r.raise_for_status()
    j = r.json() or {}

    # Normalise in case of weird shapes (API-Football returns dict here)
    if not isinstance(j, dict):
        j = {}

    now = datetime.now(timezone.utc)

    if row:
        row.stats_json = j
        row.updated_at = now
    else:
        row = TeamSeasonStats(
            team_id=team_id,
            league_id=league_id,
            season=season,
            stats_json=j,
            updated_at=now,
        )
        db.add(row)

    db.commit()
    return j