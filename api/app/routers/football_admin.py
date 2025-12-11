# api/app/routers/football_admin.py
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from datetime import datetime, timezone, timedelta

from ..db import get_db
from ..models import Fixture
from ..services.apifootball import get_fixture
from ..services.player_cache import (
    get_team_season_players_cached,
    get_fixture_players_cached,
)

# 🔥 NEW: fixture-level cache helpers (you should already have these in services/fixture_cache.py)
from ..services.fixture_cache import (
    get_fixture_detail_cached,
    get_fixture_stats_cached,
    get_fixture_events_cached,
)

admin = APIRouter(prefix="/football/admin", tags=["football-admin"])


@admin.post("/prime-players")
def prime_players(
    day: str = Query(..., description="UTC YYYY-MM-DD"),
    days: int = Query(1, ge=1, le=7),
    refresh: bool = Query(False, description="Force refresh from provider"),
    db: Session = Depends(get_db),
):
    """
    Prime player-level caches for fixtures between [day, day+days):

      - TeamSeasonPlayers (all comps for this season) for both teams
      - FixturePlayersCache (/fixtures/players for the match)

    This is used by:
      - /football/players
      - /football/season-players
      - /football/player-summary
      - /football/player/game-log
      - /football/player-props/fair
      - /football/preview (top_players block)
    """
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    end = start + timedelta(days=days)

    fixtures = (
        db.query(Fixture)
        .filter(Fixture.kickoff_utc >= start, Fixture.kickoff_utc < end)
        .filter(Fixture.provider_fixture_id.isnot(None))
        .all()
    )

    primed = 0
    skipped = 0

    for f in fixtures:
        try:
            pfx = int(f.provider_fixture_id)
            fx_json = get_fixture(pfx) or {}
            fr = (fx_json.get("response") or [None])[0] or {}

            league_block = fr.get("league") or {}
            season = int(league_block.get("season") or 0)

            teams_block = fr.get("teams") or {}
            home_id = int((teams_block.get("home") or {}).get("id") or 0)
            away_id = int((teams_block.get("away") or {}).get("id") or 0)

            if season and home_id:
                get_team_season_players_cached(db, home_id, season, refresh=refresh)
            if season and away_id:
                get_team_season_players_cached(db, away_id, season, refresh=refresh)

            # Match-level /fixtures/players payload
            get_fixture_players_cached(db, pfx, refresh=refresh)

            primed += 1
        except Exception:
            skipped += 1
            continue

    return {
        "day": day,
        "days": days,
        "refresh": refresh,
        "primed_fixtures": primed,
        "skipped": skipped,
    }


@admin.post("/prime-fixtures")
def prime_fixtures(
    day: str = Query(..., description="UTC YYYY-MM-DD"),
    days: int = Query(1, ge=1, le=7),
    refresh: bool = Query(False, description="Force refresh from provider"),
    db: Session = Depends(get_db),
):
    """
    Prime fixture-level caches for all fixtures between [day, day+days):

      - Fixture detail (/fixtures?id=...)
      - Fixture statistics (/fixtures/statistics?fixture=...)
      - Fixture events (/fixtures/events?fixture=...)
    """
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    end = start + timedelta(days=days)

    fixtures = (
        db.query(Fixture)
        .filter(Fixture.kickoff_utc >= start, Fixture.kickoff_utc < end)
        .filter(Fixture.provider_fixture_id.isnot(None))
        .all()
    )

    primed = 0
    skipped = 0
    errors: list[dict] = []  # 👈 collect some error info

    for f in fixtures:
        try:
            pfx = int(f.provider_fixture_id)

            # DB-backed caches (no refresh kwarg now)
            get_fixture_detail_cached(db, pfx)
            get_fixture_stats_cached(db, pfx)
            get_fixture_events_cached(db, pfx)

            primed += 1
        except Exception as e:
            skipped += 1
            # record a tiny bit of context for debugging
            errors.append({
                "fixture_id": f.id,
                "provider_fixture_id": f.provider_fixture_id,
                "error": repr(e),
            })
            # don't blow up, just continue

    return {
        "day": day,
        "days": days,
        "refresh": refresh,
        "primed_fixtures": primed,
        "skipped": skipped,
        # 👇 only show first few errors so the JSON isn't insane
        "errors_sample": errors[:5],
    }