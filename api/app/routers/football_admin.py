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

admin = APIRouter(prefix="/football/admin", tags=["football-admin"])


@admin.post("/prime-players")
def prime_players(
    day: str = Query(..., description="UTC YYYY-MM-DD"),
    days: int = Query(1, ge=1, le=7),
    refresh: bool = Query(False, description="Force refresh from provider"),
    db: Session = Depends(get_db),
):
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    end = start + timedelta(days=days)

    fixtures = (
        db.query(Fixture)
        .filter(Fixture.kickoff_utc >= start, Fixture.kickoff_utc < end)
        .filter(Fixture.provider_fixture_id.isnot(None))
        .all()
    )

    primed = 0
    for f in fixtures:
        try:
            pfx = int(f.provider_fixture_id)
            fx_json = get_fixture(pfx)
            fr = (fx_json.get("response") or [None])[0] or {}
            season = int((fr.get("league") or {}).get("season") or 0)
            home_id = int(((fr.get("teams") or {}).get("home") or {}).get("id") or 0)
            away_id = int(((fr.get("teams") or {}).get("away") or {}).get("id") or 0)

            if season and home_id:
                get_team_season_players_cached(db, home_id, season, refresh=refresh)
            if season and away_id:
                get_team_season_players_cached(db, away_id, season, refresh=refresh)

            get_fixture_players_cached(db, pfx, refresh=refresh)
            primed += 1
        except Exception:
            continue

    return {"day": day, "days": days, "refresh": refresh, "primed": primed}