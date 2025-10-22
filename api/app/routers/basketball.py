# api/app/routers/basketball.py
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from ..db import get_db
from ..models import Fixture, Odds

router = APIRouter(prefix="/basketball", tags=["basketball"])

def _dt(d: str) -> datetime:
    return datetime.fromisoformat(d).replace(tzinfo=timezone.utc)

@router.get("/fixtures")
def list_nba_fixtures(
    start_day: str = Query(..., description="YYYY-MM-DD"),
    ndays: int = Query(3, ge=1, le=14),
    db: Session = Depends(get_db),
):
    start = _dt(f"{start_day}T00:00:00")
    end   = start + timedelta(days=ndays)

    rows: List[Fixture] = (
        db.query(Fixture)
        .filter(Fixture.sport == "nba")
        .filter(Fixture.kickoff_utc >= start, Fixture.kickoff_utc < end)
        .order_by(Fixture.kickoff_utc.asc())
        .all()
    )

    return [
        {
            "id": f.id,
            "provider_id": f.provider_fixture_id,
            "comp": f.comp,          # "NBA"
            "home": f.home_team,
            "away": f.away_team,
            "kickoff_utc": f.kickoff_utc.isoformat(),
        }
        for f in rows
    ]

@router.get("/fixtures/{fixture_id}")
def get_nba_fixture(fixture_id: int, db: Session = Depends(get_db)):
    f = db.query(Fixture).filter(Fixture.id == fixture_id, Fixture.sport == "nba").one_or_none()
    if not f:
        raise HTTPException(status_code=404, detail="Fixture not found")
    odds = (
        db.query(Odds)
        .filter(Odds.fixture_id == fixture_id)
        .order_by(Odds.market.asc(), Odds.bookmaker.asc())
        .all()
    )
    return {
        "fixture": {
            "id": f.id,
            "comp": f.comp,
            "home": f.home_team,
            "away": f.away_team,
            "kickoff_utc": f.kickoff_utc.isoformat(),
        },
        "odds": [
            {"market": o.market, "bookmaker": o.bookmaker, "price": float(o.price), "last_seen": o.last_seen.isoformat()}
            for o in odds
        ],
    }