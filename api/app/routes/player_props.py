# api/app/routes/player_props.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Fixture, PlayerOdds
from ..services.player_props import (
    ingest_player_odds_for_fixture,
    fetch_player_odds_raw_for_fixture,  # ✅ this exists in your services/player_props.py
)

router = APIRouter(prefix="/api/player-odds", tags=["player-odds"])


@router.post("/ingest")
def ingest(
    fixture_id: int = Query(..., description="Internal fixture id (fixtures.id)"),
    db: Session = Depends(get_db),
):
    """
    Pull API-Football player odds for this fixture and upsert into PlayerOdds table.
    """
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx:
        raise HTTPException(status_code=404, detail="Fixture not found")

    if not fx.provider_fixture_id:
        raise HTTPException(status_code=400, detail="Fixture missing provider_fixture_id")

    try:
        n = ingest_player_odds_for_fixture(db, fixture_id)
        return {
            "ok": True,
            "fixture_id": fixture_id,
            "provider_fixture_id": int(fx.provider_fixture_id),
            "rows_upserted": n,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/raw")
def raw(
    fixture_id: int = Query(..., description="Internal fixture id (fixtures.id)"),
    db: Session = Depends(get_db),
):
    """
    Debug endpoint: returns the raw API payload for player odds.
    Use this when a market looks confusing (Player Singles/Doubles/Triples etc).
    """
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx:
        raise HTTPException(status_code=404, detail="Fixture not found")

    if not fx.provider_fixture_id:
        raise HTTPException(status_code=400, detail="Fixture missing provider_fixture_id")

    try:
        payload = fetch_player_odds_raw_for_fixture(db, fixture_id)
        return {
            "fixture_id": fixture_id,
            "provider_fixture_id": int(fx.provider_fixture_id),
            "raw": payload,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
def list_odds(
    fixture_id: int = Query(..., description="Internal fixture id (fixtures.id)"),
    market: str | None = Query(
        None,
        description="Filter canonical market key (lowercase) e.g. shots, sot, fouls, yellow",
    ),
    bookmaker: str | None = Query(None),
    player: str | None = Query(None, description="Case-insensitive substring match on player_name"),
    limit: int = Query(200, ge=1, le=2000),
    include_meta: bool = Query(True, description="Include markets/bookmakers counts for sanity checks"),
    db: Session = Depends(get_db),
):
    """
    Reads what you've stored in DB after ingest.
    """
    q = db.query(PlayerOdds).filter(PlayerOdds.fixture_id == fixture_id)

    if bookmaker:
        q = q.filter(PlayerOdds.bookmaker == bookmaker)

    # ✅ your ingestion stores canonical keys lowercased
    if market:
        q = q.filter(PlayerOdds.market == market.strip().lower())

    if player:
        q = q.filter(PlayerOdds.player_name.ilike(f"%{player.strip()}%"))

    rows = (
        q.order_by(PlayerOdds.market.asc(), PlayerOdds.player_name.asc(), PlayerOdds.line.asc().nullsfirst())
        .limit(limit)
        .all()
    )

    out = {
        "fixture_id": fixture_id,
        "count": len(rows),
        "rows": [
            {
                "player_id": r.player_id,
                "player_name": r.player_name,
                "market": r.market,
                "line": r.line,
                "bookmaker": r.bookmaker,
                "price": float(r.price),
                "last_seen": r.last_seen.isoformat() if r.last_seen else None,
            }
            for r in rows
        ],
    }

    if include_meta:
        out["markets"] = sorted({r.market for r in rows if r.market})
        out["bookmakers"] = sorted({r.bookmaker for r in rows if r.bookmaker})

    return out