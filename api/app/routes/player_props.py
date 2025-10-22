# api/app/routes/player_props.py
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from ..db import get_db
from ..services.player_props import ingest_player_odds_for_fixture

router = APIRouter(prefix="/player-odds", tags=["player-odds"])

@router.post("/ingest")
def ingest(fixture_id: int = Query(...), db: Session = Depends(get_db)):
    try:
        n = ingest_player_odds_for_fixture(db, fixture_id)
        return {"ok": True, "fixture_id": fixture_id, "rows": n}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))