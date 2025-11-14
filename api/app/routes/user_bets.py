# api/app/routes/user_bets.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import User, UserBet, Fixture, Tipster
from ..routers.auth import get_current_user  # ✅ existing auth dependency

router = APIRouter(tags=["user-bets"])

# ---------- Pydantic models ----------

class PlaceBetIn(BaseModel):
    fixture_id: Optional[int] = Field(None, description="Fixture.id this bet relates to")
    market: str
    bookmaker: Optional[str] = None
    price: float
    stake: float
    # optional: who you copied it from (tipster)
    source_tipster_id: Optional[int] = None

class UserBetOut(BaseModel):
    id: int
    fixture_id: Optional[int]
    market: str
    bookmaker: Optional[str]
    price: float
    stake: float
    result: Optional[str]
    ret: Optional[float]
    pnl: Optional[float]
    placed_at: datetime

    class Config:
        orm_mode = True

# ---------- Helpers ----------

def _to_out(b: UserBet) -> UserBetOut:
    return UserBetOut.from_orm(b)

# ---------- Routes ----------

@router.post("/user-bets", response_model=UserBetOut)
def create_user_bet(
    payload: PlaceBetIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Create a personal bet row tied to the current logged-in user.
    Used by the 'Place Bet' button on the fixture page.
    """
    # Optional: sanity-check fixture exists if provided
    if payload.fixture_id is not None:
        fx = db.query(Fixture).filter(Fixture.id == payload.fixture_id).one_or_none()
        if not fx:
            raise HTTPException(status_code=400, detail="Fixture not found")

    # Optional: validate source_tipster_id if provided
    if payload.source_tipster_id is not None:
        exists = (
            db.query(Tipster.id)
            .filter(Tipster.id == payload.source_tipster_id)
            .scalar()
        )
        if not exists:
            raise HTTPException(status_code=400, detail="Tipster not found")

    b = UserBet(
        user_id=user.id,
        fixture_id=payload.fixture_id,
        market=payload.market,
        bookmaker=payload.bookmaker or "",
        price=float(payload.price),
        stake=float(payload.stake),
        placed_at=datetime.now(timezone.utc),
        source_tipster_id=payload.source_tipster_id,
        # result/ret/pnl stay None until settled
    )
    db.add(b)
    db.commit()
    db.refresh(b)
    return _to_out(b)

@router.get("/user-bets", response_model=list[UserBetOut])
def list_user_bets(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Return all bets for the current user (for future personal dashboards).
    """
    rows = (
        db.query(UserBet)
        .filter(UserBet.user_id == user.id)
        .order_by(UserBet.placed_at.desc())
        .all()
    )
    return [_to_out(b) for b in rows]