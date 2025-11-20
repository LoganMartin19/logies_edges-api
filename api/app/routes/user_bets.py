# api/app/routes/user_bets.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import User, UserBet, Fixture, Tipster
from ..auth_firebase import get_current_user  # ✅ FIXED IMPORT

router = APIRouter(tags=["user-bets"])

# ---------- Pydantic models ----------

class PlaceBetIn(BaseModel):
    fixture_id: Optional[int] = Field(None, description="Fixture.id this bet relates to")
    market: str
    bookmaker: Optional[str] = None
    price: float
    stake: float
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

def _resolve_local_user(db: Session, claims: dict) -> User:
    """Fetch the User row using db_user_id from auth_firebase."""
    uid = claims.get("db_user_id")
    if not uid:
        raise HTTPException(401, "Local user not found in auth claims")
    user = db.query(User).get(uid)
    if not user:
        raise HTTPException(401, "User row does not exist")
    return user

def _to_out(b: UserBet) -> UserBetOut:
    return UserBetOut.from_orm(b)


# ---------- Routes ----------

@router.post("/user-bets", response_model=UserBetOut)
def create_user_bet(
    payload: PlaceBetIn,
    db: Session = Depends(get_db),
    claims=Depends(get_current_user),   # now returns dict
):
    user = _resolve_local_user(db, claims)

    if payload.fixture_id is not None:
        fx = db.query(Fixture).filter(Fixture.id == payload.fixture_id).one_or_none()
        if not fx:
            raise HTTPException(400, "Fixture not found")

    if payload.source_tipster_id is not None:
        exists = (
            db.query(Tipster.id)
            .filter(Tipster.id == payload.source_tipster_id)
            .scalar()
        )
        if not exists:
            raise HTTPException(400, "Tipster not found")

    b = UserBet(
        user_id=user.id,
        fixture_id=payload.fixture_id,
        market=payload.market,
        bookmaker=payload.bookmaker or "",
        price=float(payload.price),
        stake=float(payload.stake),
        placed_at=datetime.now(timezone.utc),
        source_tipster_id=payload.source_tipster_id,
    )
    db.add(b)
    db.commit()
    db.refresh(b)
    return _to_out(b)


@router.get("/user-bets", response_model=list[UserBetOut])
def list_user_bets(
    db: Session = Depends(get_db),
    claims=Depends(get_current_user),
):
    user = _resolve_local_user(db, claims)

    rows = (
        db.query(UserBet)
        .filter(UserBet.user_id == user.id)
        .order_by(UserBet.placed_at.desc())
        .all()
    )
    return [_to_out(b) for b in rows]