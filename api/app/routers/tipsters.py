# api/app/routers/tipsters.py
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from ..db import get_db
from ..models import Tipster, TipsterPick
from ..services.tipster_perf import compute_tipster_rolling_stats, model_edge_for_pick
from ..auth_firebase import get_current_user

router = APIRouter(prefix="/api/tipsters", tags=["tipsters"])

# --- Schemas ---
class TipsterIn(BaseModel):
    name: str
    username: str
    bio: str | None = None
    avatar_url: str | None = None
    sport_focus: str | None = None
    social_links: dict | None = None

class TipsterOut(BaseModel):
    id: int
    name: str
    username: str
    bio: str | None
    avatar_url: str | None
    sport_focus: str | None
    roi_30d: float
    winrate_30d: float
    profit_30d: float
    picks_30d: int

class PickIn(BaseModel):
    fixture_id: int
    market: str
    bookmaker: str | None = None
    price: float
    stake: float = 1.0

class PickOut(BaseModel):
    id: int
    fixture_id: int
    market: str
    bookmaker: str | None
    price: float
    stake: float
    created_at: datetime
    result: str | None
    profit: float
    model_edge: float | None = None

def _to_tipster_out(c: Tipster) -> dict:
    return {
        "id": c.id,
        "name": c.name,
        "username": c.username,
        "bio": c.bio,
        "avatar_url": c.avatar_url,
        "sport_focus": c.sport_focus,
        "roi_30d": c.roi_30d or 0.0,
        "winrate_30d": c.winrate_30d or 0.0,
        "profit_30d": c.profit_30d or 0.0,
        "picks_30d": c.picks_30d or 0,
    }

# --- helpers ---
def _email_of_tipster(c: Tipster) -> str | None:
    try:
        return (c.social_links or {}).get("email")
    except Exception:
        return None

def _require_owner(username: str, db: Session, user_claims: dict) -> Tipster:
    c = db.query(Tipster).filter(Tipster.username == username).first()
    if not c:
        raise HTTPException(404, "tipster not found")
    email = (user_claims.get("email") or "").lower()
    if (_email_of_tipster(c) or "").lower() != email:
        raise HTTPException(403, "not your profile")
    return c

# --- routes ---
@router.post("", response_model=TipsterOut)
def create_tipster(payload: TipsterIn, db: Session = Depends(get_db), user=Depends(get_current_user)):
    # one tipster per email — allow update if exists
    email = (user.get("email") or "").lower()
    if not email:
        raise HTTPException(400, "Email missing in Firebase token")

    if db.query(Tipster).filter(Tipster.username == payload.username).first():
        # if it exists but belongs to this email, we update instead of 400
        existing = db.query(Tipster).filter(Tipster.username == payload.username).first()
        if existing and (_email_of_tipster(existing) or "").lower() == email:
            existing.name = payload.name
            existing.bio = payload.bio
            existing.avatar_url = payload.avatar_url
            existing.sport_focus = payload.sport_focus
            existing.social_links = (payload.social_links or {}) | {"email": email}
            db.commit(); db.refresh(existing)
            return _to_tipster_out(existing)
        raise HTTPException(400, "username already exists")

    c = Tipster(**payload.model_dump())
    c.social_links = (payload.social_links or {}) | {"email": email}
    db.add(c)
    db.commit()
    db.refresh(c)
    return _to_tipster_out(c)

@router.get("", response_model=list[TipsterOut])
def list_tipsters(db: Session = Depends(get_db)):
    rows = db.query(Tipster).order_by(Tipster.profit_30d.desc()).all()
    return [_to_tipster_out(c) for c in rows]

@router.get("/{username}", response_model=TipsterOut)
def get_tipster(username: str, db: Session = Depends(get_db)):
    c = db.query(Tipster).filter(Tipster.username == username).first()
    if not c:
        raise HTTPException(404, "tipster not found")
    return _to_tipster_out(c)

@router.post("/{username}/picks", response_model=PickOut)
def create_pick(username: str, payload: PickIn, db: Session = Depends(get_db), user=Depends(get_current_user)):
    # only owner can post picks under their username
    c = _require_owner(username, db, user)
    p = TipsterPick(tipster_id=c.id, **payload.model_dump())
    db.add(p); db.commit(); db.refresh(p)
    return {
        "id": p.id, "fixture_id": p.fixture_id, "market": p.market,
        "bookmaker": p.bookmaker, "price": p.price, "stake": p.stake,
        "created_at": p.created_at, "result": p.result, "profit": p.profit,
        "model_edge": model_edge_for_pick(db, p.fixture_id, p.market, p.price)
    }

@router.get("/{username}/picks", response_model=list[PickOut])
def list_picks(username: str, db: Session = Depends(get_db)):
    c = db.query(Tipster).filter(Tipster.username == username).first()
    if not c:
        raise HTTPException(404, "tipster not found")
    rows = (db.query(TipsterPick)
              .filter(TipsterPick.tipster_id == c.id)
              .order_by(TipsterPick.created_at.desc()).all())
    out = []
    for p in rows:
        out.append({
            "id": p.id, "fixture_id": p.fixture_id, "market": p.market,
            "bookmaker": p.bookmaker, "price": p.price, "stake": p.stake,
            "created_at": p.created_at, "result": p.result, "profit": p.profit,
            "model_edge": model_edge_for_pick(db, p.fixture_id, p.market, p.price)
        })
    return out

class SettleIn(BaseModel):
    result: str  # WIN/LOSE/PUSH

def _settle_profit(result: str, stake: float, price: float) -> float:
    if result == "WIN": return stake * (price - 1.0)
    if result == "LOSE": return -stake
    return 0.0

@router.post("/picks/{pick_id}/settle", response_model=PickOut)
def settle_pick(pick_id: int, body: SettleIn, db: Session = Depends(get_db), user=Depends(get_current_user)):
    # TODO: optionally check owner of the pick via its tipster_id and user email (left simple now)
    p = db.query(TipsterPick).get(pick_id)
    if not p:
        raise HTTPException(404, "pick not found")
    p.result = body.result
    p.profit = _settle_profit(p.result, p.stake or 0.0, p.price or 0.0)
    db.commit(); db.refresh(p)
    return {
        "id": p.id, "fixture_id": p.fixture_id, "market": p.market,
        "bookmaker": p.bookmaker, "price": p.price, "stake": p.stake,
        "created_at": p.created_at, "result": p.result, "profit": p.profit,
        "model_edge": model_edge_for_pick(db, p.fixture_id, p.market, p.price)
    }

class LeaderboardRow(BaseModel):
    username: str
    name: str
    roi_30d: float
    winrate_30d: float
    profit_30d: float
    picks_30d: int

@router.get("/leaderboard/top", response_model=list[LeaderboardRow])
def leaderboard_top(limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)):
    rows = (db.query(Tipster)
            .order_by(Tipster.roi_30d.desc())
            .limit(limit)
            .all())
    return [{
        "username": c.username, "name": c.name,
        "roi_30d": c.roi_30d or 0.0, "winrate_30d": c.winrate_30d or 0.0,
        "profit_30d": c.profit_30d or 0.0, "picks_30d": c.picks_30d or 0
    } for c in rows]