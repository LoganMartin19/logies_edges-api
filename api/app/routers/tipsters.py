# api/app/routers/tipsters.py
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime
from ..db import get_db
from ..models import Tipster, TipsterPick, Fixture
from ..services.tipster_perf import compute_tipster_rolling_stats, model_edge_for_pick
from ..auth_firebase import get_current_user
from ..services.firebase import get_current_user as get_user_from_header

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
    is_owner: bool = False

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
    # NEW: fixture context
    fixture_label: str | None = None
    fixture_path: str | None = None
    sport: str | None = None

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

# ---------- NEW: fixture helpers ----------
def _fixture_info(db: Session, fixture_id: int) -> dict:
    f: Fixture | None = db.query(Fixture).get(fixture_id)
    if not f:
        return {"fixture_label": None, "fixture_path": None, "sport": None}

    # Best-effort label
    home = getattr(f, "home_name", None) or getattr(f, "home", None) or ""
    away = getattr(f, "away_name", None) or getattr(f, "away", None) or ""
    label = (f"{home} vs {away}" if (home or away) else getattr(f, "title", None) or getattr(f, "name", None) or "Fixture")

    sport = (getattr(f, "sport", "") or "").lower()
    path = None
    if sport in ("football", "soccer"):
        path = f"/fixture/{f.id}"
    elif sport == "nhl":
        path = f"/nhl/game/{f.id}"
    elif sport in ("cfb", "ncaaf"):
        path = f"/cfb/fixture/{f.id}"
    elif sport in ("nba", "basketball"):
        path = f"/basketball/game/{f.id}"
    # If we add NFL game detail later, map it here:
    elif sport in ("nfl", "american_football"):
        path = None  # no per-game page yet

    return {"fixture_label": label, "fixture_path": path, "sport": getattr(f, "sport", None)}

# --- routes ---

@router.post("", response_model=TipsterOut)
def create_tipster(payload: TipsterIn, db: Session = Depends(get_db), user=Depends(get_current_user)):
    email = (user.get("email") or "").lower()
    if not email:
        raise HTTPException(400, "Email missing in Firebase token")

    existing = db.query(Tipster).filter(Tipster.username == payload.username).first()
    if existing:
        if (_email_of_tipster(existing) or "").lower() == email:
            existing.name = payload.name
            existing.bio = payload.bio
            existing.avatar_url = payload.avatar_url
            existing.sport_focus = payload.sport_focus
            existing.social_links = (payload.social_links or {}) | {"email": email}
            db.commit(); db.refresh(existing)
            out = _to_tipster_out(existing); out["is_owner"] = True
            return out
        raise HTTPException(400, "username already exists")

    c = Tipster(**payload.model_dump())
    c.social_links = (payload.social_links or {}) | {"email": email}
    db.add(c); db.commit(); db.refresh(c)
    out = _to_tipster_out(c); out["is_owner"] = True
    return out

@router.get("", response_model=list[TipsterOut])
def list_tipsters(db: Session = Depends(get_db)):
    rows = db.query(Tipster).order_by(Tipster.profit_30d.desc()).all()
    return [{**_to_tipster_out(c), "is_owner": False} for c in rows]

@router.get("/me", response_model=TipsterOut | None)
def get_my_tipster(db: Session = Depends(get_db), user=Depends(get_current_user)):
    email = (user.get("email") or "").lower()
    if not email:
        return None
    rows = db.query(Tipster).all()
    for c in rows:
        if ((c.social_links or {}).get("email") or "").lower() == email:
            return _to_tipster_out(c)
    return None

@router.get("/{username}", response_model=TipsterOut)
def get_tipster(username: str, request: Request, db: Session = Depends(get_db)):
    c = db.query(Tipster).filter(Tipster.username == username).first()
    if not c:
        raise HTTPException(404, "tipster not found")
    viewer = get_user_from_header(request.headers.get("Authorization"))
    viewer_email = (viewer or {}).get("email", "").lower()
    tipster_email = ((_email_of_tipster(c) or "")).lower()
    out = _to_tipster_out(c)
    out["is_owner"] = bool(viewer_email and viewer_email == tipster_email)
    return out

@router.post("/{username}/picks", response_model=PickOut)
def create_pick(username: str, payload: PickIn, db: Session = Depends(get_db), user=Depends(get_current_user)):
    c = _require_owner(username, db, user)
    p = TipsterPick(tipster_id=c.id, **payload.model_dump())
    db.add(p); db.commit(); db.refresh(p)

    extra = _fixture_info(db, p.fixture_id)
    return {
        "id": p.id, "fixture_id": p.fixture_id, "market": p.market,
        "bookmaker": p.bookmaker, "price": p.price, "stake": p.stake,
        "created_at": p.created_at, "result": p.result, "profit": p.profit,
        "model_edge": model_edge_for_pick(db, p.fixture_id, p.market, p.price),
        **extra,
    }

@router.get("/{username}/picks", response_model=list[PickOut])
def list_picks(username: str, db: Session = Depends(get_db)):
    c = db.query(Tipster).filter(Tipster.username == username).first()
    if not c:
        raise HTTPException(404, "tipster not found")

    rows = (db.query(TipsterPick)
            .filter(TipsterPick.tipster_id == c.id)
            .order_by(TipsterPick.created_at.desc())
            .all())

    out = []
    for p in rows:
        extra = _fixture_info(db, p.fixture_id)
        out.append({
            "id": p.id, "fixture_id": p.fixture_id, "market": p.market,
            "bookmaker": p.bookmaker, "price": p.price, "stake": p.stake,
            "created_at": p.created_at, "result": p.result, "profit": p.profit,
            "model_edge": model_edge_for_pick(db, p.fixture_id, p.market, p.price),
            **extra,
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
    p = db.query(TipsterPick).get(pick_id)
    if not p:
        raise HTTPException(404, "pick not found")
    p.result = body.result
    p.profit = _settle_profit(p.result, p.stake or 0.0, p.price or 0.0)
    db.commit(); db.refresh(p)
    extra = _fixture_info(db, p.fixture_id)
    return {
        "id": p.id, "fixture_id": p.fixture_id, "market": p.market,
        "bookmaker": p.bookmaker, "price": p.price, "stake": p.stake,
        "created_at": p.created_at, "result": p.result, "profit": p.profit,
        "model_edge": model_edge_for_pick(db, p.fixture_id, p.market, p.price),
        **extra,
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

@router.post("/picks/{pick_id}/settle", response_model=PickOut)
def settle_pick(
    pick_id: int,
    body: SettleIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    # Validate result
    result = (body.result or "").upper()
    if result not in ("WIN", "LOSE", "PUSH"):
        raise HTTPException(400, "result must be WIN, LOSE, or PUSH")

    # Find pick
    p = db.query(TipsterPick).get(pick_id)
    if not p:
        raise HTTPException(404, "pick not found")

    # Enforce ownership: the pick's tipster must belong to this Firebase email
    tip = db.query(Tipster).get(p.tipster_id)
    if not tip:
        raise HTTPException(404, "tipster not found for pick")
    email_claim = (user.get("email") or "").lower()
    tip_email = ((tip.social_links or {}).get("email") or "").lower()
    if not email_claim or email_claim != tip_email:
        raise HTTPException(403, "not your pick")

    # Settle + save
    p.result = result
    p.profit = _settle_profit(p.result, p.stake or 0.0, p.price or 0.0)
    db.commit(); db.refresh(p)

    extra = _fixture_info(db, p.fixture_id)
    return {
        "id": p.id,
        "fixture_id": p.fixture_id,
        "market": p.market,
        "bookmaker": p.bookmaker,
        "price": p.price,
        "stake": p.stake,
        "created_at": p.created_at,
        "result": p.result,
        "profit": p.profit,
        "model_edge": model_edge_for_pick(db, p.fixture_id, p.market, p.price),
        **extra,
    }