# api/app/routes/accas.py
from __future__ import annotations
from typing import List, Optional
from datetime import date as date_cls, datetime, timezone, timedelta
from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Fixture, AccaTicket, AccaLeg

router = APIRouter(prefix="/admin/accas", tags=["accas"])
pub = APIRouter(prefix="/api/public/accas", tags=["public-accas"])

def _day_bounds(day_str: str):
    d = date_cls.fromisoformat(day_str)
    start = datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc)
    end   = start + timedelta(days=1)
    return start, end

# ---------- Admin create/update ----------
class AccaLegIn(BaseModel):
    fixture_id: int
    market: str
    price: float
    bookmaker: Optional[str] = None
    note: Optional[str] = None

class AccaIn(BaseModel):
    day: str
    sport: str = "football"
    title: Optional[str] = None
    note: Optional[str] = None
    stake_units: float = Field(1.0, ge=0)
    is_public: bool = False
    legs: List[AccaLegIn]

@router.post("/create")
def create_acca(payload: AccaIn, db: Session = Depends(get_db)):
    d = date_cls.fromisoformat(payload.day)
    if not payload.legs:
        raise HTTPException(400, "At least one leg required")

    # sanity: fixtures exist?
    fids = [l.fixture_id for l in payload.legs]
    exists = {f.id for f in db.query(Fixture).filter(Fixture.id.in_(fids)).all()}
    missing = [x for x in fids if x not in exists]
    if missing:
        raise HTTPException(404, f"Fixtures not found: {missing}")

    # combined price = product of legs
    combined = 1.0
    for l in payload.legs:
        combined *= float(l.price)

    t = AccaTicket(
        day=d,
        sport=payload.sport,
        title=payload.title,
        note=payload.note,
        stake_units=payload.stake_units,
        is_public=payload.is_public,
        combined_price=round(combined, 4),
    )
    db.add(t); db.flush()

    for l in payload.legs:
        db.add(AccaLeg(
            ticket_id=t.id,
            fixture_id=l.fixture_id,
            market=l.market,
            price=float(l.price),
            bookmaker=l.bookmaker or "",
            note=l.note or None,
        ))
    db.commit()
    return {"ok": True, "id": t.id, "combined_price": t.combined_price}

@router.post("/{ticket_id}/publish")
def publish_acca(ticket_id: int, public: int = Query(1), db: Session = Depends(get_db)):
    t = db.query(AccaTicket).filter(AccaTicket.id == ticket_id).one_or_none()
    if not t: raise HTTPException(404, "Ticket not found")
    t.is_public = bool(public)
    db.commit()
    return {"ok": True, "id": t.id, "is_public": t.is_public}

@router.delete("/{ticket_id}")
def delete_acca(ticket_id: int, db: Session = Depends(get_db)):
    n = db.query(AccaTicket).filter(AccaTicket.id == ticket_id).delete(synchronize_session=False)
    db.commit()
    return {"ok": True, "deleted": n}

# ---------- Admin list for a day ----------
@router.get("")
def list_admin_accas(day: str = Query(...), db: Session = Depends(get_db)):
    d = date_cls.fromisoformat(day)
    rows = (
        db.query(AccaTicket)
        .filter(AccaTicket.day == d)
        .order_by(AccaTicket.created_at.asc())
        .all()
    )
    ids = [t.id for t in rows]
    all_legs = db.query(AccaLeg).filter(AccaLeg.ticket_id.in_(ids)).all() if ids else []
    legs_by = {}
    for l in all_legs:
        legs_by.setdefault(l.ticket_id, []).append(l)

    return {
        "day": day,
        "accas": [
            {
                "id": t.id,
                "title": t.title,
                "note": t.note,
                "stake_units": t.stake_units,
                "is_public": t.is_public,
                "combined_price": t.combined_price,
                "legs": [
                    {
                        "fixture_id": l.fixture_id,
                        "market": l.market,
                        "price": l.price,
                        "bookmaker": l.bookmaker,
                        "result": l.result,
                        "note": l.note,
                    } for l in legs_by.get(t.id, [])
                ],
            } for t in rows
        ],
    }

# ---------- Public: SINGLE acca for a day (most recent public) ----------
@pub.get("/today")
def public_acca_today(
    day: str = Query(..., description="YYYY-MM-DD (UTC)"),
    db: Session = Depends(get_db),
):
    d = date_cls.fromisoformat(day)
    t = (
        db.query(AccaTicket)
        .filter(AccaTicket.day == d, AccaTicket.is_public == True)
        .order_by(AccaTicket.created_at.desc())
        .first()
    )
    if not t:
        return {"exists": False, "message": "No public ACCA for this day."}

    legs = db.query(AccaLeg).filter(AccaLeg.ticket_id == t.id).all()
    fids = [l.fixture_id for l in legs if l.fixture_id]
    fmap = {f.id: f for f in db.query(Fixture).filter(Fixture.id.in_(fids)).all()} if fids else {}

    # lightweight summary: N legs, ~price, first/last kickoff, and any leg notes condensed
    kickoffs = [fmap[l.fixture_id].kickoff_utc for l in legs if l.fixture_id in fmap and fmap[l.fixture_id].kickoff_utc]
    kmin = min(kickoffs).strftime("%H:%M") if kickoffs else None
    kmax = max(kickoffs).strftime("%H:%M") if kickoffs else None
    note_bits = []
    for l in legs:
        if l.note:
            fx = fmap.get(l.fixture_id)
            if fx:
                note_bits.append(f"{fx.home_team}–{fx.away_team} ({l.market}): {l.note}")
    summary = f"{len(legs)}-fold at ~{t.combined_price}x. Kickoffs {kmin}–{kmax} UTC." + (f" Notes: " + " | ".join(note_bits) if note_bits else "")

    return {
        "exists": True,
        "ticket_id": t.id,
        "day": t.day.isoformat(),
        "title": t.title or f"{d.strftime('%a %d %b')} ACCA",
        "note": t.note,
        "stake_units": t.stake_units,
        "combined_price": t.combined_price,
        "summary": summary,
        "legs": [
            {
                "fixture_id": l.fixture_id,
                "matchup": f"{fmap[l.fixture_id].home_team} vs {fmap[l.fixture_id].away_team}" if l.fixture_id in fmap else "",
                "comp": fmap[l.fixture_id].comp if l.fixture_id in fmap else "",
                "kickoff_utc": fmap[l.fixture_id].kickoff_utc.isoformat() if l.fixture_id in fmap else None,
                "market": l.market,
                "bookmaker": l.bookmaker,
                "price": l.price,
                "result": l.result,
                "note": l.note,
            } for l in legs
        ],
    }

# ---------- Public: keep your multi-acca daily endpoint (unchanged) ----------
@pub.get("/daily")
def public_accas_daily(day: str = Query(...), db: Session = Depends(get_db)):
    d = date_cls.fromisoformat(day)
    rows = (
        db.query(AccaTicket)
        .filter(AccaTicket.day == d, AccaTicket.is_public == True)
        .order_by(AccaTicket.created_at.asc())
        .all()
    )
    ids = [t.id for t in rows]
    legs = db.query(AccaLeg).filter(AccaLeg.ticket_id.in_(ids)).all() if ids else []
    fids = list({l.fixture_id for l in legs})
    fmap = {f.id: f for f in db.query(Fixture).filter(Fixture.id.in_(fids)).all()} if fids else {}

    legs_by = {}
    for l in legs:
        legs_by.setdefault(l.ticket_id, []).append(l)

    return {
        "day": day,
        "accas": [
            {
                "id": t.id,
                "title": t.title or "ACCA",
                "note": t.note,
                "stake_units": t.stake_units,
                "combined_price": t.combined_price,
                "legs": [
                    {
                        "fixture_id": l.fixture_id,
                        "matchup": f"{fmap[l.fixture_id].home_team} vs {fmap[l.fixture_id].away_team}" if l.fixture_id in fmap else "",
                        "comp": fmap[l.fixture_id].comp if l.fixture_id in fmap else "",
                        "kickoff_utc": fmap[l.fixture_id].kickoff_utc.isoformat() if l.fixture_id in fmap else None,
                        "market": l.market,
                        "bookmaker": l.bookmaker,
                        "price": l.price,
                        "result": l.result,
                        "note": l.note,
                    } for l in legs_by.get(t.id, [])
                ],
            } for t in rows
        ],
    }