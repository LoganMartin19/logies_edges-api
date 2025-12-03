from __future__ import annotations
from typing import List, Optional
from datetime import date as date_cls, datetime, timezone, timedelta
from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from itertools import combinations
from math import log

from ..db import get_db
from ..models import Fixture, AccaTicket, AccaLeg, Edge

router = APIRouter(prefix="/admin/accas", tags=["accas"])
pub = APIRouter(prefix="/api/public/accas", tags=["public-accas"])

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _day_bounds(day_str: str):
    d = date_cls.fromisoformat(day_str)
    start = datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc)
    end   = start + timedelta(days=1)
    return start, end

# ---------------------------------------------------------
# MANUAL CREATE ACCA
# ---------------------------------------------------------

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

    fids = [l.fixture_id for l in payload.legs]
    exists = {
        f.id for f in db.query(Fixture).filter(Fixture.id.in_(fids)).all()
    }
    missing = [x for x in fids if x not in exists]
    if missing:
        raise HTTPException(404, f"Fixtures not found: {missing}")

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
    db.add(t)
    db.flush()

    for l in payload.legs:
        db.add(
            AccaLeg(
                ticket_id=t.id,
                fixture_id=l.fixture_id,
                market=l.market,
                price=float(l.price),
                bookmaker=l.bookmaker or "",
                note=l.note or None,
            )
        )

    db.commit()
    return {"ok": True, "id": t.id, "combined_price": t.combined_price}

# ---------------------------------------------------------
# PUBLISH / DELETE
# ---------------------------------------------------------

@router.post("/{ticket_id}/publish")
def publish_acca(ticket_id: int, public: int = Query(1), db: Session = Depends(get_db)):
    t = db.query(AccaTicket).filter(AccaTicket.id == ticket_id).one_or_none()
    if not t:
        raise HTTPException(404, "Ticket not found")
    t.is_public = bool(public)
    db.commit()
    return {"ok": True, "id": t.id, "is_public": t.is_public}

@router.delete("/{ticket_id}")
def delete_acca(ticket_id: int, db: Session = Depends(get_db)):
    n = (
        db.query(AccaTicket)
        .filter(AccaTicket.id == ticket_id)
        .delete(synchronize_session=False)
    )
    db.commit()
    return {"ok": True, "deleted": n}

# ---------------------------------------------------------
# LIST ACCAS FOR ADMIN
# ---------------------------------------------------------

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

    all_legs = (
        db.query(AccaLeg).filter(AccaLeg.ticket_id.in_(ids)).all()
        if ids else []
    )

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
                    }
                    for l in legs_by.get(t.id, [])
                ],
            }
            for t in rows
        ],
    }

# ---------------------------------------------------------
# AUTO-ACCA (UPDATED: min_edge optional → can use odds only)
# ---------------------------------------------------------

class AutoAccaIn(BaseModel):
    day: str
    hours_ahead: int = Field(48, ge=1, le=14 * 24)

    # ⬇️ optional now
    min_edge: float | None = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="If set, require edge >= this. If null, ignore edge filter entirely."
    )

    bookmaker: str = "bet365"
    target_odds: float = Field(5.0, ge=1.1)
    legs_min: int = Field(3, ge=2, le=6)
    legs_max: int = Field(4, ge=2, le=8)
    diversify_markets: bool = True
    avoid_same_fixture: bool = True
    is_public: bool = False
    title: Optional[str] = None
    note: Optional[str] = None
    stake_units: float = Field(1.0, ge=0)

    min_price: float = Field(1.4, ge=1.01)
    max_price: float = Field(2.4, ge=1.05)
    target_tolerance_pct: float = Field(0.25, ge=0.05, le=1.0)
    search_pool_size: int = Field(16, ge=6, le=40)


def _market_bucket(m: str) -> str:
    m = (m or "").upper()
    if "HOME_WIN" in m or m == "1": return "1X2"
    if "AWAY_WIN" in m or m == "2": return "1X2"
    if "DRAW" in m or m == "X": return "1X2"
    if "BTTS" in m: return "BTTS"
    if m.startswith("O") or m.startswith("U"): return "TOTALS"
    return "OTHER"


def _format_leg_note(e: Edge, fx: Fixture) -> str:
    try:
        prob = float(e.prob) if e.prob is not None else None
        edge_pct = float(e.edge) * 100 if e.edge is not None else None
        fair = (1.0 / prob) if prob else None

        bits = [
            f"{fx.home_team} v {fx.away_team} • {e.market}",
            f"{e.bookmaker} {float(e.price):.2f}",
        ]
        if prob is not None:
            bits.append(f"model {prob*100:.1f}%")
        if fair is not None:
            bits.append(f"fair {fair:.2f}")
        if edge_pct is not None:
            bits.append(f"edge {edge_pct:.1f}%")

        return " | ".join(bits)
    except:
        return f"{fx.home_team} v {fx.away_team} • {e.market} @ {e.bookmaker} {e.price}"


@router.post("/auto")
def admin_auto_acca(payload: AutoAccaIn, db: Session = Depends(get_db)):
    start, end = _day_bounds(payload.day)
    now = datetime.now(timezone.utc)
    window_end = min(end, now + timedelta(hours=payload.hours_ahead))

    # Base query — no edge required now
    q = (
        db.query(Edge, Fixture)
        .join(Fixture, Edge.fixture_id == Fixture.id)
        .filter(Fixture.kickoff_utc >= now)
        .filter(Fixture.kickoff_utc >= start, Fixture.kickoff_utc < window_end)
    )

    # edge filter only if provided
    if payload.min_edge is not None:
        q = q.filter(Edge.edge >= payload.min_edge)

    q = q.order_by(Edge.edge.desc().nullslast(), Edge.created_at.desc())

    rows = q.all()
    if not rows:
        raise HTTPException(404, "No odds/edges found for this window.")

    # Price band filter
    def _in_band(e: Edge) -> bool:
        try:
            p = float(e.price)
            return payload.min_price <= p <= payload.max_price
        except:
            return False

    rows = [(e, f) for (e, f) in rows if _in_band(e)]
    if not rows:
        raise HTTPException(404, "No markets within price band.")

    # dedupe per fixture, best market wins
    best_by_fx = {}
    for e, f in rows:
        if f.id not in best_by_fx:
            best_by_fx[f.id] = (e, f)

    pool = sorted(
        best_by_fx.values(),
        key=lambda x: float(x[0].edge or 0),
        reverse=True,
    )[: payload.search_pool_size]

    if len(pool) < payload.legs_min:
        raise HTTPException(404, "Too few fixtures for acca.")

    # combination logic
    def _ok(combo):
        if payload.avoid_same_fixture:
            if len({f.id for _, f in combo}) != len(combo):
                return False
        if payload.diversify_markets:
            if len({_market_bucket(e.market) for e, _ in combo}) != len(combo):
                return False
        return True

    log_target = log(payload.target_odds)

    def _score(combo):
        prod = 1.0
        for e, _ in combo:
            prod *= float(e.price)
        return abs(log(prod) - log_target), prod

    best_combo = None
    best_dist = 1e9
    best_prod = None

    tol_low = payload.target_odds * (1 - payload.target_tolerance_pct)
    tol_high = payload.target_odds * (1 + payload.target_tolerance_pct)

    for k in range(payload.legs_min, payload.legs_max + 1):
        for combo in combinations(pool, k):
            combo = list(combo)
            if not _ok(combo):
                continue
            dist, prod = _score(combo)
            inside = tol_low <= prod <= tol_high

            if inside and dist < best_dist:
                best_combo, best_dist, best_prod = combo, dist, prod
            elif best_combo is None and dist < best_dist:
                best_combo, best_dist, best_prod = combo, dist, prod

    if not best_combo:
        raise HTTPException(500, "Could not assemble an acca.")

    d = date_cls.fromisoformat(payload.day)

    t = AccaTicket(
        day=d,
        sport="football",
        title=payload.title or f"{payload.day} Auto {len(best_combo)}-Leg ACCA",
        note=payload.note or f"Auto-generated to target ~{payload.target_odds}x",
        stake_units=payload.stake_units,
        is_public=payload.is_public,
        combined_price=round(best_prod, 2),
    )

    db.add(t)
    db.flush()

    legs_out = []
    for e, f in best_combo:
        note = _format_leg_note(e, f)
        db.add(
            AccaLeg(
                ticket_id=t.id,
                fixture_id=f.id,
                market=e.market,
                bookmaker=e.bookmaker,
                price=float(e.price),
                note=note,
            )
        )
        legs_out.append({
            "fixture_id": f.id,
            "matchup": f"{f.home_team} vs {f.away_team}",
            "comp": f.comp,
            "kickoff_utc": f.kickoff_utc.isoformat() if f.kickoff_utc else None,
            "market": e.market,
            "bookmaker": e.bookmaker,
            "price": float(e.price),
            "edge": float(e.edge) if e.edge is not None else None,
            "note": note,
        })

    db.commit()

    explanation = (
        f"Target {payload.target_odds}x | "
        f"Price band {payload.min_price}-{payload.max_price} | "
        f"Tolerance ±{int(payload.target_tolerance_pct*100)}%"
    )

    return {
        "ok": True,
        "ticket": {
            "id": t.id,
            "day": payload.day,
            "title": t.title,
            "note": t.note,
            "is_public": t.is_public,
            "combined_price": round(best_prod, 2),
            "legs_count": len(legs_out),
            "legs": legs_out,
            "explanation": explanation,
        },
    }

# ---------------------------------------------------------
# PUBLIC ACCA
# ---------------------------------------------------------

@pub.get("/today")
def public_acca_today(day: str, db: Session = Depends(get_db)):
    d = date_cls.fromisoformat(day)
    t = (
        db.query(AccaTicket)
        .filter(AccaTicket.day == d, AccaTicket.is_public == True)
        .order_by(AccaTicket.created_at.desc())
        .first()
    )
    if not t:
        return {"exists": False}

    legs = db.query(AccaLeg).filter(AccaLeg.ticket_id == t.id).all()
    ids = [l.fixture_id for l in legs]

    fmap = (
        {f.id: f for f in db.query(Fixture).filter(Fixture.id.in_(ids)).all()}
        if ids else {}
    )

    return {
        "exists": True,
        "ticket_id": t.id,
        "day": t.day.isoformat(),
        "title": t.title,
        "note": t.note,
        "stake_units": t.stake_units,
        "combined_price": t.combined_price,
        "legs": [
            {
                "fixture_id": l.fixture_id,
                "matchup": f"{fmap[l.fixture_id].home_team} vs {fmap[l.fixture_id].away_team}" if l.fixture_id in fmap else "",
                "market": l.market,
                "bookmaker": l.bookmaker,
                "price": l.price,
                "result": l.result,
                "note": l.note,
            }
            for l in legs
        ],
    }

# ---------------------------------------------------------
# PUBLIC DAILY MULTI ACCA
# ---------------------------------------------------------

@pub.get("/daily")
def public_accas_daily(day: str, db: Session = Depends(get_db)):
    d = date_cls.fromisoformat(day)
    rows = (
        db.query(AccaTicket)
        .filter(AccaTicket.day == d, AccaTicket.is_public == True)
        .order_by(AccaTicket.created_at.asc())
        .all()
    )
    ids = [t.id for t in rows]

    legs = (
        db.query(AccaLeg).filter(AccaLeg.ticket_id.in_(ids)).all()
        if ids else []
    )

    fids = list({l.fixture_id for l in legs})

    fmap = (
        {f.id: f for f in db.query(Fixture).filter(Fixture.id.in_(fids)).all()}
        if fids else {}
    )

    legs_by = {}
    for l in legs:
        legs_by.setdefault(l.ticket_id, []).append(l)

    return {
        "day": day,
        "accas": [
            {
                "id": t.id,
                "title": t.title,
                "note": t.note,
                "stake_units": t.stake_units,
                "combined_price": t.combined_price,
                "legs": [
                    {
                        "fixture_id": l.fixture_id,
                        "matchup": f"{fmap[l.fixture_id].home_team} vs {fmap[l.fixture_id].away_team}" if l.fixture_id in fmap else "",
                        "market": l.market,
                        "bookmaker": l.bookmaker,
                        "price": l.price,
                        "result": l.result,
                        "note": l.note,
                    }
                    for l in legs_by.get(t.id, [])
                ],
            }
            for t in rows
        ],
    }