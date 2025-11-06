# api/app/routes/accas.py
from __future__ import annotations
from typing import List, Optional
from datetime import date as date_cls, datetime, timezone, timedelta
from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from itertools import combinations
from math import log
from ..db import get_db
from ..models import Fixture, AccaTicket, AccaLeg

router = APIRouter(prefix="/admin/accas", tags=["accas"])
pub = APIRouter(prefix="/public/accas", tags=["public-accas"])

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

# ---------- Admin: auto-generate ACCA (3–4 legs around target odds) ----------
from pydantic import BaseModel, Field
from sqlalchemy import or_, func
from ..models import Edge, Fixture, AccaTicket, AccaLeg

class AutoAccaIn(BaseModel):
    day: str                                  # YYYY-MM-DD (UTC)
    hours_ahead: int = Field(48, ge=1, le=14*24)
    min_edge: float = Field(0.03, ge=-1.0, le=1.0)
    bookmaker: str = "bet365"
    target_odds: float = Field(5.0, ge=1.1)
    legs_min: int = Field(3, ge=2, le=6)
    legs_max: int = Field(4, ge=2, le=8)
    diversify_markets: bool = True
    avoid_same_fixture: bool = True
    is_public: bool = False
    title: str | None = None
    note: str | None = None
    stake_units: float = Field(1.0, ge=0)
    # NEW: keep legs in a realistic range for ~5x acca
    min_price: float = Field(1.4, ge=1.01, description="filter out super-short legs")
    max_price: float = Field(2.4, ge=1.05, description="filter out huge prices that blow the product")
    # NEW: accept if within tolerance of target
    target_tolerance_pct: float = Field(0.25, ge=0.05, le=1.0, description="±% tolerance (0.25=±25%)")
    # NEW: search breadth (top N candidates), bigger = slower but better fit
    search_pool_size: int = Field(16, ge=6, le=40)


def _norm(s: str | None) -> str:
    return (s or "").strip().lower()

def _book_norm():
    # SQL-friendly normalization to match "bet365", "Bet 365", etc.
    def _chain(col):
        x = func.lower(col)
        for ch in (" ", "-", "_", "."):
            x = func.replace(x, ch, "")
        return x
    return _chain

def _market_bucket(m: str) -> str:
    m = (_norm(m)).upper()
    if "HOME_WIN" in m or m == "1": return "1X2"
    if "AWAY_WIN" in m or m == "2": return "1X2"
    if "DRAW" in m or m == "X":    return "1X2"
    if "BTTS" in m:                return "BTTS"
    if m.startswith("O") or m.startswith("U"): return "TOTALS"
    return "OTHER"

def _format_leg_note(e: Edge, fx: Fixture) -> str:
    # Keep it short and useful for the UI
    try:
        prob = float(e.prob) if e.prob is not None else None
        edge_pct = float(e.edge) * 100 if e.edge is not None else None
        fair = (1.0 / prob) if prob and prob > 0 else None
        parts = []
        parts.append(f"{fx.home_team} v {fx.away_team} • {e.market}")
        parts.append(f"{e.bookmaker} {float(e.price):.2f}")
        if prob is not None:
            parts.append(f"model {prob*100:.1f}%")
        if fair is not None:
            parts.append(f"fair {fair:.2f}")
        if edge_pct is not None:
            parts.append(f"edge {edge_pct:.1f}%")
        return " | ".join(parts)
    except Exception:
        return f"{fx.home_team} v {fx.away_team} • {e.market} @ {e.bookmaker} {float(e.price):.2f}"

@router.post("/auto")
def admin_auto_acca(payload: AutoAccaIn, db: Session = Depends(get_db)):
    # Time window
    start, end = _day_bounds(payload.day)
    now = datetime.now(timezone.utc)
    window_end = min(end, now + timedelta(hours=payload.hours_ahead))

    # Base query
    q = (
        db.query(Edge, Fixture)
        .join(Fixture, Edge.fixture_id == Fixture.id)
        .filter(Fixture.kickoff_utc >= now)
        .filter(Fixture.kickoff_utc >= start, Fixture.kickoff_utc < window_end)
        .filter(Edge.edge >= payload.min_edge)
        .order_by(Edge.edge.desc(), Edge.created_at.desc())
    )

    # Bookmaker normalization (portable)
    if payload.bookmaker:
        wanted = (payload.bookmaker or "").lower().replace(" ", "").replace("-", "").replace("_", "").replace(".", "")
        # Fetch then filter in Python to avoid DB-specific func.replace chains
        rows_db = q.all()
        rows = []
        for e, f in rows_db:
            bk = (e.bookmaker or "").lower().replace(" ", "").replace("-", "").replace("_", "").replace(".", "")
            if bk == wanted:
                rows.append((e, f))
    else:
        rows = q.all()

    if not rows:
        raise HTTPException(404, "No candidate edges found for that window/bookmaker.")

    # Price band filter
    def _in_price_band(e) -> bool:
        try:
            p = float(e.price)
            return payload.min_price <= p <= payload.max_price
        except Exception:
            return False

    rows = [(e, f) for (e, f) in rows if _in_price_band(e)]
    if not rows:
        raise HTTPException(404, f"No candidates within price band {payload.min_price:.2f}–{payload.max_price:.2f}.")

    # Deduplicate per fixture (keep best edge per fixture)
    best_by_fx: dict[int, tuple[Edge, Fixture]] = {}
    for e, f in rows:
        if f.id not in best_by_fx:
            best_by_fx[f.id] = (e, f)

    # Pool: take top-N by edge
    pool = sorted(best_by_fx.values(), key=lambda t: float(t[0].edge or 0), reverse=True)[:payload.search_pool_size]
    if len(pool) < payload.legs_min:
        raise HTTPException(404, "Too few fixtures after filtering to build an acca.")

    # Helper to test a combo against rules
    def _ok_combo(combo: list[tuple[Edge, Fixture]]) -> bool:
        if payload.avoid_same_fixture:
            fx_ids = {f.id for _, f in combo}
            if len(fx_ids) != len(combo):
                return False
        if payload.diversify_markets:
            buckets = {_market_bucket(e.market or "") for e, _ in combo}
            if len(buckets) != len(combo):
                return False
        return True

    # Product and distance on log scale (more stable)
    log_target = log(payload.target_odds)
    def _score(combo):
        prod = 1.0
        for e, _ in combo:
            prod *= float(e.price)
        return abs(log(prod) - log_target), prod

    # Search all combinations within legs range and pick best within tolerance
    best_combo = None
    best_dist = 1e9
    best_prod = None
    tol_low = payload.target_odds * (1.0 - payload.target_tolerance_pct)
    tol_high = payload.target_odds * (1.0 + payload.target_tolerance_pct)

    for k in range(payload.legs_min, payload.legs_max + 1):
        for combo in combinations(pool, k):
            combo = list(combo)
            if not _ok_combo(combo):
                continue
            dist, prod = _score(combo)
            # Prefer combos inside tolerance; otherwise keep closest
            inside = (tol_low <= prod <= tol_high)
            if inside and dist < best_dist:
                best_combo, best_dist, best_prod = combo, dist, prod
            elif best_combo is None and dist < best_dist:
                best_combo, best_dist, best_prod = combo, dist, prod

    if not best_combo:
        raise HTTPException(500, "Failed to assemble an ACCA combo (after search).")

    # Create ticket
    d = date_cls.fromisoformat(payload.day)
    t = AccaTicket(
        day=d,
        sport="football",
        title=payload.title or f"Daily {len(best_combo)}-Leg ACCA",
        note=payload.note or (
            f"Auto-generated to target ~{payload.target_odds:.2f}x from best {payload.bookmaker} edges; "
            f"legs priced {payload.min_price:.2f}–{payload.max_price:.2f}, "
            f"{'diversified' if payload.diversify_markets else 'no market diversity'}, "
            f"tolerance ±{int(payload.target_tolerance_pct*100)}%."
        ),
        stake_units=payload.stake_units,
        is_public=payload.is_public,
        combined_price=round(best_prod, 2),
    )
    db.add(t); db.flush()

    legs_out = []
    for e, f in best_combo:
        leg_note = _format_leg_note(e, f)
        db.add(AccaLeg(
            ticket_id=t.id,
            fixture_id=f.id,
            market=e.market,
            bookmaker=e.bookmaker,
            price=float(e.price),
            note=leg_note,
        ))
        legs_out.append({
            "fixture_id": f.id,
            "matchup": f"{f.home_team} vs {f.away_team}",
            "comp": f.comp,
            "kickoff_utc": f.kickoff_utc.isoformat() if f.kickoff_utc else None,
            "market": e.market,
            "bookmaker": e.bookmaker,
            "price": float(e.price),
            "edge": float(e.edge) if e.edge is not None else None,
            "note": leg_note,
        })

    db.commit()

    explanation_bits = []
    explanation_bits.append(f"Target {payload.target_odds:.2f}x ±{int(payload.target_tolerance_pct*100)}%")
    explanation_bits.append(f"Price band {payload.min_price:.2f}–{payload.max_price:.2f}")
    explanation_bits.append(f"Diversity: {'on' if payload.diversify_markets else 'off'}")
    explanation_bits.append(f"Search pool {len(pool)} candidates")
    if not (tol_low <= (best_prod or 0) <= tol_high):
        explanation_bits.append("⚠️ Closest outside tolerance due to available prices/markets")

    return {
        "ok": True,
        "ticket": {
            "id": t.id,
            "day": payload.day,
            "title": t.title,
            "note": t.note,
            "is_public": t.is_public,
            "legs_count": len(legs_out),
            "combined_price": round(best_prod, 2) if best_prod else None,
            "target_odds": payload.target_odds,
            "legs": legs_out,
            "explanation": " | ".join(explanation_bits),
        }
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