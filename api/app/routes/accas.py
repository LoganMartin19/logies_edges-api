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

# ---------- Admin: auto-generate ACCA (3–4 legs around target odds) ----------
from pydantic import BaseModel, Field
from sqlalchemy import or_, func
from ..models import Edge, Fixture, AccaTicket, AccaLeg

class AutoAccaIn(BaseModel):
    day: str                                  # YYYY-MM-DD (UTC)
    hours_ahead: int = Field(48, ge=1, le=14*24)
    min_edge: float = Field(0.03, ge=-1.0, le=1.0)   # 3%+
    bookmaker: str = "bet365"
    target_odds: float = Field(5.0, ge=1.1)         # aim ~5.0x
    legs_min: int = Field(3, ge=2, le=6)
    legs_max: int = Field(4, ge=2, le=8)
    diversify_markets: bool = True                  # mix markets if possible
    avoid_same_fixture: bool = True
    is_public: bool = False                         # review first by default
    title: str | None = None
    note: str | None = None
    stake_units: float = Field(1.0, ge=0)

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
    """
    Auto-build a 3–4 leg ACCA near target_odds (default ~5.0x) from today's best edges.
    Improvements:
      - Postgres-safe bookmaker match (normalize column in SQL; normalize RHS in Python only).
      - Market diversification: max 1 leg per bucket (1X2 / BTTS / TOTALS / OTHER), with graceful relax
        if we can't hit legs_min.
      - Bounded combo search (top-K pool, try all combos of size [legs_min..legs_max]) to hit target band.
      - Per-leg odds guardrails to avoid silly products.
    """
    import math, re, itertools

    # --- Params / guards ----------------------------------------------------
    start, end = _day_bounds(payload.day)
    now = datetime.now(timezone.utc)
    window_end = min(end, now + timedelta(hours=payload.hours_ahead))

    # Reasonable odds guardrails per leg (tweak to taste)
    MIN_LEG_ODDS = 1.50
    MAX_LEG_ODDS = 3.00

    # Aim for target within ±15%
    TOL = 0.15
    tgt = max(float(payload.target_odds or 5.0), 1.01)
    lo = tgt * (1 - TOL)
    hi = tgt * (1 + TOL)

    # --- Base query: best edges, upcoming today -----------------------------
    q = (
        db.query(Edge, Fixture)
        .join(Fixture, Edge.fixture_id == Fixture.id)
        .filter(Fixture.kickoff_utc >= now)
        .filter(Fixture.kickoff_utc >= start, Fixture.kickoff_utc < window_end)
        .filter(Edge.edge >= payload.min_edge)
        .order_by(Edge.edge.desc(), Edge.created_at.desc())
    )

    # Postgres-safe bookmaker normalization: normalize the COLUMN in SQL,
    # normalize the input in Python (no SQL functions on the bindparam).
    def _book_chain(col):
        x = func.lower(col)
        for ch in (" ", "-", "_", "."):
            x = func.replace(x, ch, "")
        return x

    if payload.bookmaker:
        bk_norm = re.sub(r"[^a-z0-9]+", "", payload.bookmaker.lower())
        q = q.filter(_book_chain(Edge.bookmaker) == bk_norm)

    rows = q.all()
    if not rows:
        raise HTTPException(404, "No candidate edges found for that window/bookmaker.")

    # --- Pick best per fixture (avoid repeats) & apply leg odds guardrails ---
    best_per_fixture: dict[int, tuple[Edge, Fixture]] = {}
    for e, f in rows:
        try:
            price = float(e.price)
        except Exception:
            continue
        if not (MIN_LEG_ODDS <= price <= MAX_LEG_ODDS):
            continue
        if payload.avoid_same_fixture and f.id in best_per_fixture:
            continue
        # first time we see a fixture is the highest edge due to ordering
        best_per_fixture.setdefault(f.id, (e, f))

    if not best_per_fixture:
        raise HTTPException(404, "No suitable fixtures survived filtering.")

    # --- Build a candidate pool (top-K by edge) -----------------------------
    cands = list(best_per_fixture.values())
    cands.sort(key=lambda t: float(t[0].edge or 0), reverse=True)
    K = max(24, payload.legs_max * 8)  # bounded pool for the combinatorial step
    cands = cands[:K]

    # --- Enforce market diversification (at most one per bucket initially) ---
    def bucket_of(t):
        e, _ = t
        return _market_bucket(e.market or "")

    diversified: list[tuple[Edge, Fixture]] = []
    seen_buckets: set[str] = set()
    if payload.diversify_markets:
        for t in cands:
            b = bucket_of(t)
            if b in seen_buckets:
                continue
            seen_buckets.add(b)
            diversified.append(t)
        # If we still don't have enough for legs_min, relax and add more (even if bucket repeats)
        if len(diversified) < payload.legs_min:
            for t in cands:
                if t not in diversified:
                    diversified.append(t)
                if len(diversified) >= max(payload.legs_max * 3, payload.legs_min + 4):
                    break
        pool = diversified
    else:
        pool = cands

    if len(pool) < payload.legs_min:
        raise HTTPException(404, "Not enough candidates to form an ACCA with current constraints.")

    # --- Combo search: try all combos in [legs_min..legs_max] ---------------
    best_combo = None
    best_diff = float("inf")
    best_prod = None

    def prod_odds(combo):
        p = 1.0
        for (e, _) in combo:
            p *= float(e.price)
        return p

    # Prefer combos inside the band; among those pick closest to target.
    # If none in band, pick globally closest.
    for r in range(payload.legs_min, payload.legs_max + 1):
        # quick optimization: if we demanded diversification, skip combos that duplicate bucket
        if payload.diversify_markets:
            def valid_combo(c):
                bs = set()
                for t in c:
                    b = bucket_of(t)
                    if b in bs:
                        return False
                    bs.add(b)
                return True
        else:
            def valid_combo(c): return True

        # iterate over combos (bounded by K)
        for combo in itertools.combinations(pool, r):
            if not valid_combo(combo):
                continue
            p = prod_odds(combo)
            diff = abs(p - tgt)
            inside = (lo <= p <= hi)
            if inside:
                # Prefer inside-band and closer to target
                if diff < best_diff:
                    best_diff = diff
                    best_combo = combo
                    best_prod = p
            else:
                # Track closest overall if we never find an inside-band combo
                if best_combo is None and diff < best_diff:
                    best_diff = diff
                    best_combo = combo
                    best_prod = p

    if not best_combo:
        raise HTTPException(500, "Failed to assemble an ACCA combo.")

    legs = list(best_combo)
    combined = round(best_prod or prod_odds(legs), 2)

    # --- Create ticket ------------------------------------------------------
    d = date_cls.fromisoformat(payload.day)
    t = AccaTicket(
        day=d,
        sport=payload.sport or "football",
        title=payload.title or f"Daily {len(legs)}-Leg ACCA",
        note=payload.note or f"Auto-generated near {tgt:.2f}x from best {payload.bookmaker} edges.",
        stake_units=payload.stake_units,
        is_public=payload.is_public,
        combined_price=combined,
    )
    db.add(t); db.flush()

    legs_out = []
    for e, f in legs:
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

    return {
        "ok": True,
        "ticket": {
            "id": t.id,
            "day": payload.day,
            "title": t.title,
            "note": t.note,
            "is_public": t.is_public,
            "legs_count": len(legs_out),
            "combined_price": t.combined_price,
            "target_odds": tgt,
            "legs": legs_out,
            "explanation": (
                f"Built from best {payload.bookmaker} edges with per-leg odds "
                f"{MIN_LEG_ODDS:.2f}–{MAX_LEG_ODDS:.2f}. "
                f"{'Diversified by market' if payload.diversify_markets else 'Market repeats allowed'}. "
                f"Searched combos of {payload.legs_min}–{payload.legs_max} legs to hit ~{tgt:.2f}x "
                f"(accept band {lo:.2f}–{hi:.2f})."
            ),
        },
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