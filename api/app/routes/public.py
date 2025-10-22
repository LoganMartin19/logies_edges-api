# api/app/routes/public.py
from __future__ import annotations

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo  # ✅ for British local time conversion

from ..db import get_db
from ..models import Fixture, Odds, FeaturedPick

pub = APIRouter(prefix="/public", tags=["public"])

# --- helpers -----------------------------------------------------------------

LONDON_TZ = ZoneInfo("Europe/London")

def _parse_day(day_str: str):
    """Parse a YYYY-MM-DD date string and return (start_utc, end_utc)."""
    d = datetime.fromisoformat(day_str).date()
    start = datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start, end

def _norm_sport(s: str) -> str:
    s = (s or "all").strip().lower()
    aliases = {
        "soccer": "football",
        "footy": "football",
        "futbol": "football",
        "fútbol": "football",
    }
    return aliases.get(s, s)

def _to_bst_iso(dt: datetime | None) -> str | None:
    """Convert UTC datetime → Europe/London ISO string (with offset)."""
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local_dt = dt.astimezone(LONDON_TZ)
    return local_dt.isoformat()

# --- public fixtures ---------------------------------------------------------

@pub.get("/fixtures/daily")
def fixtures_daily(
    day: str = Query(..., description="YYYY-MM-DD"),
    sport: str = Query("all"),
    db: Session = Depends(get_db),
):
    s = _norm_sport(sport)
    allowed = {"all", "football", "nba", "nhl", "nfl", "cfb"}
    if s not in allowed:
        raise HTTPException(status_code=400, detail=f"Unknown sport '{sport}'")

    start, end = _parse_day(day)

    q = (
        db.query(Fixture)
        .filter(Fixture.kickoff_utc >= start, Fixture.kickoff_utc < end)
        .order_by(Fixture.kickoff_utc.asc())
    )
    if s != "all":
        q = q.filter(Fixture.sport == s)

    fixtures = q.all()

    out = []
    for f in fixtures:
        best_home = None
        best_away = None

        odds_rows = (
            db.query(Odds)
            .filter(Odds.fixture_id == f.id, Odds.market.in_(["HOME_WIN", "AWAY_WIN"]))
            .all()
        )
        for o in odds_rows:
            try:
                price = float(o.price)
            except Exception:
                continue
            if o.market == "HOME_WIN":
                if (best_home is None) or (price > best_home["price"]):
                    best_home = {"bookmaker": o.bookmaker, "price": price}
            elif o.market == "AWAY_WIN":
                if (best_away is None) or (price > best_away["price"]):
                    best_away = {"bookmaker": o.bookmaker, "price": price}

        out.append({
            "id": f.id,
            "home_team": f.home_team,
            "away_team": f.away_team,
            "comp": f.comp,
            "kickoff_utc": _to_bst_iso(f.kickoff_utc),  # ✅ now in British local time
            "sport": f.sport,
            "best_home": best_home,
            "best_away": best_away,
        })

    return {"day": day, "sport": s, "count": len(out), "fixtures": out}

# --- public curated picks ----------------------------------------------------

@pub.get("/picks/daily")
def public_picks_daily(
    day: str = Query(..., description="YYYY-MM-DD"),
    sport: str = Query("all"),
    db: Session = Depends(get_db),
):
    s = _norm_sport(sport)
    start, end = _parse_day(day)

    q = (
        db.query(FeaturedPick, Fixture)
        .join(Fixture, Fixture.id == FeaturedPick.fixture_id)
        .filter(FeaturedPick.kickoff_utc >= start, FeaturedPick.kickoff_utc < end)
        .order_by(FeaturedPick.created_at.asc())
    )
    if s != "all":
        q = q.filter(FeaturedPick.sport == s)

    rows = q.all()
    picks = []
    for p, f in rows:
        ko = p.kickoff_utc or f.kickoff_utc
        picks.append({
            "pick_id": p.id,
            "fixture_id": f.id,
            "matchup": f"{f.home_team} vs {f.away_team}",
            "home_team": f.home_team,
            "away_team": f.away_team,
            "comp": p.comp or f.comp,
            "sport": p.sport or f.sport,
            "kickoff_utc": _to_bst_iso(ko),  # ✅ BST conversion
            "market": p.market,
            "bookmaker": p.bookmaker,
            "price": p.price,
            "note": p.note,
        })

    return {"day": day, "sport": s, "count": len(picks), "picks": picks}

# --- admin: add/remove curated picks ----------------------------------------

@pub.post("/admin/picks/add")
def admin_picks_add(
    fixture_id: int = Query(...),
    market: str = Query(...),
    bookmaker: str = Query(...),
    price: float = Query(...),
    note: str | None = Query(None),
    day: str | None = Query(None),
    db: Session = Depends(get_db),
):
    f = db.query(Fixture).filter(Fixture.id == fixture_id).one_or_none()
    if not f:
        raise HTTPException(status_code=404, detail="Fixture not found")

    ko = f.kickoff_utc
    if not ko:
        raise HTTPException(status_code=400, detail="Fixture has no kickoff_utc")
    if ko.tzinfo is None:
        ko = ko.replace(tzinfo=timezone.utc)

    chosen_day = datetime.fromisoformat(day).date() if day else ko.date()

    sport = f.sport or "football"
    comp = f.comp or ""
    home_name = f.home_team or ""
    away_name = f.away_team or ""
    if not home_name or not away_name:
        raise HTTPException(status_code=400, detail="Fixture missing team names")

    existing = db.query(FeaturedPick).filter(
        FeaturedPick.day == chosen_day,
        FeaturedPick.fixture_id == f.id,
        FeaturedPick.market == market,
        FeaturedPick.bookmaker == bookmaker,
    ).one_or_none()

    if existing:
        existing.price = float(price)
        if note is not None:
            existing.note = note or None
        db.commit()
        return {"ok": True, "message": "Pick updated", "pick_id": existing.id}

    fp = FeaturedPick(
        fixture_id=f.id,
        day=chosen_day,
        sport=sport,
        comp=comp,
        home_team=home_name,
        away_team=away_name,
        kickoff_utc=ko,
        market=market,
        bookmaker=bookmaker,
        price=float(price),
        note=(note or None),
    )
    db.add(fp)
    db.commit()
    return {"ok": True, "message": "Pick added", "pick_id": fp.id}

@pub.post("/admin/picks/remove")
def admin_picks_remove(
    pick_id: int = Query(...),
    db: Session = Depends(get_db),
):
    p = db.query(FeaturedPick).filter(FeaturedPick.id == pick_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Pick not found")
    db.delete(p)
    db.commit()
    return {"ok": True, "message": "Pick removed", "pick_id": pick_id}