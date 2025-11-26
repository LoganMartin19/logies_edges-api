from __future__ import annotations

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from ..db import get_db
from ..models import Fixture, Odds, FeaturedPick, User
from ..auth_firebase import optional_user   # ⭐ viewer identity (may be None)

pub = APIRouter(prefix="/public", tags=["public"])

# ---------------------------------------------------------------------------

LONDON_TZ = ZoneInfo("Europe/London")

def _parse_day(day_str: str):
    d = datetime.fromisoformat(day_str).date()
    start = datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start, end

def _norm_sport(s: str) -> str:
    s = (s or "all").strip().lower()
    return {
        "soccer": "football",
        "footy": "football",
        "futbol": "football",
        "fútbol": "football",
    }.get(s, s)

def _to_bst_iso(dt: datetime | None) -> str | None:
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(LONDON_TZ).isoformat()

# ---------------------------------------------------------------------------
# ⭐ Premium helper

def _viewer_is_premium(db: Session, viewer: dict | None) -> bool:
    viewer = viewer or {}
    uid = viewer.get("uid")
    email = (viewer.get("email") or "").lower()

    user_row = None
    if uid:
        user_row = db.query(User).filter(User.firebase_uid == uid).first()
    if not user_row and email:
        user_row = db.query(User).filter(User.email == email).first()

    return bool(user_row and user_row.is_premium)

# ---------------------------------------------------------------------------
# PUBLIC FIXTURES
# ---------------------------------------------------------------------------

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
            except:
                continue

            if o.market == "HOME_WIN":
                if not best_home or price > best_home["price"]:
                    best_home = {"bookmaker": o.bookmaker, "price": price}
            if o.market == "AWAY_WIN":
                if not best_away or price > best_away["price"]:
                    best_away = {"bookmaker": o.bookmaker, "price": price}

        out.append({
            "id": f.id,
            "home_team": f.home_team,
            "away_team": f.away_team,
            "comp": f.comp,
            "kickoff_utc": _to_bst_iso(f.kickoff_utc),
            "sport": f.sport,
            "best_home": best_home,
            "best_away": best_away,
        })

    return {"day": day, "sport": s, "count": len(out), "fixtures": out}

# ---------------------------------------------------------------------------
# PUBLIC FEATURED PICKS (with premium locking)
# ---------------------------------------------------------------------------

@pub.get("/picks/daily")
def public_picks_daily(
    day: str = Query(..., description="YYYY-MM-DD"),
    sport: str = Query("all"),
    db: Session = Depends(get_db),
    viewer=Depends(optional_user),
):
    s = _norm_sport(sport)

    from datetime import date as _date
    chosen_day = _date.fromisoformat(day)

    viewer_is_premium = _viewer_is_premium(db, viewer)

    q = (
        db.query(FeaturedPick, Fixture)
        .join(Fixture, Fixture.id == FeaturedPick.fixture_id)
        .filter(FeaturedPick.day == chosen_day)
        .order_by(FeaturedPick.created_at.asc())
    )
    if s != "all":
        q = q.filter(FeaturedPick.sport == s)

    rows = q.all()
    picks = []

    for r, f in rows:
        # r = FeaturedPick
        # f = Fixture

        matchup = f"{f.home_team} v {f.away_team}"
        ko = r.kickoff_utc or f.kickoff_utc
        local_ko = _to_bst_iso(ko)

        is_premium = bool(getattr(r, "is_premium_only", False))
        locked = is_premium and not viewer_is_premium

        # Base output
        base = {
            "pick_id": r.id,
            "fixture_id": r.fixture_id,
            "matchup": matchup,
            "comp": f.comp,
            "kickoff_utc": local_ko,
            "sport": r.sport,
            "stake": r.stake,
            "is_premium_only": is_premium,
            "result": r.result,
            "settled_at": r.settled_at.isoformat() if r.settled_at else None,
        }

        if locked:
            picks.append({
                **base,
                "market": "Premium pick",
                "bookmaker": None,
                "price": None,
                "note": "Unlock this premium featured pick with CSB Premium.",
                "edge": None,
            })
        else:
            picks.append({
                **base,
                "market": r.market,
                "bookmaker": r.bookmaker,
                "price": float(r.price) if r.price else None,
                "note": r.note,
                "edge": float(r.edge) if r.edge else None,
            })

    return {"day": day, "sport": s, "count": len(picks), "picks": picks}
# ---------------------------------------------------------------------------
# ADMIN ADD/REMOVE PICK (unchanged except compatible with premium field if DB has it)
# ---------------------------------------------------------------------------

@pub.post("/admin/picks/add")
def admin_picks_add(
    fixture_id: int = Query(...),
    market: str = Query(...),
    bookmaker: str = Query(...),
    price: float = Query(...),
    note: str | None = Query(None),
    day: str | None = Query(None),
    is_premium_only: bool = Query(False),   # ⭐ allow premium flag
    db: Session = Depends(get_db),
):
    f = db.query(Fixture).filter(Fixture.id == fixture_id).one_or_none()
    if not f:
        raise HTTPException(status_code=404, detail="Fixture not found")

    ko = f.kickoff_utc
    if ko is None:
        raise HTTPException(status_code=400, detail="Fixture missing kickoff time")
    if ko.tzinfo is None:
        ko = ko.replace(tzinfo=timezone.utc)

    chosen_day = datetime.fromisoformat(day).date() if day else ko.date()

    existing = db.query(FeaturedPick).filter(
        FeaturedPick.day == chosen_day,
        FeaturedPick.fixture_id == f.id,
        FeaturedPick.market == market,
        FeaturedPick.bookmaker == bookmaker,
    ).one_or_none()

    if existing:
        existing.price = float(price)
        existing.note = note or None
        existing.is_premium_only = is_premium_only   # ⭐ update
        db.commit()
        return {"ok": True, "message": "Pick updated", "pick_id": existing.id}

    fp = FeaturedPick(
        fixture_id=f.id,
        day=chosen_day,
        sport=f.sport or "football",
        comp=f.comp,
        home_team=f.home_team,
        away_team=f.away_team,
        kickoff_utc=ko,
        market=market,
        bookmaker=bookmaker,
        price=float(price),
        note=note or None,
        is_premium_only=is_premium_only,   # ⭐ save
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