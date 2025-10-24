# api/app/routes/picks.py
from __future__ import annotations
from datetime import datetime, timedelta, timezone, date as date_cls
from typing import List, Dict, Optional
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from ..db import get_db
from ..models import Fixture, FeaturedPick, Edge

router = APIRouter(prefix="/admin", tags=["picks"])

# --- helpers ---------------------------------------------------------------

def _day_bounds(day_str: str):
    d = date_cls.fromisoformat(day_str)
    start = datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc)
    end   = start + timedelta(days=1)
    return start, end

def _sport_matcher(sport: str):
    s = (sport or "all").lower()
    def ok(f: Fixture) -> bool:
        comp = (f.comp or "").upper()
        fs = (f.sport or "").lower() if hasattr(f, "sport") else ""
        if s == "all": return True
        if s == "soccer":
            if fs and fs not in ("", "soccer", "football"): 
                return False
            return not any(tag in comp for tag in ("NFL","NCAA","NHL","NBA"))
        if s == "nba":
            return (fs == "nba") or ("NBA" in comp)
        if s == "nhl":
            return (fs == "ice") or ("NHL" in comp) or ("ICE" in comp) or ("HOCKEY" in comp)
        if s == "nfl":
            return (fs == "gridiron") or ("NFL" in comp)
        if s == "cfb":
            return (fs == "gridiron") or ("NCAA" in comp) or ("CFB" in comp) or ("COLLEGE" in comp)
        return True
    return ok

# --- Admin: list fixtures for a day (for the picker UI) --------------------

@router.get("/picks/fixtures")
def admin_picker_fixtures(
    day: str = Query(..., description="YYYY-MM-DD (UTC)"),
    sport: str = Query("all", description="all | soccer | nba | nhl | nfl | cfb"),
    db: Session = Depends(get_db),
):
    start, end = _day_bounds(day)
    rows: List[Fixture] = (
        db.query(Fixture)
        .filter(Fixture.kickoff_utc >= start, Fixture.kickoff_utc < end)
        .order_by(Fixture.kickoff_utc.asc(), Fixture.id.asc())
        .all()
    )
    ok = _sport_matcher(sport)
    rows = [f for f in rows if ok(f)]
    return {
        "day": day,
        "count": len(rows),
        "fixtures": [
            {
                "id": f.id,
                "home": f.home_team,
                "away": f.away_team,
                "comp": f.comp,
                "kickoff_utc": f.kickoff_utc.isoformat(),
            } for f in rows
        ]
    }

# --- Admin: list existing picks for a day ----------------------------------

@router.get("/picks")
def admin_list_picks(
    day: str = Query(..., description="YYYY-MM-DD (UTC)"),
    db: Session = Depends(get_db),
):
    d = date_cls.fromisoformat(day)
    rows: List[FeaturedPick] = (
        db.query(FeaturedPick)
        .filter(FeaturedPick.day == d)
        .order_by(FeaturedPick.created_at.asc())
        .all()
    )
    # join fixture basics
    fids = [r.fixture_id for r in rows]
    fmap = {f.id: f for f in db.query(Fixture).filter(Fixture.id.in_(fids)).all()} if fids else {}
    return {
        "day": day,
        "count": len(rows),
        "picks": [
            {
                "id": r.id,
                "fixture_id": r.fixture_id,
                "sport": r.sport,
                "title": r.title,
                "blurb": r.blurb,
                "is_public": r.is_public,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "fixture": {
                    "home": fmap.get(r.fixture_id).home_team if fmap.get(r.fixture_id) else None,
                    "away": fmap.get(r.fixture_id).away_team if fmap.get(r.fixture_id) else None,
                    "comp":  fmap.get(r.fixture_id).comp if fmap.get(r.fixture_id) else None,
                    "ko":    fmap.get(r.fixture_id).kickoff_utc.isoformat() if fmap.get(r.fixture_id) else None,
                }
            } for r in rows
        ]
    }

# --- Admin: add/update/delete picks ----------------------------------------

from pydantic import BaseModel

class PickIn(BaseModel):
    day: str
    fixture_id: int
    sport: str = "all"
    title: str | None = None
    blurb: str | None = None
    is_public: bool = False

@router.post("/picks")
def admin_save_pick(payload: PickIn, db: Session = Depends(get_db)):
    d = date_cls.fromisoformat(payload.day)
    # ensure fixture exists
    fx = db.query(Fixture).filter(Fixture.id == payload.fixture_id).one_or_none()
    if not fx:
        raise HTTPException(404, "Fixture not found")
    # upsert by (day, fixture_id)
    r = (
        db.query(FeaturedPick)
        .filter(FeaturedPick.day == d, FeaturedPick.fixture_id == payload.fixture_id)
        .one_or_none()
    )
    if r:
        r.sport = payload.sport
        r.title = payload.title
        r.blurb = payload.blurb
        r.is_public = bool(payload.is_public)
    else:
        r = FeaturedPick(
            day=d, fixture_id=payload.fixture_id, sport=payload.sport,
            title=payload.title, blurb=payload.blurb, is_public=bool(payload.is_public)
        )
        db.add(r)
    db.commit()
    return {"ok": True, "id": r.id}

@router.post("/picks/{pick_id}/publish")
def admin_publish_pick(pick_id: int, public: int = Query(1), db: Session = Depends(get_db)):
    r = db.query(FeaturedPick).filter(FeaturedPick.id == pick_id).one_or_none()
    if not r:
        raise HTTPException(404, "Pick not found")
    r.is_public = bool(public)
    db.commit()
    return {"ok": True, "id": r.id, "is_public": r.is_public}

@router.delete("/picks/{pick_id}")
def admin_delete_pick(pick_id: int, db: Session = Depends(get_db)):
    n = db.query(FeaturedPick).filter(FeaturedPick.id == pick_id).delete(synchronize_session=False)
    db.commit()
    return {"ok": True, "deleted": n}

# --- Public endpoints -------------------------------------------------------

pub = APIRouter(prefix="/public", tags=["public"])

@pub.get("/fixtures/today")
def public_fixtures_today(
    day: str | None = Query(None, description="YYYY-MM-DD (UTC). Default=today"),
    sport: str = Query("all", description="all | soccer | nba | nhl | nfl | cfb"),
    db: Session = Depends(get_db),
):
    if not day:
        day = datetime.now(timezone.utc).date().isoformat()
    start, end = _day_bounds(day)

    rows: List[Fixture] = (
        db.query(Fixture)
        .filter(Fixture.kickoff_utc >= start, Fixture.kickoff_utc < end)
        .order_by(Fixture.kickoff_utc.asc(), Fixture.id.asc())
        .all()
    )
    ok = _sport_matcher(sport)
    rows = [f for f in rows if ok(f)]

    return {
        "day": day,
        "sport": sport,
        "count": len(rows),
        "fixtures": [
            {
                "id": f.id,
                "home": f.home_team,
                "away": f.away_team,
                "comp": f.comp,
                "kickoff_utc": f.kickoff_utc.isoformat(),
            } for f in rows
        ]
    }

@pub.get("/picks")
def public_picks(
    day: str | None = Query(None, description="YYYY-MM-DD (UTC). Default=today"),
    db: Session = Depends(get_db),
):
    if not day:
        day = datetime.now(timezone.utc).date().isoformat()
    d = date_cls.fromisoformat(day)

    picks: List[FeaturedPick] = (
        db.query(FeaturedPick)
        .filter(FeaturedPick.day == d, FeaturedPick.is_public == True)
        .order_by(FeaturedPick.created_at.asc())
        .all()
    )

    fids = [p.fixture_id for p in picks]
    fmap = {f.id: f for f in db.query(Fixture).filter(Fixture.id.in_(fids)).all()} if fids else {}

    # optional: include best edge per fixture (if present)
    best_by_fixture: Dict[int, dict] = {}
    if fids:
        # take highest edge row per fixture
        rows = (
            db.query(Edge)
            .filter(Edge.fixture_id.in_(fids))
            .order_by(Edge.fixture_id.asc(), Edge.edge.desc())
            .all()
        )
        for r in rows:
            if r.fixture_id not in best_by_fixture:
                best_by_fixture[r.fixture_id] = {
                    "market": r.market, "bookmaker": r.bookmaker,
                    "price": float(r.price), "edge": float(r.edge)
                }

    return {
        "day": day,
        "count": len(picks),
        "picks": [
            {
                "id": p.id,
                "sport": p.sport,
                "title": p.title,
                "blurb": p.blurb,
                "fixture_id": p.fixture_id,
                "fixture": {
                    "home": fmap[p.fixture_id].home_team if p.fixture_id in fmap else None,
                    "away": fmap[p.fixture_id].away_team if p.fixture_id in fmap else None,
                    "comp":  fmap[p.fixture_id].comp if p.fixture_id in fmap else None,
                    "kickoff_utc": fmap[p.fixture_id].kickoff_utc.isoformat() if p.fixture_id in fmap else None,
                },
                "best_edge": best_by_fixture.get(p.fixture_id)
            } for p in picks
        ]
    }

# --- Public: featured picks record (W-L-V, ROI, list) -----------------------

@pub.get("/picks/record")
def public_picks_record(
    span: str = Query("30d", description="Window: '7d','30d','90d','all'"),
    db: Session = Depends(get_db),
):
    from math import isfinite
    today = datetime.now(timezone.utc)
    if span == "all":
        start_dt = None
    else:
        days = int(span.rstrip("d"))
        start_dt = today - timedelta(days=days)

    q = db.query(FeaturedPick).filter(FeaturedPick.is_public == True)
    if start_dt: q = q.filter(FeaturedPick.created_at >= start_dt)
    rows = q.order_by(FeaturedPick.created_at.desc()).limit(1000).all()

    fids = [r.fixture_id for r in rows]
    fmap = {f.id: f for f in db.query(Fixture).filter(Fixture.id.in_(fids)).all()} if fids else {}

    won = lost = void = 0; staked = returned = 0.0; items = []
    for r in rows:
        stake = float(r.stake or 1.0); price = float(r.price or 0.0)
        res = (r.result or "").lower(); units = 0.0
        if res == "won": won += 1; units = stake * (price - 1.0); returned += stake * price
        elif res == "lost": lost += 1; units = -stake
        elif res == "void": void += 1; returned += stake
        staked += stake
        fx = fmap.get(r.fixture_id)
        items.append({
            "pick_id": r.id, "fixture_id": r.fixture_id, "sport": r.sport, "comp": r.comp,
            "home_team": r.home_team, "away_team": r.away_team,
            "kickoff_utc": r.kickoff_utc.isoformat() if r.kickoff_utc else None,
            "market": r.market, "bookmaker": r.bookmaker, "price": r.price, "edge": r.edge,
            "stake": stake, "result": r.result, "units": round(units, 2), "note": r.note,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })
    pnl = returned - staked; roi = (pnl / staked * 100) if staked else 0.0
    return {"span": span, "summary": {
        "record": {"won": won, "lost": lost, "void": void},
        "staked": round(staked, 2), "returned": round(returned, 2),
        "pnl": round(pnl, 2), "roi": round(roi, 2)
    }, "picks": items}

@router.post("/picks/{pick_id}/stake")
def admin_set_stake(pick_id: int, stake: float = Query(..., ge=0.0), db: Session = Depends(get_db)):
    r = db.query(FeaturedPick).filter(FeaturedPick.id == pick_id).one_or_none()
    if not r: raise HTTPException(404, "Pick not found")
    r.stake = stake; db.commit()
    return {"ok": True, "id": r.id, "stake": r.stake}

@router.post("/picks/{pick_id}/settle")
def admin_settle_pick(pick_id: int, result: str = Query(..., pattern="^(won|lost|void)$"), db: Session = Depends(get_db)):
    r = db.query(FeaturedPick).filter(FeaturedPick.id == pick_id).one_or_none()
    if not r: raise HTTPException(404, "Pick not found")
    r.result = result; r.settled_at = datetime.utcnow(); db.commit()
    return {"ok": True, "id": r.id, "result": r.result}