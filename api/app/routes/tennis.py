from datetime import datetime, timedelta, timezone, date
from typing import Dict, List, Optional, Tuple
from statistics import median
import re

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from ..db import get_db
from ..models import Fixture, Odds, ModelProb, Edge

router = APIRouter(prefix="/tennis", tags=["tennis"])

MIN_DEC = 1.02
MAX_DEC = 100.0
MIN_BOOKS_FOR_MODEL = 2

def _ok(x: float|None)->bool:
    try:
        v=float(x); return MIN_DEC <= v <= MAX_DEC
    except: return False

def _devig_pair(p1: float, p2: float) -> Tuple[float,float]:
    i1 = 1.0/float(p1); i2 = 1.0/float(p2)
    s = i1+i2
    if s<=0: return (0.0,0.0)
    return (i1/s, i2/s)

def _median_price(arr: List[float]) -> Optional[float]:
    a=[float(x) for x in arr if _ok(x)]
    if len(a) < MIN_BOOKS_FOR_MODEL: return None
    return float(median(a))

def _moneyline_probs_from_db(db: Session, fx_id: int) -> Tuple[Optional[float], Optional[float]]:
    # Expect markets "HOME_WIN"/"AWAY_WIN" style or "PLAYER_A"/"PLAYER_B"
    rows: List[Odds] = db.query(Odds).filter(Odds.fixture_id==fx_id).all()
    a_prices, b_prices = [], []
    for o in rows:
        mk = (o.market or "").upper().replace(" ", "").replace("-", "")
        if mk in ("HOMEWIN","PLAYER_A","A","HOME_WIN"): a_prices.append(float(o.price))
        if mk in ("AWAYWIN","PLAYER_B","B","AWAY_WIN"): b_prices.append(float(o.price))
    mp_a = _median_price(a_prices)
    mp_b = _median_price(b_prices)
    if mp_a and mp_b:
        pa,pb = _devig_pair(mp_a, mp_b)
        return pa,pb
    return None,None

@router.get("/matches")
def list_matches(
    date: str = Query(..., description="YYYY-MM-DD"),
    db: Session = Depends(get_db),
):
    """Return normalized tennis fixtures for that day with quick ML probs/edges if available."""
    day = datetime.fromisoformat(date).date()
    start = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
    end   = start + timedelta(days=1)

    fixtures: List[Fixture] = (
        db.query(Fixture)
        .filter(Fixture.kickoff_utc >= start, Fixture.kickoff_utc < end)
        .filter(Fixture.sport == "tennis")   # <- make sure your ingestor sets this
        .order_by(Fixture.kickoff_utc.asc())
        .all()
    )

    out=[]
    for f in fixtures:
        # quick probs (median de-vig)
        pa,pb = _moneyline_probs_from_db(db, f.id)
        # best price per side for quick edge display
        best_a = db.query(func.max(Odds.price)).filter(Odds.fixture_id==f.id, Odds.market.in_(["HOME_WIN","PLAYER_A","A"])).scalar()
        best_b = db.query(func.max(Odds.price)).filter(Odds.fixture_id==f.id, Odds.market.in_(["AWAY_WIN","PLAYER_B","B"])).scalar()
        edge_a = (pa*best_a - 1.0) if (pa and best_a and _ok(best_a)) else None
        edge_b = (pb*best_b - 1.0) if (pb and best_b and _ok(best_b)) else None
        edge_best = None
        if edge_a is not None or edge_b is not None:
            edge_best = max([e for e in [edge_a, edge_b] if e is not None], default=None)

        out.append({
            "fixture_id": f.id,
            "kickoff_utc": f.kickoff_utc.isoformat(),
            "tour": f.comp,                      # reuse comp for tour name (ATP, WTA, etc.)
            "player_a": f.home_team,
            "player_b": f.away_team,
            "odds_a": best_a,
            "odds_b": best_b,
            "prob_a": pa,
            "prob_b": pb,
            "edge_best": edge_best,
        })
    return {"matches": out}

@router.get("/match/{fixture_id}")
def view_match(fixture_id: int, db: Session = Depends(get_db)):
    f: Fixture|None = db.query(Fixture).filter(Fixture.id==fixture_id).one_or_none()
    if not f:
        raise HTTPException(404, "Fixture not found")
    pa,pb = _moneyline_probs_from_db(db, f.id)
    # odds snapshot
    rows = (
        db.query(Odds)
        .filter(Odds.fixture_id==f.id)
        .order_by(Odds.market.asc(), Odds.last_seen.desc())
        .all()
    )
    # best price each side (quick)
    best_a = db.query(func.max(Odds.price)).filter(Odds.fixture_id==f.id, Odds.market.in_(["HOME_WIN","PLAYER_A","A"])).scalar()
    best_b = db.query(func.max(Odds.price)).filter(Odds.fixture_id==f.id, Odds.market.in_(["AWAY_WIN","PLAYER_B","B"])).scalar()
    edge_a = (pa*best_a - 1.0) if (pa and best_a and _ok(best_a)) else None
    edge_b = (pb*best_b - 1.0) if (pb and best_b and _ok(best_b)) else None

    return {
        "fixture_id": f.id,
        "kickoff_utc": f.kickoff_utc.isoformat(),
        "tour": f.comp,
        "player_a": f.home_team,
        "player_b": f.away_team,
        "odds_a": best_a, "odds_b": best_b,
        "prob_a": pa, "prob_b": pb,
        "edge_a": edge_a, "edge_b": edge_b,
        "books": [
            {"bookmaker": o.bookmaker, "side": o.market, "price": float(o.price)}
            for o in rows
        ],
    }