# api/app/routes/performance.py
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
import re
from math import isfinite

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from ..db import get_db
from ..models import Fixture, ModelProb, ClosingOdds

router = APIRouter(prefix="/historic", tags=["performance"])

# --- tiny helpers ------------------------------------------------------------

def _parse_total_market(mkt: str) -> Tuple[Optional[str], Optional[float]]:
    # "O2.5" -> ("O", 2.5), "U1.5" -> ("U", 1.5)
    m = re.fullmatch(r"([OU])\s*([0-9]+(?:\.[0-9]+)?)", mkt.strip())
    if not m:
        return None, None
    side = m.group(1)
    line = float(m.group(2))
    return side, line

def _market_win(mkt: str, ft_home: Optional[int], ft_away: Optional[int]) -> Optional[bool]:
    if ft_home is None or ft_away is None:
        return None
    total = ft_home + ft_away
    m = mkt.upper()

    # 1X2
    if m in ("HOME_WIN", "DRAW", "AWAY_WIN"):
        if m == "HOME_WIN": return ft_home > ft_away
        if m == "DRAW":     return ft_home == ft_away
        if m == "AWAY_WIN": return ft_away > ft_home

    # BTTS
    if m == "BTTS_Y": return (ft_home > 0 and ft_away > 0)
    if m == "BTTS_N": return not (ft_home > 0 and ft_away > 0)

    # Totals O/U (main lines only)
    side, line = _parse_total_market(m)
    if side is not None and line is not None:
        if side == "O": return total > line
        if side == "U": return total < line

    return None  # unknown market → skip

# --- 1) Coverage summary: how much data we have to evaluate ------------------

@router.get("/coverage-summary")
def coverage_summary(
    days_back: int = Query(30, ge=1, le=180),
    book: str = Query("bet365"),
    markets: str | None = Query(None, description="CSV; default common markets"),
    comps: str | None = Query(None, description="CSV of Fixture.comp to include"),  # NEW
    db: Session = Depends(get_db),
):
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days_back)

    wanted_markets = (
        [s.strip() for s in markets.split(",") if s.strip()] if markets else
        ["HOME_WIN","DRAW","AWAY_WIN","BTTS_Y","BTTS_N","O2.5","U2.5"]
    )
    comp_list = [s.strip() for s in comps.split(",")] if comps else None  # NEW

    q = (
        db.query(
            ClosingOdds.market,
            func.count(ClosingOdds.id)
        )
        .join(Fixture, Fixture.id == ClosingOdds.fixture_id)
        .filter(Fixture.kickoff_utc >= since, Fixture.kickoff_utc <= now)
        .filter(Fixture.result_settled == True)
        .filter(ClosingOdds.bookmaker == book)
        .filter(ClosingOdds.market.in_(wanted_markets))
    )
    if comp_list:
        q = q.filter(Fixture.comp.in_(comp_list))  # NEW

    rows = q.group_by(ClosingOdds.market).all()
    total = sum(c for _, c in rows)
    by_market = {m: int(c) for (m, c) in rows}

    return {"days_back": days_back, "book": book, "comps": comp_list or [], "total_rows": total, "by_market": by_market}
# --- 2) ROI of positive-edge selections -------------------------------------

@router.get("/roi")
def roi_report(
    market: str = Query(..., description="e.g. O2.5, BTTS_Y, HOME_WIN"),
    book: str = Query("bet365"),
    edge_min: float = Query(0.02, ge=-1.0, le=1.0),
    days_back: int = Query(60, ge=1, le=180),
    source: str = Query("team_form"),
    comps: Optional[str] = Query(None, description="CSV of Fixture.comp names to restrict"),
    countries: Optional[str] = Query(None, description="CSV of Fixture.country names to restrict"),
    db: Session = Depends(get_db),
):
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days_back)

    comp_list = [s.strip() for s in comps.split(",")] if comps else []
    country_list = [s.strip() for s in countries.split(",")] if countries else []

    q = (
        db.query(
            Fixture.id,
            Fixture.full_time_home,
            Fixture.full_time_away,
            ModelProb.prob,
            ClosingOdds.price
        )
        .join(ModelProb, and_(
            ModelProb.fixture_id == Fixture.id,
            ModelProb.market == market,
            ModelProb.source == source
        ))
        .join(ClosingOdds, and_(
            ClosingOdds.fixture_id == Fixture.id,
            ClosingOdds.market == market,
            ClosingOdds.bookmaker == book
        ))
        .filter(Fixture.kickoff_utc >= since, Fixture.kickoff_utc <= now)
        .filter(Fixture.result_settled == True)
    )

    if comp_list:
        q = q.filter(Fixture.comp.in_(comp_list))
    if country_list:
        q = q.filter(Fixture.country.in_(country_list))

    picks = []
    for fx_id, ft_h, ft_a, prob, price in q.all():
        try:
            p = float(prob); o = float(price)
        except Exception:
            continue
        if not (isfinite(p) and isfinite(o) and p > 0 and o > 1.01):
            continue

        edge = p * o - 1.0
        if edge < edge_min:
            continue

        win = _market_win(market, ft_h, ft_a)
        if win is None:
            continue

        ret = (o if win else 0.0)
        pnl = ret - 1.0
        picks.append({"fixture_id": fx_id, "p": p, "o": o, "edge": edge, "win": win, "pnl": pnl})

    n = len(picks)
    roi = (sum(x["pnl"] for x in picks) / n) if n > 0 else 0.0
    hit = (sum(1 for x in picks if x["win"]) / n) if n > 0 else 0.0
    avg_edge = (sum(x["edge"] for x in picks) / n) if n > 0 else 0.0
    avg_odds = (sum(x["o"] for x in picks) / n) if n > 0 else 0.0

    return {
        "market": market,
        "book": book,
        "source": source,
        "days_back": days_back,
        "edge_min": edge_min,
        "n_bets": n,
        "roi_per_bet": roi,
        "hit_rate": hit,
        "avg_edge": avg_edge,
        "avg_odds": avg_odds,
        "sample": picks[:50],
    }
# --- 3) Calibration: bucket prob and compare to reality ----------------------

@router.get("/calibration")
def calibration(
    market: str = Query(...),
    book: str = Query("bet365"),
    days_back: int = Query(60, ge=1, le=180),
    bins: int = Query(10, ge=3, le=20),
    source: str = Query("consensus_v2"),
    comps: str | None = Query(None, description="CSV of exact Fixture.comp names to include"),
    db: Session = Depends(get_db),
):
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days_back)

    # parse optional comps filter
    comp_list = [s.strip() for s in comps.split(",") if s.strip()] if comps else None

    q = (
        db.query(
            ModelProb.prob,
            ClosingOdds.price,
            Fixture.full_time_home,
            Fixture.full_time_away,
        )
        .join(Fixture, Fixture.id == ModelProb.fixture_id)
        .join(
            ClosingOdds,
            and_(
                ClosingOdds.fixture_id == Fixture.id,
                ClosingOdds.market == market,
                ClosingOdds.bookmaker == book,
            ),
        )
        .filter(ModelProb.market == market, ModelProb.source == source)
        .filter(Fixture.kickoff_utc >= since, Fixture.kickoff_utc <= now)
        .filter(Fixture.result_settled == True)
    )

    if comp_list:
        q = q.filter(Fixture.comp.in_(comp_list))

    rows = q.all()
    if not rows:
        return {
            "market": market,
            "book": book,
            "source": source,
            "days_back": days_back,
            "bins": bins,
            "comps": comp_list or [],
            "items": [],
        }

    # bucket by predicted prob
    buckets = [[] for _ in range(bins)]
    for p, _o, ft_h, ft_a in rows:
        try:
            prob = float(p)
        except Exception:
            continue
        if not (0.0 <= prob <= 1.0):
            continue
        b = min(bins - 1, int(prob * bins))  # 0..bins-1
        win = _market_win(market, ft_h, ft_a)
        if win is None:
            continue
        buckets[b].append((prob, 1.0 if win else 0.0))

    out = []
    for i, items in enumerate(buckets):
        if not items:
            out.append({"bin": i, "n": 0, "avg_pred": None, "emp_rate": None})
            continue
        n = len(items)
        avg_pred = sum(p for p, _ in items) / n
        emp = sum(w for _, w in items) / n
        out.append({"bin": i, "n": n, "avg_pred": avg_pred, "emp_rate": emp})

    return {
        "market": market,
        "book": book,
        "source": source,
        "days_back": days_back,
        "bins": bins,
        "comps": comp_list or [],
        "items": out,
    }