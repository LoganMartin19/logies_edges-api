# api/app/routes/admin_calibration.py
from fastapi import APIRouter, Depends, Query
from typing import Optional, List
from sqlalchemy.orm import Session
from ..db import get_db
from ..services.calibrate import train_and_store_calibration, apply_calibration_to_upcoming
from datetime import datetime, timedelta
from ..edge import compute_edges, STALE_ODDS_HOURS, CAL_BOOK, ensure_baseline_probs
from ..models import Fixture, ModelProb, Edge
router = APIRouter(prefix="/admin", tags=["calibration"])

@router.post("/train-calibration")
def admin_train_calibration(
    market: str = Query(...),
    book: str = Query("bet365"),
    days_back: int = Query(60, ge=7, le=365),
    source: str = Query("consensus_v2"),
    comps: Optional[str] = Query(None),
    scope: str = Query("soccer_global"),
    min_samples: int = Query(10, ge=5, le=200),
    use_all_books: int = Query(1),
    db: Session = Depends(get_db),
):
    if source == "team_form":
        return {
            "ok": False,
            "message": "Calibration not applicable for team_form model (already calibrated + adjusted)."
        }

    comp_list = [c.strip() for c in (comps or "").split(",") if c.strip()] or None

    res = train_and_store_calibration(
        db, market, book, days_back, source, comp_list, scope,
        min_samples=min_samples,
        use_all_books=bool(use_all_books)
    )
    return res

@router.post("/apply-calibration")
def admin_apply_calibration(
    source_in: str = Query("consensus_v2"),
    source_out: str = Query("consensus_calib"),
    book: str = Query("bet365"),
    hours_ahead: int = Query(168, ge=1, le=336),
    comps: Optional[str] = Query(None, description="CSV of Fixture.comp"),
    scope: str = Query("global"),
    db: Session = Depends(get_db),
):
    comp_list = [c.strip() for c in (comps or "").split(",") if c.strip()] or None
    res = apply_calibration_to_upcoming(db, source_in, source_out, book, hours_ahead, comp_list, scope)
    return res

@router.post("/backfill-modelprobs")
def admin_backfill_model_probs(
    source: str = Query("team_form"),
    days_back: int = Query(60, ge=1, le=365),
    comps: Optional[str] = Query(None),
    scope: str = Query("soccer_global"),
    db: Session = Depends(get_db),
):
    from ..edge import ensure_baseline_probs
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days_back)
    end = now

    comp_list = [c.strip() for c in (comps or "").split(",") if c.strip()] or None

    ensure_baseline_probs(
        db=db,
        now=now,
        source=source,
        time_window=(start, end),
        league_comp_filter=comp_list,
        use_closing_when_past=True,
    )
    return {"ok": True, "message": f"Backfilled modelprobs for source={source} over {days_back} days."}

@router.post("/recompute-edges")
def admin_recompute_edges(
    source: str = Query(...),
    hours_ahead: int = Query(72),
    since: Optional[datetime] = Query(None),
    fixture_id: Optional[int] = Query(None),
    min_edge: float = Query(0.0),
    ensure_baseline: int = Query(1),  # ✅ NEW toggle
    db: Session = Depends(get_db),
):
    now = datetime.utcnow()

    if ensure_baseline:
        ensure_baseline_probs(db=db, now=now, source=source)  # ✅ Will trigger strength logs

    num = compute_edges(
        db=db,
        now=now,
        since=since,
        source=source,
        fixture_id=fixture_id,
        min_edge=min_edge,
        hours_ahead=hours_ahead,
        staleness_hours=STALE_ODDS_HOURS,
        prefer_book=CAL_BOOK,
    )

    return {"ok": True, "source": source, "recomputed": num}


from sqlalchemy import func
from ..models import ModelProb

@router.get("/probe-modelprobs")
def admin_probe_model_probs(
    source: str = Query(...),
    days: int = Query(3, ge=1, le=30),
    db: Session = Depends(get_db),
):
    cutoff = datetime.utcnow() - timedelta(days=days)
    results = (
        db.query(func.date(ModelProb.as_of), func.count())
        .filter(ModelProb.source == source)
        .filter(ModelProb.as_of >= cutoff)
        .group_by(func.date(ModelProb.as_of))
        .order_by(func.date(ModelProb.as_of))
        .all()
    )
    return [{"date": str(date), "count": count} for date, count in results]

@router.get("/fixture-probs")
def get_fixture_probs(
    source: str = Query(...),
    day: str = Query(...),
    db: Session = Depends(get_db),
):
    """
    Show model probabilities for a given day, grouped by fixture.
    """
    date_obj = datetime.strptime(day, "%Y-%m-%d").date()
    start = datetime.combine(date_obj, datetime.min.time())
    end = start + timedelta(days=1)

    probs = (
        db.query(ModelProb, Fixture)
        .join(Fixture, Fixture.id == ModelProb.fixture_id)
        .filter(ModelProb.source == source)
        .filter(Fixture.kickoff_utc >= start, Fixture.kickoff_utc < end)  # < not <=
        .all()
    )

    grouped = {}
    for mp, fixture in probs:
        fid = fixture.id
        if fid not in grouped:
            grouped[fid] = {
                "fixture_id": fid,
                "home_team": fixture.home_team,
                "away_team": fixture.away_team,
                "markets": {}
            }
        grouped[fid]["markets"][mp.market] = round(mp.prob, 4)

    return list(grouped.values())

@router.get("/fixture-edges")
def get_fixture_edges(
    source: str = Query(...),
    fixture_id: int = Query(...),
    db: Session = Depends(get_db),
):
    """
    Show all edge entries for a given fixture and model source.
    """
    edges = (
        db.query(Edge, Fixture)
        .join(Fixture, Fixture.id == Edge.fixture_id)
        .filter(Edge.fixture_id == fixture_id)
        .filter(Edge.model_source == source)
        .all()
    )

    result = []
    for edge, fixture in edges:
        result.append({
            "fixture_id": fixture.id,
            "home_team": fixture.home_team,
            "away_team": fixture.away_team,
            "market": edge.market,
            "price": float(edge.price),
            "prob": round(edge.prob, 4),
            "edge": round(edge.edge, 4),
            "bookmaker": edge.bookmaker,
        })

    return result