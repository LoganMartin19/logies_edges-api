# api/app/routers/ingest.py
from __future__ import annotations

from datetime import date, datetime, timezone, timedelta
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Form, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..edge import ensure_baseline_probs, compute_edges
from ..settings import settings
from ..models import Fixture, Odds

# --- Football (soccer) ---
from ..services.fixtures import (
    ingest_fixtures_and_odds as ingest_football,
    ingest_odds_for_fixture_id as refresh_football_odds,
)

# --- NHL (hockey) ---
# Be tolerant to differing function names
ingest_nhl = None
refresh_nhl_odds = None
try:
    # preferred names (match your posted file if present)
    from ..services.fixtures_ice import (
        ingest_ice_and_odds as ingest_nhl,
        ingest_odds_for_game_id as refresh_nhl_odds,
    )
except Exception:
    try:
        # older alias used in some branches
        from ..services.fixtures_ice import (
            ingest_games_and_odds as ingest_nhl,
            ingest_odds_for_game_id as refresh_nhl_odds,  # may still be absent
        )
    except Exception:
        ingest_nhl = None
        refresh_nhl_odds = None

# --- NFL (gridiron) ---
try:
    from ..services.fixtures_gridiron import (
        ingest_games_and_odds as ingest_nfl,
        ingest_odds_for_game_id as refresh_nfl_odds,
    )
except Exception:
    ingest_nfl = None
    refresh_nfl_odds = None

# --- NBA (basketball) ---
ingest_nba = None
refresh_nba_odds = None
try:
    from ..services.fixtures_nba import (
        ingest_nba_and_odds as ingest_nba,
        ingest_odds_for_game_id as refresh_nba_odds,  # optional; may not exist yet
    )
except Exception:
    # allow ingest without single-fixture refresh if you haven’t added it yet
    try:
        from ..services.fixtures_nba import ingest_nba_and_odds as ingest_nba
    except Exception:
        ingest_nba = None
    refresh_nba_odds = None


router = APIRouter(prefix="/ingest", tags=["ingest"])

# ---- process-local status cache (resets on server restart) ----
LAST_INGEST: Dict[str, Any] = {
    "ok": None,
    "at": None,
    "sport": None,
    "day": None,
    "leagues": [],
    "fixtures_upserted": 0,
    "odds_rows_written": 0,
    "edges_recomputed": False,
    "message": None,
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _compute_if_requested(db: Session, do_it: int) -> bool:
    if not do_it:
        return False
    now = datetime.now(timezone.utc)
    ensure_baseline_probs(db, now)
    compute_edges(db, now, settings.EDGE_MIN)
    return True

def _update_last(status: dict) -> None:
    LAST_INGEST.update({
        "at": datetime.now(timezone.utc).isoformat(),
        **status,
    })

def _sport_safe(s: Optional[str]) -> str:
    x = (s or "").lower()
    if x in ("football", "soccer"): return "football"
    if x in ("nhl", "hockey"):      return "nhl"
    if x in ("nfl", "gridiron"):    return "nfl"
    if x in ("nba", "basketball"):  return "nba"
    return "football"

def _refresh_odds_for_fixture(db: Session, f: Fixture, prefer_book: Optional[str], allow_fallback: bool) -> dict:
    """
    Unified odds refresher that routes to the correct provider based on f.sport.
    """
    sport = (f.sport or "").lower()
    # NBA
    if sport == "nba" and refresh_nba_odds:
        return refresh_nba_odds(
            db,
            provider_game_id=f.provider_fixture_id,
            prefer_book=prefer_book,
            allow_fallback=allow_fallback,
        )
    # NHL
    if sport == "nhl" and refresh_nhl_odds:
        return refresh_nhl_odds(
            db,
            provider_game_id=f.provider_fixture_id,
            prefer_book=prefer_book,
            allow_fallback=allow_fallback,
        )
    # NFL / CFB
    if sport in ("nfl", "cfb") and refresh_nfl_odds:
        return refresh_nfl_odds(
            db,
            provider_game_id=f.provider_fixture_id,
            prefer_book=prefer_book,
            allow_fallback=allow_fallback,
        )
    # default: football (soccer)
    return refresh_football_odds(
        db,
        provider_fixture_id=f.provider_fixture_id,
        prefer_book=prefer_book,
        allow_fallback=allow_fallback,
    )

# ---------------------------------------------------------------------
# Football (keeps your legacy /fixtures route as an alias)
# ---------------------------------------------------------------------

@router.post("/fixtures")
def ingest_fixtures_legacy(
    day: date = Form(...),
    leagues: List[str] | None = Form(None),
    max_fixtures: int = Form(30),
    odds_delay_sec: float = Form(0.30),
    compute_after: int = Form(1),
    db: Session = Depends(get_db),
):
    """Legacy alias for football ingestion to avoid breaking the FE."""
    return ingest_football_route(
        day=day,
        leagues=leagues,
        max_fixtures=max_fixtures,
        odds_delay_sec=odds_delay_sec,
        compute_after=compute_after,
        db=db,
    )

@router.post("/football")
def ingest_football_route(
    day: date = Form(...),
    leagues: List[str] | None = Form(None),
    max_fixtures: int = Form(30),
    odds_delay_sec: float = Form(0.30),
    compute_after: int = Form(1),
    db: Session = Depends(get_db),
):
    """
    Ingest FOOTBALL (soccer) fixtures + odds for a given day/leagues.
    """
    try:
        league_list = leagues or ["EPL", "CHAMP", "SCO_PREM", "LA_LIGA", "UCL"]
        res = ingest_football(
            db,
            day,
            league_list,
            prefer_book=None,
            max_fixtures=max_fixtures,
            odds_delay_sec=odds_delay_sec,
        )
        did_compute = _compute_if_requested(db, compute_after)
        _update_last({
            "ok": True,
            "sport": "football",
            "day": str(day),
            "leagues": league_list,
            "fixtures_upserted": res.get("fixtures_upserted", 0),
            "odds_rows_written": res.get("odds_rows_written", 0),
            "edges_recomputed": did_compute,
            "message": None,
        })
        return {"ok": True, "ingest": res, "computed": did_compute}
    except Exception as e:
        _update_last({"ok": False, "sport": "football", "day": str(day), "leagues": leagues or [], "message": f"{e!r}"})
        raise HTTPException(status_code=502, detail=f"football ingest error: {e!r}")

# ---------------------------------------------------------------------
# NHL
# ---------------------------------------------------------------------

@router.post("/nhl")
def ingest_nhl_route(
    day: date = Form(...),
    leagues: List[str] | None = Form(None),  # e.g. ["NHL"]
    season: Optional[int] = Form(None),      # pass season if your service supports it
    max_games: int = Form(50),
    odds_delay_sec: float = Form(0.30),
    compute_after: int = Form(0),
    db: Session = Depends(get_db),
):
    if not ingest_nhl:
        raise HTTPException(status_code=501, detail="NHL ingester not available (fixtures_ice missing)")
    try:
        league_list = leagues or ["NHL"]
        # Support both signatures (with/without season)
        try:
            res = ingest_nhl(
                db,
                day,
                league_list,
                season=season,
                prefer_book=None,
                max_games=max_games,
                odds_delay_sec=odds_delay_sec,
            )
        except TypeError:
            # older signature without season
            res = ingest_nhl(
                db,
                day,
                league_list,
                prefer_book=None,
                max_games=max_games,
                odds_delay_sec=odds_delay_sec,
            )
        did_compute = _compute_if_requested(db, compute_after)
        _update_last({
            "ok": True,
            "sport": "nhl",
            "day": str(day),
            "leagues": league_list,
            "fixtures_upserted": res.get("fixtures_upserted", 0),
            "odds_rows_written": res.get("odds_rows_written", 0),
            "edges_recomputed": did_compute,
            "message": None,
        })
        return {"ok": True, "ingest": res, "computed": did_compute}
    except Exception as e:
        _update_last({"ok": False, "sport": "nhl", "day": str(day), "leagues": leagues or [], "message": f"{e!r}"})
        raise HTTPException(status_code=502, detail=f"NHL ingest error: {e!r}")

# ---------------------------------------------------------------------
# NFL
# ---------------------------------------------------------------------

@router.post("/nfl")
def ingest_nfl_route(
    day: date = Form(...),
    leagues: List[str] | None = Form(None),  # ["NFL"]
    max_games: int = Form(50),
    odds_delay_sec: float = Form(0.30),
    compute_after: int = Form(0),
    db: Session = Depends(get_db),
):
    if not ingest_nfl:
        raise HTTPException(status_code=501, detail="NFL ingester not available (fixtures_gridiron missing)")
    try:
        league_list = leagues or ["NFL"]
        res = ingest_nfl(
            db,
            day,
            league_list,
            prefer_book=None,
            max_games=max_games,
            odds_delay_sec=odds_delay_sec,
        )
        did_compute = _compute_if_requested(db, compute_after)
        _update_last({
            "ok": True,
            "sport": "nfl",
            "day": str(day),
            "leagues": league_list,
            "fixtures_upserted": res.get("fixtures_upserted", 0),
            "odds_rows_written": res.get("odds_rows_written", 0),
            "edges_recomputed": did_compute,
            "message": None,
        })
        return {"ok": True, "ingest": res, "computed": did_compute}
    except Exception as e:
        _update_last({"ok": False, "sport": "nfl", "day": str(day), "leagues": leagues or [], "message": f"{e!r}"})
        raise HTTPException(status_code=502, detail=f"NFL ingest error: {e!r}")

# ---------------------------------------------------------------------
# NBA
# ---------------------------------------------------------------------

@router.post("/nba")
def ingest_nba_route(
    day: date = Form(...),
    leagues: List[str] | None = Form(None),  # e.g. ["NBA"]
    max_games: int = Form(50),
    odds_delay_sec: float = Form(0.30),
    compute_after: int = Form(0),
    db: Session = Depends(get_db),
):
    if not ingest_nba:
        raise HTTPException(status_code=501, detail="NBA ingester not available (fixtures_nba missing)")
    try:
        league_list = leagues or ["NBA"]
        res = ingest_nba(
            db,
            day,
            league_list,
            prefer_book=None,
            max_games=max_games,
            odds_delay_sec=odds_delay_sec,
        )
        did_compute = _compute_if_requested(db, compute_after)
        _update_last({
            "ok": True,
            "sport": "nba",
            "day": str(day),
            "leagues": league_list,
            "fixtures_upserted": res.get("fixtures_upserted", 0),
            "odds_rows_written": res.get("odds_rows_written", 0),
            "edges_recomputed": did_compute,
            "message": None,
        })
        return {"ok": True, "ingest": res, "computed": did_compute}
    except Exception as e:
        _update_last({"ok": False, "sport": "nba", "day": str(day), "leagues": leagues or [], "message": f"{e!r}"})
        raise HTTPException(status_code=502, detail=f"NBA ingest error: {e!r}")

# ---------------------------------------------------------------------
# Status & odds refresh
# ---------------------------------------------------------------------

@router.get("/status")
def ingest_status(db: Session = Depends(get_db)):
    """
    Lightweight health/status for the ingest pipeline + a quick DB snapshot.
    """
    total_fixtures = db.query(Fixture).count()
    total_odds = db.query(Odds).count()

    latest_fixture = (
        db.query(Fixture.kickoff_utc).order_by(Fixture.kickoff_utc.desc()).first()
    )
    latest_odds = (
        db.query(Odds.last_seen).order_by(Odds.last_seen.desc()).first()
    )

    return {
        "ok": True,
        "snapshot": {
            "total_fixtures": total_fixtures,
            "total_odds": total_odds,
            "latest_fixture": latest_fixture[0].isoformat() if latest_fixture else None,
            "latest_odds_seen": latest_odds[0].isoformat() if latest_odds else None,
        },
        "last_run": LAST_INGEST,
        "server_time": datetime.now(timezone.utc).isoformat(),
    }

@router.post("/ingest-odds")
def admin_ingest_odds(
    fixture_id: int = Query(..., description="Local DB Fixture.id"),
    db: Session = Depends(get_db),
):
    """
    Refresh odds for ONE fixture, auto-routing by sport.
    """
    f = db.query(Fixture).filter(Fixture.id == fixture_id).one_or_none()
    if not f:
        raise HTTPException(status_code=404, detail="Fixture not found")

    return _refresh_odds_for_fixture(db, f, prefer_book=None, allow_fallback=True)

@router.post("/refresh-odds-window")
def admin_refresh_odds_window(
    start_day: str = Query(..., description="YYYY-MM-DD"),
    ndays: int = Query(2, ge=1, le=14),
    leagues: str = Query("", description="CSV league keys (EPL,CHAMP,...) or empty=all"),
    sport: Optional[str] = Query(None, description="football|nhl|nfl|nba (optional filter)"),
    prefer_book: str | None = Query(None),
    delay_sec: float = Query(0.25, ge=0.0, le=2.0),
    db: Session = Depends(get_db),
):
    """
    Refresh odds for fixtures in a given date window.
    Works on fixtures already in DB; routes per-sport to the correct provider.
    """
    start = datetime.fromisoformat(start_day).date()
    end   = start + timedelta(days=ndays)

    q = (
        db.query(Fixture)
        .filter(
            Fixture.kickoff_utc >= datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc),
            Fixture.kickoff_utc <  datetime.combine(end,   datetime.min.time(), tzinfo=timezone.utc),
        )
    )

    # Optional sport filter
    if sport:
        q = q.filter(Fixture.sport == _sport_safe(sport))

    # Optional league filter (exact comp keys)
    if leagues:
        wanted = [s.strip() for s in leagues.split(",") if s.strip()]
        if wanted:
            q = q.filter(Fixture.comp.in_(wanted))

    fixtures = q.order_by(Fixture.kickoff_utc.asc()).all()

    results = []
    for f in fixtures:
        try:
            r = _refresh_odds_for_fixture(db, f, prefer_book=prefer_book, allow_fallback=True)
            results.append({"id": f.id, "sport": f.sport, "match": f"{f.home_team} v {f.away_team}", **r})
        except Exception as ex:
            results.append({"id": f.id, "sport": f.sport, "match": f"{f.home_team} v {f.away_team}", "error": repr(ex)})

        if delay_sec:
            time.sleep(delay_sec)

    return {
        "ok": True,
        "start_day": start_day,
        "ndays": ndays,
        "fixtures": len(fixtures),
        "results": results,
    }