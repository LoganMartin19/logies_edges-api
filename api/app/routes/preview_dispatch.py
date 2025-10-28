from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import date
from ..db import get_db
from ..models import Fixture, AIPreview

# Import the 3 sport modules (football = your upgraded file)
from .preview import generate_ai_preview as gen_football, generate_daily_ai_previews as gen_football_daily
from .preview_gridiron import generate_ai_preview as gen_gridiron, generate_daily_ai_previews as gen_gridiron_daily
from .preview_hockey import generate_ai_preview as gen_hockey, generate_daily_ai_previews as gen_hockey_daily

router = APIRouter(prefix="/ai/preview", tags=["AI Preview (dispatch)"])
pub = APIRouter(prefix="/public/ai/preview", tags=["Public AI Preview (dispatch)"])

SPORT_MAP = {
    "football": {"gen": gen_football, "daily": gen_football_daily},
    "soccer":   {"gen": gen_football, "daily": gen_football_daily},
    "nfl":      {"gen": gen_gridiron, "daily": gen_gridiron_daily},
    "gridiron": {"gen": gen_gridiron, "daily": gen_gridiron_daily},
    "hockey":   {"gen": gen_hockey, "daily": gen_hockey_daily},
    "nhl":      {"gen": gen_hockey, "daily": gen_hockey_daily},
}

def _resolve_handlers(sport: str):
    s = (sport or "").lower().strip()
    return SPORT_MAP.get(s, SPORT_MAP["football"])

@router.post("/generate/auto")
def generate_auto(fixture_id: int = Query(...), overwrite: bool = Query(False), db: Session = Depends(get_db)):
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx:
        raise HTTPException(status_code=404, detail="Fixture not found")
    handlers = _resolve_handlers(fx.sport or "football")
    # call the sport-specific generator
    return handlers["gen"](fixture_id=fixture_id, overwrite=overwrite, db=db)

@router.post("/generate/daily/auto")
def generate_daily_auto(day: str = Query(date.today().isoformat()), overwrite: bool = Query(False), db: Session = Depends(get_db)):
    # fire daily per-sport for all sports you support
    out = {}
    for key, h in SPORT_MAP.items():
        try:
            res = h["daily"](day=day, overwrite=overwrite, sport=key, db=db)  # sport param is accepted by our modules
            out[key] = {"ok": True, "added": res.get("added", 0), "skipped": res.get("skipped", 0)}
        except Exception as e:
            out[key] = {"ok": False, "error": str(e)}
    return out

@pub.get("/by-fixture")
def public_preview_by_fixture(fixture_id: int = Query(...), day: str | None = Query(None), db: Session = Depends(get_db)):
    from datetime import date as _date
    d = _date.today() if not day else _date.fromisoformat(day)
    row = (
        db.query(AIPreview)
        .filter(AIPreview.fixture_id == fixture_id, AIPreview.day == d)
        .order_by(AIPreview.updated_at.desc())
        .first()
    )
    if not row:
        # No preview cached today? Generate on-demand once.
        fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
        if not fx:
            raise HTTPException(status_code=404, detail="Fixture not found")
        handlers = _resolve_handlers(fx.sport or "football")
        try:
            handlers["gen"](fixture_id=fixture_id, overwrite=False, db=db)
        except Exception:
            pass
        row = (
            db.query(AIPreview)
            .filter(AIPreview.fixture_id == fixture_id, AIPreview.day == d)
            .order_by(AIPreview.updated_at.desc())
            .first()
        )
        if not row:
            raise HTTPException(status_code=404, detail="Preview not found")
    return {
        "fixture_id": row.fixture_id,
        "day": row.day.isoformat(),
        "sport": (db.query(Fixture).get(row.fixture_id).sport if row.fixture_id else None),
        "preview": row.preview,
        "model": row.model,
        "updated_at": row.updated_at.isoformat(),
    }