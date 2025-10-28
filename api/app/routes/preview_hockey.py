from __future__ import annotations
import os
from datetime import datetime, timezone, date, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from openai import OpenAI
from ..db import get_db
from ..models import Fixture, AIPreview, ModelProb

router = APIRouter(prefix="/ai/preview/hockey", tags=["AI Preview (NHL)"])

def _openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    return OpenAI(api_key=key)

def _get_probs(db: Session, fixture_id: int, source: str = "team_form") -> dict:
    latest = (
        db.query(ModelProb.market, func.max(ModelProb.as_of).label("latest"))
        .filter(ModelProb.fixture_id == fixture_id, ModelProb.source == source)
        .group_by(ModelProb.market).subquery()
    )
    rows = (
        db.query(ModelProb).join(latest, and_(ModelProb.market == latest.c.market, ModelProb.as_of == latest.c.latest)).all()
    )
    out = {"home": None, "away": None}
    for r in rows:
        if r.market in ("HOME_WIN","MONEYLINE_HOME"): out["home"] = float(r.prob)
        if r.market in ("AWAY_WIN","MONEYLINE_AWAY"): out["away"] = float(r.prob)
    return out

def _build_prompt(fx: Fixture, probs: dict) -> str:
    home, away = fx.home_team, fx.away_team
    p_home = f"{probs['home']*100:.1f}%" if probs["home"] is not None else "n/a"
    p_away = f"{probs['away']*100:.1f}%" if probs["away"] is not None else "n/a"
    return f"""
Write a concise NHL game preview (5–7 lines) for {home} vs {away}.
Neutral tone. Use ONLY model probabilities; no invented player news.
Note tempo/variance of hockey briefly and finish with a single deciding-factor line.

Model probabilities (moneyline): {home} {p_home}, {away} {p_away}.
If probabilities are missing, keep it generic without inventing stats.
""".strip()

@router.post("/generate")
def generate_ai_preview(fixture_id: int = Query(...), overwrite: bool = Query(False), db: Session = Depends(get_db)):
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx: raise HTTPException(status_code=404, detail="Fixture not found")
    today = date.today()
    existing = db.query(AIPreview).filter(AIPreview.fixture_id == fixture_id, AIPreview.day == today).first()
    if existing and not overwrite:
        return {"ok": True, "cached": True, "fixture_id": fixture_id, "preview": existing.preview}

    probs = _get_probs(db, fixture_id)
    prompt = _build_prompt(fx, probs)
    client = _openai_client()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":"You are a neutral NHL analyst."},{"role":"user","content":prompt}],
        temperature=0.6, max_tokens=220,
    )
    text = (resp.choices[0].message.content or "").strip()
    tokens = getattr(getattr(resp, "usage", None), "total_tokens", None)

    if existing:
        existing.preview, existing.tokens, existing.updated_at = text, tokens, datetime.utcnow()
        db.add(existing)
    else:
        db.add(AIPreview(fixture_id=fixture_id, day=today, sport="nhl", comp=fx.comp, preview=text, model="gpt-4o-mini", tokens=tokens))
    db.commit()
    return {"ok": True, "fixture_id": fixture_id, "preview": text, "tokens": tokens}

@router.post("/generate/daily")
def generate_daily_ai_previews(day: str = Query(date.today().isoformat()), sport: str = Query("nhl"), overwrite: bool = Query(False), db: Session = Depends(get_db)):
    start = datetime.fromisoformat(day+"T00:00:00+00:00")
    end = start + timedelta(days=1)
    fixtures = db.query(Fixture).filter(Fixture.sport.in_(["nhl","hockey"]), Fixture.kickoff_utc >= start, Fixture.kickoff_utc < end).all()
    added, skipped, errors = 0, 0, []
    for fx in fixtures:
        try:
            generate_ai_preview(fixture_id=fx.id, overwrite=overwrite, db=db)
            added += 1
        except Exception as e:
            db.rollback(); errors.append({"fixture": fx.id, "error": str(e)})
    return {"ok": True, "added": added, "skipped": skipped, "errors": errors}