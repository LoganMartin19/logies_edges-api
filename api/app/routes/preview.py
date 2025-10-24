from __future__ import annotations

import os
from datetime import datetime, timezone, date, timedelta
from typing import Any, Dict

import requests
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from openai import OpenAI

from ..db import get_db
from ..models import Fixture, AIPreview

router = APIRouter(prefix="/ai/preview", tags=["AI Preview"])
pub = APIRouter(prefix="/public/ai/preview", tags=["Public AI Preview"])

# -------------------------------------------------------------------
# Config / Helpers
# -------------------------------------------------------------------

def _api_base() -> str:
    """
    Resolve a base URL that works both locally and on Render.
    Prefer INTERNAL_API_BASE (set this in Render), then PUBLIC_API_BASE/API_BASE,
    and finally fall back to localhost for dev.
    """
    return (
        os.getenv("INTERNAL_API_BASE")
        or os.getenv("PUBLIC_API_BASE")
        or os.getenv("API_BASE")
        or "http://127.0.0.1:8000"
    ).rstrip("/")


def _fetch_form_summaries(fixture_id: int, n: int = 5) -> Dict[str, Any]:
    """
    Fetch hybrid form JSON from:
      GET {BASE}/form/fixture/json?fixture_id=...&n=...
    Expected keys: home_form, away_form
    Returns dict with normalized 'home' and 'away' summaries (or {} on error).
    """
    base = _api_base()
    url = f"{base}/form/fixture/json"
    params = {"fixture_id": fixture_id, "n": n}
    try:
        r = requests.get(url, params=params, timeout=12)
        if not r.ok:
            print(f"[preview] Form fetch failed {r.status_code} ({url}): {r.text[:200]}")
            return {}
        j = r.json() or {}
        # Normalize to { home: {...}, away: {...} }
        return {
            "home": j.get("home_form") or {},
            "away": j.get("away_form") or {},
        }
    except Exception as e:
        print("[preview] Form fetch error:", e)
        return {}


def _build_prompt(fixture: Fixture, form_data: Dict[str, Any], n: int) -> str:
    """
    Build a concise analyst-style prompt. Explicitly mentions last N matches.
    """
    home = fixture.home_team or "Home"
    away = fixture.away_team or "Away"
    comp = fixture.comp or fixture.sport or "match"

    home_form = form_data.get("home") or {}
    away_form = form_data.get("away") or {}

    h_w, h_d, h_l = int(home_form.get("wins", 0)), int(home_form.get("draws", 0)), int(home_form.get("losses", 0))
    a_w, a_d, a_l = int(away_form.get("wins", 0)), int(away_form.get("draws", 0)), int(away_form.get("losses", 0))

    h_gf = float(home_form.get("avg_goals_for", 0.0))
    h_ga = float(home_form.get("avg_goals_against", 0.0))
    a_gf = float(away_form.get("avg_goals_for", 0.0))
    a_ga = float(away_form.get("avg_goals_against", 0.0))

    return f"""
Write a concise {comp} match preview (4–6 lines) for {home} vs {away}.
Use ONLY the stats below; do not make up players or injuries.
Explicitly state that the form is from the last {n} matches for each team.
Keep it neutral, factual, and in natural football English. Finish with one balanced 'what decides it' line.

Stats (last {n} matches each):
- {home}: {h_w}W–{h_d}D–{h_l}L, avg {h_gf:.1f} scored / {h_ga:.1f} conceded per game.
- {away}: {a_w}W–{a_d}D–{a_l}L, avg {a_gf:.1f} scored / {a_ga:.1f} conceded per game.
""".strip()


def _openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in environment")
    return OpenAI(api_key=key)

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@router.post("/generate")
def generate_ai_preview(
    fixture_id: int = Query(...),
    n: int = Query(5, ge=3, le=10),
    overwrite: bool = Query(False),
    db: Session = Depends(get_db),
):
    """
    Generate (or return cached) AI preview using Chat Completions (gpt-4o-mini).
    """
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx:
        raise HTTPException(status_code=404, detail="Fixture not found")

    today = date.today()

    # Cache check for today's preview
    existing = (
        db.query(AIPreview)
        .filter(AIPreview.fixture_id == fixture_id, AIPreview.day == today)
        .first()
    )
    if existing and not overwrite:
        return {"ok": True, "cached": True, "fixture_id": fixture_id, "preview": existing.preview}

    # Fetch form data (prod-safe)
    form_data = _fetch_form_summaries(fixture_id, n=n)
    if not form_data or (not form_data.get("home") and not form_data.get("away")):
        raise HTTPException(status_code=500, detail="Form data unavailable (check INTERNAL_API_BASE/env)")

    prompt = _build_prompt(fx, form_data, n)

    # --- OpenAI call (Chat Completions) ---
    client = _openai_client()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a sharp, neutral football analyst who writes concise previews."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=220,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise ValueError("Empty content from chat.completions")
        tokens = getattr(getattr(resp, "usage", None), "total_tokens", None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI call failed: {e}")

    # Upsert cache
    if existing:
        existing.preview = text
        existing.tokens = tokens
        existing.updated_at = datetime.utcnow()
        db.add(existing)
    else:
        ai = AIPreview(
            fixture_id=fixture_id,
            day=today,
            sport=fx.sport or "football",
            comp=fx.comp,
            preview=text,
            model="gpt-4o-mini",
            tokens=tokens,
        )
        db.add(ai)
    db.commit()

    return {"ok": True, "fixture_id": fixture_id, "preview": text, "tokens": tokens}


@router.get("/get")
def get_ai_preview(fixture_id: int, db: Session = Depends(get_db)):
    today = date.today()
    row = (
        db.query(AIPreview)
        .filter(AIPreview.fixture_id == fixture_id, AIPreview.day == today)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Preview not found")
    return {"fixture_id": fixture_id, "preview": row.preview, "model": row.model}


# --- Debug: confirm form payload we feed the LLM -----------------------------

@router.get("/debug/form")
def debug_form_data(fixture_id: int, db: Session = Depends(get_db)):
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx:
        return {"error": "Fixture not found"}
    form = _fetch_form_summaries(fixture_id, n=5)
    return {
        "fixture_id": fixture_id,
        "home_team": fx.home_team,
        "away_team": fx.away_team,
        "home_summary": form.get("home"),
        "away_summary": form.get("away"),
    }


@router.post("/generate/daily")
def generate_daily_ai_previews(
    day: str = Query(date.today().isoformat(), description="YYYY-MM-DD"),
    sport: str = Query("football"),
    n: int = Query(5, ge=3, le=10),
    overwrite: bool = Query(False),
    db: Session = Depends(get_db),
):
    """
    Generate AI previews for all fixtures on a given day (default = today).
    Uses the same logic as single fixture generation.
    """
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    fixtures = (
        db.query(Fixture)
        .filter(Fixture.sport == sport)
        .filter(Fixture.kickoff_utc >= start, Fixture.kickoff_utc < end)
        .order_by(Fixture.kickoff_utc.asc())
        .all()
    )

    if not fixtures:
        return {"ok": True, "count": 0, "message": f"No {sport} fixtures for {day}"}

    added, skipped, errors = 0, 0, []

    for fx in fixtures:
        try:
            # Check cache for today
            existing = (
                db.query(AIPreview)
                .filter(AIPreview.fixture_id == fx.id, AIPreview.day == date.today())
                .first()
            )
            if existing and not overwrite:
                skipped += 1
                continue

            form_data = _fetch_form_summaries(fx.id, n=n)
            if not form_data or (not form_data.get("home") and not form_data.get("away")):
                skipped += 1
                continue

            prompt = _build_prompt(fx, form_data, n)
            client = _openai_client()

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a sharp, neutral football analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.6,
                max_tokens=220,
            )

            text = (resp.choices[0].message.content or "").strip()
            tokens = getattr(getattr(resp, "usage", None), "total_tokens", None)

            if existing:
                existing.preview = text
                existing.tokens = tokens
                existing.updated_at = datetime.utcnow()
                db.add(existing)
            else:
                ai = AIPreview(
                    fixture_id=fx.id,
                    day=date.today(),
                    sport=fx.sport or "football",
                    comp=fx.comp,
                    preview=text,
                    model="gpt-4o-mini",
                    tokens=tokens,
                )
                db.add(ai)
            db.commit()
            added += 1
        except Exception as e:
            db.rollback()
            errors.append({"fixture": fx.id, "error": str(e)})

    return {"ok": True, "added": added, "skipped": skipped, "errors": errors}

# --- PUBLIC: read preview for a fixture --------------------------------------

@pub.get("/by-fixture")
def public_preview_by_fixture(
    fixture_id: int = Query(...),
    day: str | None = Query(None, description="Optional YYYY-MM-DD; defaults to today"),
    db: Session = Depends(get_db),
):
    """
    Return the AI preview text for a given fixture.
    - If ?day=YYYY-MM-DD is provided, return that day's preview (404 if none).
    - Otherwise, return today's preview if present, else the most recent one.
    """
    from datetime import date as _date

    def _row_to_payload(row: AIPreview) -> dict:
        return {
            "fixture_id": row.fixture_id,
            "day": row.day.isoformat(),
            "preview": row.preview,
            "model": row.model,
            "updated_at": row.updated_at.isoformat(),
        }

    # specific day?
    if day:
        try:
            d = _date.fromisoformat(day)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid day; use YYYY-MM-DD")
        row = (
            db.query(AIPreview)
            .filter(AIPreview.fixture_id == fixture_id, AIPreview.day == d)
            .order_by(AIPreview.updated_at.desc())
            .first()
        )
        if not row:
            raise HTTPException(status_code=404, detail="Preview not found for that day")
        return _row_to_payload(row)

    # default: today then latest
    today = _date.today()
    today_row = (
        db.query(AIPreview)
        .filter(AIPreview.fixture_id == fixture_id, AIPreview.day == today)
        .order_by(AIPreview.updated_at.desc())
        .first()
    )
    if today_row:
        return _row_to_payload(today_row)

    latest = (
        db.query(AIPreview)
        .filter(AIPreview.fixture_id == fixture_id)
        .order_by(AIPreview.day.desc(), AIPreview.updated_at.desc())
        .first()
    )
    if not latest:
        raise HTTPException(status_code=404, detail="No preview found")
    return _row_to_payload(latest)