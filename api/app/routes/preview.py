# api/app/routes/previews.py
from __future__ import annotations

import os
from datetime import datetime, timezone, date, timedelta
from typing import Any, Dict

import requests
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from openai import OpenAI

from ..db import get_db
from ..models import Fixture, AIPreview, ModelProb, Edge

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

# ---- NEW: DB helper for probs + edges (no HTTP mismatch issues) --------------

def _get_win_probs_and_edges(
    db: Session,
    fixture_id: int,
    source: str = "team_form",
    top_k_edges: int = 5,
) -> dict:
    """
    Return latest HOME/DRAW/AWAY probabilities for a fixture + top positive edges.
    """
    # latest rows for 1X2 markets
    latest = (
        db.query(
            ModelProb.market,
            func.max(ModelProb.as_of).label("latest"),
        )
        .filter(
            ModelProb.fixture_id == fixture_id,
            ModelProb.source == source,
            ModelProb.market.in_(["HOME_WIN", "DRAW", "AWAY_WIN"]),
        )
        .group_by(ModelProb.market)
        .subquery()
    )

    probs_rows = (
        db.query(ModelProb)
        .join(latest, and_(
            ModelProb.market == latest.c.market,
            ModelProb.as_of == latest.c.latest
        ))
        .all()
    )

    p_home = p_draw = p_away = None
    for r in probs_rows:
        if r.market == "HOME_WIN":
            p_home = float(r.prob)
        elif r.market == "DRAW":
            p_draw = float(r.prob)
        elif r.market == "AWAY_WIN":
            p_away = float(r.prob)

    edges_rows = (
        db.query(Edge)
        .filter(Edge.fixture_id == fixture_id, Edge.model_source == source)
        .order_by(Edge.edge.desc(), Edge.created_at.desc())
        .all()
    )

    pos_edges = [
        {
            "market": e.market,
            "bookmaker": e.bookmaker,
            "price": float(e.price) if e.price is not None else None,
            "prob": float(e.prob) if e.prob is not None else None,
            "edge": float(e.edge) if e.edge is not None else None,
        }
        for e in edges_rows
        if (e.edge or 0) > 0
    ][:top_k_edges]

    return {"probs": {"home": p_home, "draw": p_draw, "away": p_away}, "edges": pos_edges}

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

# --- Expert bettor analysis (JSON for Predictions tab) -----------------------

@router.get("/expert")
def expert_prediction(
    fixture_id: int = Query(...),
    n: int = Query(5, ge=3, le=10),
    db: Session = Depends(get_db),
):
    """Return JSON: paragraphs + W/D/W probabilities + best bets (from model edges)."""
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx:
        raise HTTPException(status_code=404, detail="Fixture not found")

    form_data = _fetch_form_summaries(fixture_id, n=n)
    model = _get_win_probs_and_edges(db, fixture_id, source="team_form", top_k_edges=5)

    p_home = model["probs"]["home"]
    p_draw = model["probs"]["draw"]
    p_away = model["probs"]["away"]
    top_edges = model["edges"]

    prompt = f"""
You are a professional sports betting analyst.
Write a succinct bettor-facing note for {fx.home_team} vs {fx.away_team} ({fx.comp or fx.sport}).

Context (last {n}):
- {fx.home_team}: {form_data.get('home',{})}
- {fx.away_team}: {form_data.get('away',{})}

Model win probabilities (0..1, any missing = unknown):
- home: {p_home}
- draw: {p_draw}
- away: {p_away}

Edges (candidate bets from model, include only if they look positive):
{top_edges}

Produce a SINGLE JSON object with:
- "paragraphs": array of 2–3 short paragraphs in plain text (no Markdown),
- "probabilities": object {{ "home": number|null, "draw": number|null, "away": number|null }} in PERCENT (0–100, 1 decimal),
- "best_bets": array of up to 3 objects {{
    "market": string,
    "bookmaker": string|null,
    "price": number|null,
    "edge_pct": number|null,
    "why": string
  }},
- "confidence": one of ["Low","Medium","High"] based on edge quality + price availability,
- "disclaimer": short string reminding about variance and bankroll discipline.

Keep it grounded in the numbers above. Do not invent players/injuries. If no edges are positive, say so and keep best_bets empty.
""".strip()

    client = _openai_client()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=450,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI call failed: {e}")

    # Safety net: parse
    try:
        import json
        data = json.loads(raw)
    except Exception:
        data = {
            "paragraphs": ["Analysis unavailable."],
            "probabilities": {"home": None, "draw": None, "away": None},
            "best_bets": [],
            "confidence": "Low",
            "disclaimer": "No bet if price/edge isn’t there."
        }

    return {
        "fixture_id": fixture_id,
        "home": fx.home_team,
        "away": fx.away_team,
        "analysis": data,
    }

# (Optional) tiny debug helpers for probing via HTTP
@router.get("/winprobs")
def admin_win_probs(fixture_id: int, source: str = Query("team_form"), db: Session = Depends(get_db)):
    data = _get_win_probs_and_edges(db, fixture_id, source=source, top_k_edges=0)
    return data["probs"]

@router.get("/edges")
def admin_edges(fixture_id: int, source: str = Query("team_form"), db: Session = Depends(get_db)):
    data = _get_win_probs_and_edges(db, fixture_id, source=source, top_k_edges=50)
    return data["edges"]