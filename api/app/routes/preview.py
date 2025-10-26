# api/app/routes/preview.py
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
from ..models import Fixture, AIPreview, ModelProb, Edge, ExpertPrediction, LeagueStanding
from ..services.form import get_fixture_form_summary  # ✅ use DB helper directly

router = APIRouter(prefix="/ai/preview", tags=["AI Preview"])
pub = APIRouter(prefix="/public/ai/preview", tags=["Public AI Preview"])

# -------------------------------------------------------------------
# Config / Helpers
# -------------------------------------------------------------------

def _api_base() -> str:
    return (
        os.getenv("INTERNAL_API_BASE")
        or os.getenv("PUBLIC_API_BASE")
        or os.getenv("API_BASE")
        or "http://127.0.0.1:8000"
    ).rstrip("/")


def _fetch_form_summaries(fixture_id: int, n: int = 5) -> Dict[str, Any]:
    """
    Legacy HTTP fetch (kept for compatibility). Prefer DB helper below.
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


def _form_from_db(db: Session, fixture_id: int, n: int) -> Dict[str, Any]:
    """
    Preferred: pull summaries straight from DB (no HTTP).
    """
    payload = get_fixture_form_summary(db, fixture_id, n=n) or {}
    return {
        "home": payload.get("home") or {},
        "away": payload.get("away") or {},
    }


def _build_prompt(fixture: Fixture, form_data: Dict[str, Any], n: int) -> str:
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

# ---- DB helper for probs + edges --------------------------------------------

def _get_win_probs_and_edges(
    db: Session,
    fixture_id: int,
    source: str = "team_form",
    top_k_edges: int = 5,
) -> dict:
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
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx:
        raise HTTPException(status_code=404, detail="Fixture not found")

    today = date.today()

    existing = (
        db.query(AIPreview)
        .filter(AIPreview.fixture_id == fixture_id, AIPreview.day == today)
        .first()
    )
    if existing and not overwrite:
        return {"ok": True, "cached": True, "fixture_id": fixture_id, "preview": existing.preview}

    # ✅ use DB helper (no NameError)
    form_data = _form_from_db(db, fixture_id, n=n)
    if not form_data or (not form_data.get("home") and not form_data.get("away")):
        raise HTTPException(status_code=500, detail="Form data unavailable")

    prompt = _build_prompt(fx, form_data, n)

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


@router.get("/debug/form")
def debug_form_data(fixture_id: int, db: Session = Depends(get_db)):
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx:
        return {"error": "Fixture not found"}
    form = _form_from_db(db, fixture_id, n=5)
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
    """
    try:
        day_obj = date.fromisoformat(day)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid day; use YYYY-MM-DD")

    start = datetime.combine(day_obj, datetime.min.time(), tzinfo=timezone.utc)
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
                .filter(AIPreview.fixture_id == fx.id, AIPreview.day == day_obj)
                .first()
            )
            if existing and not overwrite:
                skipped += 1
                continue

            # ✅ DB form summaries (no HTTP, no NameError)
            form_data = _form_from_db(db, fx.id, n=n)
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
                    day=day_obj,                      # ✅ save for requested day
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
    overwrite: bool = Query(False, description="Force regeneration even if cached"),
    db: Session = Depends(get_db),
):
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx:
        raise HTTPException(status_code=404, detail="Fixture not found")

    today = date.today()

    existing = (
        db.query(ExpertPrediction)
        .filter(ExpertPrediction.fixture_id == fixture_id, ExpertPrediction.day == today)
        .first()
    )
    if existing and not overwrite:
        return {
            "fixture_id": fixture_id,
            "home": fx.home_team,
            "away": fx.away_team,
            "analysis": existing.payload,
            "cached": True,
        }

    # --- Build model + context data ---
    # Use DB summaries to avoid HTTP dependency
    form_data = _form_from_db(db, fixture_id, n=n)

    model = _get_win_probs_and_edges(db, fixture_id, source="team_form", top_k_edges=5)
    p_home = model["probs"]["home"]
    p_draw = model["probs"]["draw"]
    p_away = model["probs"]["away"]
    top_edges = model["edges"]

    league_positions = (
        db.query(LeagueStanding.team, LeagueStanding.position)
        .filter(LeagueStanding.league == fx.comp)
        .all()
    )
    home_pos = next((p for t, p in league_positions if t == fx.home_team), None)
    away_pos = next((p for t, p in league_positions if t == fx.away_team), None)

    home_summary = form_data.get("home", {})
    away_summary = form_data.get("away", {})

    prompt = f"""
You are a professional sports betting analyst.
Write a succinct bettor-facing note for {fx.home_team} vs {fx.away_team} ({fx.comp or fx.sport}).

League positions: 
- {fx.home_team}: {home_pos or "unknown"} 
- {fx.away_team}: {away_pos or "unknown"}

Context (last {n}):
- {fx.home_team}: {home_summary}
- {fx.away_team}: {away_summary}

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
        import json
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        data = {
            "paragraphs": ["Analysis unavailable."],
            "probabilities": {"home": None, "draw": None, "away": None},
            "best_bets": [],
            "confidence": "Low",
            "disclaimer": "No bet if price/edge isn’t there."
        }
        print(f"[expert_prediction] OpenAI fallback: {e}")

    try:
        if existing:
            existing.payload = data
            existing.home_prob = p_home
            existing.draw_prob = p_draw
            existing.away_prob = p_away
            existing.confidence = data.get("confidence")
            existing.updated_at = datetime.utcnow()
        else:
            row = ExpertPrediction(
                fixture_id=fixture_id,
                day=today,
                payload=data,
                home_prob=p_home,
                draw_prob=p_draw,
                away_prob=p_away,
                confidence=data.get("confidence"),
            )
            db.add(row)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[expert_prediction] DB write error: {e}")

    return {
        "fixture_id": fixture_id,
        "home": fx.home_team,
        "away": fx.away_team,
        "analysis": data,
        "cached": False,
    }


@router.get("/winprobs")
def admin_win_probs(fixture_id: int, source: str = Query("team_form"), db: Session = Depends(get_db)):
    data = _get_win_probs_and_edges(db, fixture_id, source=source, top_k_edges=0)
    return data["probs"]

@router.get("/edges")
def admin_edges(fixture_id: int, source: str = Query("team_form"), db: Session = Depends(get_db)):
    data = _get_win_probs_and_edges(db, fixture_id, source=source, top_k_edges=50)
    return data["edges"]