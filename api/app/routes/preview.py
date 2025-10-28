# api/app/routes/preview.py
from __future__ import annotations

import os
import json
from datetime import datetime, timezone, date, timedelta
from typing import Any, Dict

import requests
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from openai import OpenAI

from ..db import get_db
from ..models import Fixture, AIPreview, ModelProb, Edge, ExpertPrediction, LeagueStanding
from ..services.form import get_fixture_form_summary
from ..services.apifootball import get_fixture as _af_get_fixture, get_team_recent_results as _af_recent

router = APIRouter(prefix="/ai/preview", tags=["AI Preview"])
pub = APIRouter(prefix="/public/ai/preview", tags=["Public AI Preview"])

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _api_base() -> str:
    return (
        os.getenv("INTERNAL_API_BASE")
        or os.getenv("PUBLIC_API_BASE")
        or os.getenv("API_BASE")
        or "http://127.0.0.1:8000"
    ).rstrip("/")


def _openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in environment")
    return OpenAI(api_key=key)


def _form_from_db(db: Session, fixture_id: int, n: int) -> Dict[str, Any]:
    payload = get_fixture_form_summary(db, fixture_id, n=n) or {}
    return {"home": payload.get("home") or {}, "away": payload.get("away") or {}}


def _get_team_stats(fixture_id: int) -> Dict[str, Any]:
    """
    Calls internal /football/team-stats route for richer team data (WDL, goals, avg GF/GA, form string).
    """
    base = _api_base()
    try:
        r = requests.get(f"{base}/football/team-stats", params={"fixture_id": fixture_id}, timeout=12)
        if not r.ok:
            print(f"[preview] team-stats failed {r.status_code}: {r.text[:120]}")
            return {}
        return r.json()
    except Exception as e:
        print("[preview] team-stats error:", e)
        return {}


def _fixture_ctx_for_shared(fixture: Fixture) -> dict:
    try:
        if not fixture.provider_fixture_id:
            return {"league_id": 0, "season": 0, "home_pid": 0, "away_pid": 0}
        fx_json = _af_get_fixture(int(fixture.provider_fixture_id)) or {}
        fr = (fx_json.get("response") or [None])[0] or {}
        lg = fr.get("league") or {}
        t = fr.get("teams") or {}
        return {
            "league_id": int(lg.get("id") or 0),
            "season": int(lg.get("season") or 0),
            "home_pid": int((t.get("home") or {}).get("id") or 0),
            "away_pid": int((t.get("away") or {}).get("id") or 0),
        }
    except Exception:
        return {"league_id": 0, "season": 0, "home_pid": 0, "away_pid": 0}


def _shared_opponents_text(home_pid: int, away_pid: int, season: int, league_id: int, lookback: int = 8) -> str:
    if not (home_pid and away_pid and season and league_id):
        return ""
    home_recent = _af_recent(home_pid, season=season, limit=lookback, league_id=league_id) or []
    away_recent = _af_recent(away_pid, season=season, limit=lookback, league_id=league_id) or []

    H = {r.get("opponent"): r for r in home_recent if r.get("opponent")}
    A = {r.get("opponent"): r for r in away_recent if r.get("opponent")}
    shared = [o for o in H.keys() if o in A]
    if not shared:
        return ""

    lines = []
    for opp in shared[:5]:
        h, a = H[opp], A[opp]
        h_score = h.get("score") or f"{h.get('goals_for', '?')}-{h.get('goals_against', '?')}"
        a_score = a.get("score") or f"{a.get('goals_for', '?')}-{a.get('goals_against', '?')}"
        lines.append(f"- vs {opp}: {h.get('result', '?')} {h_score} | {a.get('result', '?')} {a_score}")
    return "Shared Opponents (recent same-league):\n" + "\n".join(lines)


def _get_win_probs_and_edges(db: Session, fixture_id: int, source: str = "team_form") -> dict:
    latest = (
        db.query(
            ModelProb.market,
            func.max(ModelProb.as_of).label("latest"),
        )
        .filter(ModelProb.fixture_id == fixture_id, ModelProb.source == source)
        .group_by(ModelProb.market)
        .subquery()
    )
    rows = (
        db.query(ModelProb)
        .join(latest, and_(ModelProb.market == latest.c.market, ModelProb.as_of == latest.c.latest))
        .all()
    )
    probs = {"home": None, "draw": None, "away": None}
    for r in rows:
        if r.market == "HOME_WIN": probs["home"] = float(r.prob)
        elif r.market == "DRAW": probs["draw"] = float(r.prob)
        elif r.market == "AWAY_WIN": probs["away"] = float(r.prob)
    return probs


def _build_prompt(fixture: Fixture, form_data, team_stats, shared_text, probs, n: int) -> str:
    home, away = fixture.home_team or "Home", fixture.away_team or "Away"
    comp = fixture.comp or fixture.sport or "match"

    # form
    hf, af = form_data.get("home", {}), form_data.get("away", {})
    h_w, h_d, h_l = hf.get("wins", 0), hf.get("draws", 0), hf.get("losses", 0)
    a_w, a_d, a_l = af.get("wins", 0), af.get("draws", 0), af.get("losses", 0)
    h_gf, h_ga = hf.get("avg_goals_for", 0), hf.get("avg_goals_against", 0)
    a_gf, a_ga = af.get("avg_goals_for", 0), af.get("avg_goals_against", 0)

    # team stats summary
    summary = team_stats.get("summary") or {}
    home_sum = summary.get("home") or {}
    away_sum = summary.get("away") or {}
    h_form, a_form = home_sum.get("form") or "?", away_sum.get("form") or "?"
    h_avg_gf, h_avg_ga = home_sum.get("avg_gf", 0), home_sum.get("avg_ga", 0)
    a_avg_gf, a_avg_ga = away_sum.get("avg_gf", 0), away_sum.get("avg_ga", 0)

    # model probs
    p_home, p_draw, p_away = probs.get("home"), probs.get("draw"), probs.get("away")
    prob_line = ""
    if any(v is not None for v in [p_home, p_draw, p_away]):
        def fmt(x): return f"{x*100:.1f}%" if x is not None else "n/a"
        prob_line = f"\nModel probabilities: {home} {fmt(p_home)}, Draw {fmt(p_draw)}, {away} {fmt(p_away)}."

    shared_block = f"\n\n{shared_text}" if shared_text else ""

    return f"""
Write a concise {comp} preview (6–8 lines) for {home} vs {away}.
Use ONLY the factual data below — do not invent lineups or player news.
Summarize recent form, season averages, and shared opponents where relevant.
Conclude with one balanced 'what decides it' line.

Recent Form (last {n}):
- {home}: {h_w}W–{h_d}D–{h_l}L, {h_gf:.1f} scored / {h_ga:.1f} conceded.
- {away}: {a_w}W–{a_d}D–{a_l}L, {a_gf:.1f} scored / {a_ga:.1f} conceded.

Season Stats:
- {home}: avg {h_avg_gf:.1f} GF / {h_avg_ga:.1f} GA, Form {h_form}
- {away}: avg {a_avg_gf:.1f} GF / {a_avg_ga:.1f} GA, Form {a_form}
{prob_line}{shared_block}
""".strip()

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

    form_data = _form_from_db(db, fixture_id, n)
    team_stats = _get_team_stats(fixture_id)
    probs = _get_win_probs_and_edges(db, fixture_id)
    ctx = _fixture_ctx_for_shared(fx)
    shared_text = _shared_opponents_text(ctx["home_pid"], ctx["away_pid"], ctx["season"], ctx["league_id"], n)

    prompt = _build_prompt(fx, form_data, team_stats, shared_text, probs, n)
    client = _openai_client()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a neutral, sharp football analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=280,
        )
        text = (resp.choices[0].message.content or "").strip()
        tokens = getattr(getattr(resp, "usage", None), "total_tokens", None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI call failed: {e}")

    if existing:
        existing.preview, existing.tokens = text, tokens
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


@router.post("/generate/daily")
def generate_daily_ai_previews(
    day: str = Query(date.today().isoformat(), description="YYYY-MM-DD"),
    sport: str = Query("football"),
    n: int = Query(5, ge=3, le=10),
    overwrite: bool = Query(False),
    db: Session = Depends(get_db),
):
    try:
        day_obj = date.fromisoformat(day)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")
    start = datetime.combine(day_obj, datetime.min.time(), tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    fixtures = (
        db.query(Fixture)
        .filter(Fixture.sport == sport)
        .filter(Fixture.kickoff_utc >= start, Fixture.kickoff_utc < end)
        .order_by(Fixture.kickoff_utc.asc())
        .all()
    )

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

            form_data = _form_from_db(db, fx.id, n)
            team_stats = _get_team_stats(fx.id)
            probs = _get_win_probs_and_edges(db, fx.id)
            ctx = _fixture_ctx_for_shared(fx)
            shared_text = _shared_opponents_text(ctx["home_pid"], ctx["away_pid"], ctx["season"], ctx["league_id"], n)

            prompt = _build_prompt(fx, form_data, team_stats, shared_text, probs, n)
            client = _openai_client()

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a sharp, neutral football analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.6,
                max_tokens=280,
            )
            text = (resp.choices[0].message.content or "").strip()
            tokens = getattr(getattr(resp, "usage", None), "total_tokens", None)

            if existing:
                existing.preview, existing.tokens = text, tokens
                existing.updated_at = datetime.utcnow()
                db.add(existing)
            else:
                db.add(AIPreview(
                    fixture_id=fx.id,
                    day=day_obj,
                    sport=sport,
                    comp=fx.comp,
                    preview=text,
                    model="gpt-4o-mini",
                    tokens=tokens,
                ))
            db.commit()
            added += 1
        except Exception as e:
            db.rollback()
            errors.append({"fixture": fx.id, "error": str(e)})

    return {"ok": True, "added": added, "skipped": skipped, "errors": errors}

# -------------------------------------------------------------------
# Public route (unchanged)
# -------------------------------------------------------------------

@pub.get("/by-fixture")
def public_preview_by_fixture(
    fixture_id: int = Query(...),
    day: str | None = Query(None),
    db: Session = Depends(get_db),
):
    d = date.today() if not day else date.fromisoformat(day)
    row = (
        db.query(AIPreview)
        .filter(AIPreview.fixture_id == fixture_id, AIPreview.day == d)
        .order_by(AIPreview.updated_at.desc())
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Preview not found")
    return {
        "fixture_id": fixture_id,
        "day": row.day.isoformat(),
        "preview": row.preview,
        "model": row.model,
        "updated_at": row.updated_at.isoformat(),
    }