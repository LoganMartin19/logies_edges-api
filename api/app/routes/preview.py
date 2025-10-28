# api/app/routes/preview.py
from __future__ import annotations

import os
import json
from datetime import datetime, timezone, date, timedelta
from typing import Any, Dict, List

import requests
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from openai import OpenAI

from ..db import get_db
from ..models import (
    Fixture,
    AIPreview,
    ModelProb,
    Edge,
    ExpertPrediction,
    LeagueStanding,
)
from ..services.form import get_fixture_form_summary  # DB helper for form
# for shared-opponents context (IDs + recent same-league)
from ..services.apifootball import (
    get_fixture as _af_get_fixture,
    get_team_recent_results as _af_recent,
)

router = APIRouter(prefix="/ai/preview", tags=["AI Preview"])
pub = APIRouter(prefix="/public/ai/preview", tags=["Public AI Preview"])

# -------------------------------------------------------------------
# Config / Base helpers
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

# -------------------------------------------------------------------
# Data helpers (form, team-stats, probs/edges, shared opponents)
# -------------------------------------------------------------------

def _form_from_db(db: Session, fixture_id: int, n: int) -> Dict[str, Any]:
    payload = get_fixture_form_summary(db, fixture_id, n=n) or {}
    return {"home": payload.get("home") or {}, "away": payload.get("away") or {}}


def _get_team_stats(fixture_id: int) -> Dict[str, Any]:
    """
    Hit our internal richer team-stats endpoint. Safe no-op if missing.
    """
    base = _api_base()
    try:
        r = requests.get(f"{base}/football/team-stats", params={"fixture_id": fixture_id}, timeout=12)
        if not r.ok:
            print(f"[preview] team-stats failed {r.status_code}: {r.text[:150]}")
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


def _shared_opponents_text(
    home_pid: int, away_pid: int, season: int, league_id: int, lookback: int = 8
) -> str:
    """
    Build 'shared opponents' lines using opponent IDs first; name fallback if needed.
    """
    if not (home_pid and away_pid and season and league_id):
        return ""

    def norm(s: str | None) -> str:
        if not s:
            return ""
        s = s.lower().strip()
        for t in [" fc", " cf", " sc", " afc", " ssc", " ac", " calcio", ".", "-"]:
            s = s.replace(t, " ")
        return " ".join(s.split())

    H = _af_recent(home_pid, season=season, limit=lookback, league_id=league_id) or []
    A = _af_recent(away_pid, season=season, limit=lookback, league_id=league_id) or []

    def maps(rows):
        by_id, by_nm = {}, {}
        for r in rows:
            oid = r.get("opponent_id") or r.get("opponent_team_id")
            if oid: by_id[int(oid)] = r
            nm = r.get("opponent")
            if nm: by_nm[norm(nm)] = r
        return by_id, by_nm

    H_id, H_nm = maps(H)
    A_id, A_nm = maps(A)

    take: List[str] = []
    shared_ids = list(set(H_id) & set(A_id))
    if shared_ids:
        for oid in shared_ids[:5]:
            h, a = H_id[oid], A_id[oid]
            h_score = h.get("score") or f"{h.get('goals_for', '?')}-{h.get('goals_against', '?')}"
            a_score = a.get("score") or f"{a.get('goals_for', '?')}-{a.get('goals_against', '?')}"
            opp = h.get("opponent") or a.get("opponent") or f"Team {oid}"
            take.append(f"- vs {opp}: {h.get('result','?')} {h_score} | {a.get('result','?')} {a_score}")
    else:
        shared_nm = list(set(H_nm) & set(A_nm))
        for nm in shared_nm[:5]:
            h, a = H_nm[nm], A_nm[nm]
            h_score = h.get("score") or f"{h.get('goals_for', '?')}-{h.get('goals_against', '?')}"
            a_score = a.get("score") or f"{a.get('goals_for', '?')}-{a.get('goals_against', '?')}"
            opp = h.get("opponent") or a.get("opponent") or "Opponent"
            take.append(f"- vs {opp}: {h.get('result','?')} {h_score} | {a.get('result','?')} {a_score}")

    return ("Shared Opponents (recent, same league):\n" + "\n".join(take)) if take else ""


# ---- ORIGINAL helper (kept for expert route, strict single-source) ----------
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


# ---- NEW robust helper (multi-source fallback for previews) -----------------
def _get_win_probs_and_edges_any(
    db: Session,
    fixture_id: int,
    sources: List[str] = ["team_form", "consensus_calib", "consensus_v2"],
    top_k_edges: int = 5,
) -> dict:
    probs = {"home": None, "draw": None, "away": None}
    used_source = None

    # try each source until we find probs
    for src in sources:
        latest = (
            db.query(ModelProb.market, func.max(ModelProb.as_of).label("latest"))
            .filter(ModelProb.fixture_id == fixture_id, ModelProb.source == src)
            .group_by(ModelProb.market)
            .subquery()
        )
        rows = (
            db.query(ModelProb)
            .join(latest, and_(ModelProb.market == latest.c.market, ModelProb.as_of == latest.c.latest))
            .all()
        )
        if rows:
            for r in rows:
                m = (r.market or "").upper()
                if m in ("HOME_WIN", "HOME"): probs["home"] = float(r.prob)
                elif m == "DRAW": probs["draw"] = float(r.prob)
                elif m in ("AWAY_WIN", "AWAY"): probs["away"] = float(r.prob)
            used_source = src
            break

    # edges: prefer same source if we found one; else take best-all
    q = db.query(Edge).filter(Edge.fixture_id == fixture_id)
    if used_source:
        rows = q.filter(Edge.model_source == used_source).order_by(Edge.edge.desc(), Edge.created_at.desc()).all()
    else:
        rows = q.order_by(Edge.edge.desc(), Edge.created_at.desc()).all()

    pos_edges = [
        {
            "market": e.market,
            "bookmaker": e.bookmaker,
            "price": float(e.price) if e.price is not None else None,
            "prob": float(e.prob) if e.prob is not None else None,
            "edge": float(e.edge) if e.edge is not None else None,
            "model_source": e.model_source,
        }
        for e in rows
        if (e.edge or 0) > 0
    ][:top_k_edges]

    return {"probs": probs, "edges": pos_edges, "source_used": used_source}

# -------------------------------------------------------------------
# Prompt builder (now includes team-stats + shared opponents + probs)
# -------------------------------------------------------------------

def _build_prompt_enriched(
    fixture: Fixture,
    form_data: dict,
    team_stats: dict,
    shared_text: str,
    probs: dict,
    n: int,
) -> str:
    home = fixture.home_team or "Home"
    away = fixture.away_team or "Away"
    comp  = fixture.comp or fixture.sport or "match"

    # --- recent form (aggregates) ---
    hf, af = form_data.get("home", {}) or {}, form_data.get("away", {}) or {}
    h_w, h_d, h_l = int(hf.get("wins", 0)), int(hf.get("draws", 0)), int(hf.get("losses", 0))
    a_w, a_d, a_l = int(af.get("wins", 0)), int(af.get("draws", 0)), int(af.get("losses", 0))
    h_gf = float(hf.get("avg_goals_for") or 0.0);   h_ga = float(hf.get("avg_goals_against") or 0.0)
    a_gf = float(af.get("avg_goals_for") or 0.0);   a_ga = float(af.get("avg_goals_against") or 0.0)

    # --- season/comp summary ---
    summary  = team_stats.get("summary") or {}
    home_sum = summary.get("home") or {}
    away_sum = summary.get("away") or {}
    h_form   = home_sum.get("form") or "?"
    a_form   = away_sum.get("form") or "?"
    h_avg_gf = float(home_sum.get("avg_gf") or 0.0); h_avg_ga = float(home_sum.get("avg_ga") or 0.0)
    a_avg_gf = float(away_sum.get("avg_gf") or 0.0); a_avg_ga = float(away_sum.get("avg_ga") or 0.0)

    # --- probabilities line ---
    p_home, p_draw, p_away = probs.get("home"), probs.get("draw"), probs.get("away")
    _pct = lambda x: f"{(x*100):.1f}%" if x is not None else "n/a"
    prob_line = f"Model probabilities — {home}: {_pct(p_home)}, Draw: {_pct(p_draw)}, {away}: {_pct(p_away)}." \
                if any(v is not None for v in (p_home, p_draw, p_away)) else ""

    # --- shared opponents (force a compact sentence) ---
    shared_sentence = ""
    if shared_text:
        # Expect lines like: "- vs AC Milan: loss 0-2 | loss 2-1"
        bullets = [ln.strip() for ln in shared_text.splitlines() if ln.strip().startswith("- ")]
        parsed = []
        for ln in bullets[:4]:  # cap to 4
            m = re.match(r"-\s*vs\s*(.+?):\s*(.+?)\s*\|\s*(.+)$", ln)
            if m:
                opp, home_res, away_res = m.groups()
                parsed.append(f"{opp} ({home_res} / {away_res})")
        if parsed:
            shared_sentence = "Shared opponents: " + ", ".join(parsed) + "."

    # --- final prompt (explicitly require the sentence) ---
    return f"""Write a concise {comp} preview (6–8 lines) for {home} vs {away}.
Use ONLY the factual data below — do not invent lineups or player news.
Include the shared-opponents sentence exactly once.

Recent form (last {n}):
- {home}: {h_w}W–{h_d}D–{h_l}L, {h_gf:.1f} scored / {h_ga:.1f} conceded.
- {away}: {a_w}W–{a_d}D–{a_l}L, {a_gf:.1f} scored / {a_ga:.1f} conceded.

Season stats:
- {home}: avg {h_avg_gf:.1f} GF / {h_avg_ga:.1f} GA, Form {h_form}
- {away}: avg {a_avg_gf:.1f} GF / {a_avg_ga:.1f} GA, Form {a_form}

{prob_line}
{shared_sentence}
End with one balanced 'what decides it' line."""
# -------------------------------------------------------------------
# Routes — Generate Preview (uses multi-source + enrichments)
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
    if not form_data or (not form_data.get("home") and not form_data.get("away")):
        raise HTTPException(status_code=500, detail="Form data unavailable")

    # enrich
    team_stats = _get_team_stats(fixture_id)
    ctx = _fixture_ctx_for_shared(fx)
    shared_text = _shared_opponents_text(ctx["home_pid"], ctx["away_pid"], ctx["season"], ctx["league_id"], n)
    mdl = _get_win_probs_and_edges_any(db, fixture_id)
    probs = mdl["probs"]

    prompt = _build_prompt_enriched(fx, form_data, team_stats, shared_text, probs, n)
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

@router.get("/debug/shared")
def debug_shared(fixture_id: int, db: Session = Depends(get_db)):
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx:
        raise HTTPException(status_code=404, detail="Fixture not found")

    ctx = _fixture_ctx_for_shared(fx)
    shared = _shared_opponents_text(
        ctx["home_pid"], ctx["away_pid"], ctx["season"], ctx["league_id"], 8
    )
    return {
        "fixture_id": fixture_id,
        "home_pid": ctx["home_pid"],
        "away_pid": ctx["away_pid"],
        "league_id": ctx["league_id"],
        "season": ctx["season"],
        "shared_text": shared,
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

            form_data = _form_from_db(db, fx.id, n=n)
            if not form_data or (not form_data.get("home") and not form_data.get("away")):
                skipped += 1
                continue

            team_stats = _get_team_stats(fx.id)
            ctx = _fixture_ctx_for_shared(fx)
            shared_text = _shared_opponents_text(ctx["home_pid"], ctx["away_pid"], ctx["season"], ctx["league_id"], n)
            mdl = _get_win_probs_and_edges_any(db, fx.id)
            probs = mdl["probs"]

            prompt = _build_prompt_enriched(fx, form_data, team_stats, shared_text, probs, n)
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
                existing.preview = text
                existing.tokens = tokens
                existing.updated_at = datetime.utcnow()
                db.add(existing)
            else:
                ai = AIPreview(
                    fixture_id=fx.id,
                    day=day_obj,                      # save for requested day
                    sport=fx.sport or sport,
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
# (kept as your working version; still uses single-source team_form)

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

    # context
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