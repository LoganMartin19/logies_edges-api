# api/app/routes/preview.py
from __future__ import annotations

import os
import re
import json
from datetime import datetime, timezone, date, timedelta
from typing import Any, Dict, List, Tuple, Optional

import requests
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc
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
from ..services.form import get_fixture_form_summary
from ..services.apifootball import (
    get_fixture as _af_get_fixture,
    get_team_recent_results as _af_recent,
)

router = APIRouter(prefix="/ai/preview", tags=["AI Preview"])
pub = APIRouter(prefix="/public/ai/preview", tags=["Public AI Preview"])

# -------------------------------------------------------------------
# Base / Config
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
# Helpers — Form, Stats, Shared Opponents
# -------------------------------------------------------------------

def _form_from_db(db: Session, fixture_id: int, n: int) -> Dict[str, Any]:
    payload = get_fixture_form_summary(db, fixture_id, n=n) or {}
    return {"home": payload.get("home") or {}, "away": payload.get("away") or {}}


def _get_team_stats(fixture_id: int) -> Dict[str, Any]:
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

# -------------------------------------------------------------------
# Domestic / H2H helpers (for cup intelligence)
# -------------------------------------------------------------------

def _is_cup_like(name: str | None) -> bool:
    if not name:
        return False
    n = name.lower()
    cup_markers = [
        "cup", "coppa", "copa", "dfb", "fa ", "super cup", "supercup", "shield",
        "trophy", "league cup", "carabao", "dfb-pokal", "knvb", "taça", "reykjavik",
        "challenge", "community", "supertaça", "pokal"
    ]
    return any(t in n for t in cup_markers)

# --- replace _pick_most_recent_h2h with this ---
def _norm_team(s: str | None) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    junk = [" fc", " cf", " sc", " afc", " ssc", " ac", " calcio", " bk", ".", "-", "  "]
    for t in junk:
        s = s.replace(t, " ")
    return " ".join(s.split())

def _pick_most_recent_h2h(home_pid: int, away_pid: int, season: int, lookback: int = 20) -> dict:
    """
    Return the most-recent competitive match between the two teams, across any competition.
    Uses opponent_id matching (no league filter), sorts by parsed datetime, and
    returns neutral info + winner side so the caller can format with team names.
    """
    H = _af_recent(home_pid, season=season, limit=lookback) or []
    A = _af_recent(away_pid, season=season, limit=lookback) or []

    # index both lists by opponent_id for fast pairing
    def by_opp(rows):
        m = {}
        for r in rows:
            oid = r.get("opponent_id") or r.get("opponent_team_id")
            if not oid:
                continue
            m.setdefault(int(oid), []).append(r)
        return m

    Hmap = by_opp(H)
    Amap = by_opp(A)

    # candidates are rows where each side’s recent list has the other as opponent
    pairs = []
    if away_pid in Hmap:
        for hr in Hmap[away_pid]:
            pairs.append(("home_side", hr))  # row from home_pid POV
    if home_pid in Amap:
        for ar in Amap[home_pid]:
            pairs.append(("away_side", ar))  # row from away_pid POV

    def parse_dt(s: str | None) -> float:
        if not s:
            return 0.0
        try:
            return datetime.fromisoformat(s.replace("Z", "")).timestamp()
        except Exception:
            return 0.0

    # Sort by actual fixture datetime
    pairs.sort(key=lambda tup: parse_dt((tup[1].get("date") or tup[1].get("fixture_date"))), reverse=True)
    if not pairs:
        return {}

    side, row = pairs[0]
    gf = float(row.get("goals_for") or 0)
    ga = float(row.get("goals_against") or 0)
    score = f"{int(gf)}-{int(ga)}"
    dt = (row.get("date") or row.get("fixture_date")) or ""
    comp = row.get("league") or row.get("competition") or "league"

    # winner_pid decided from the POV of the row’s team
    if gf > ga:
        winner_pid = home_pid if side == "home_side" else away_pid
    elif ga > gf:
        winner_pid = away_pid if side == "home_side" else home_pid
    else:
        winner_pid = None

    return {
        "date": dt,
        "competition": comp,
        "score": score,           # always "X-Y" from the reporting side’s POV (already neutral enough)
        "winner_pid": winner_pid, # None if draw
        "source_side": side,      # "home_side" or "away_side"
    }

def _pretty_h2h_line(fx: Fixture, h2h: dict) -> str:
    if not h2h: 
        return ""
    # who won?
    side = (h2h.get("side") or "").lower()
    res  = (h2h.get("result") or "").lower()
    home, away = fx.home_team or "Home", fx.away_team or "Away"

    if res == "draw":
        outcome = "It finished level"
    elif res == "win":
        winner = away if side == "away" else home if side == "home" else "The visitors"
        outcome = f"{winner} won"
    elif res == "loss":
        loser = away if side == "away" else home if side == "home" else "The visitors"
        outcome = f"{loser} lost"
    else:
        outcome = "Result unknown"

    score = h2h.get("score") or "?"
    comp  = h2h.get("competition") or "a league match"

    # days-ago tag
    when = ""
    try:
        dt = datetime.fromisoformat((h2h.get("date") or "").replace("Z", ""))
        days = (datetime.utcnow() - dt).days
        when = f" just {days} days ago" if 0 <= days <= 10 else f" on {dt.date().isoformat()}"
    except Exception:
        pass

    return f"The sides met{when} in {comp}: {outcome} {score}."
# -------------------------------------------------------------------
# Probability helpers (STRICT H/D/A)
# -------------------------------------------------------------------

HDA_ONLY = ("HOME_WIN", "DRAW", "AWAY_WIN")

def _pick_latest_complete_triplet(rows: List[ModelProb]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[datetime]]:
    buckets: Dict[str, Dict[str, ModelProb]] = {}
    for r in rows:
        tkey = (r.as_of.replace(microsecond=0)).isoformat()
        buckets.setdefault(tkey, {})
        buckets[tkey][(r.market or "").upper()] = r
    for tkey in sorted(buckets.keys(), reverse=True):
        bucket = buckets[tkey]
        if all(m in bucket for m in HDA_ONLY):
            h = float(bucket["HOME_WIN"].prob)
            d = float(bucket["DRAW"].prob)
            a = float(bucket["AWAY_WIN"].prob)
            as_of = bucket["HOME_WIN"].as_of
            return h, d, a, as_of
    return None, None, None, None


def _get_win_probs_strict(db: Session, fixture_id: int, source: str = "team_form") -> Dict[str, Optional[float]]:
    rows: List[ModelProb] = (
        db.query(ModelProb)
        .filter(ModelProb.fixture_id == fixture_id)
        .filter(ModelProb.source == source)
        .filter(ModelProb.market.in_(HDA_ONLY))
        .order_by(desc(ModelProb.as_of))
        .all()
    )
    h, d, a, _ = _pick_latest_complete_triplet(rows)
    return {"home": h, "draw": d, "away": a}


def _get_win_probs_and_edges_any(db: Session, fixture_id: int, sources: List[str] = ["team_form"], top_k_edges: int = 5) -> dict:
    probs = {"home": None, "draw": None, "away": None}
    used_source = None

    for src in sources:
        strict = _get_win_probs_strict(db, fixture_id, source=src)
        if all(strict[k] is not None for k in ("home", "draw", "away")):
            probs = strict
            used_source = src
            break

    q = db.query(Edge).filter(Edge.fixture_id == fixture_id)
    if used_source:
        edges_rows = q.filter(Edge.model_source == used_source).order_by(Edge.edge.desc(), Edge.created_at.desc()).all()
    else:
        edges_rows = q.order_by(Edge.edge.desc(), Edge.created_at.desc()).all()

    pos_edges = [
        {
            "market": e.market,
            "bookmaker": e.bookmaker,
            "price": float(e.price) if e.price is not None else None,
            "prob": float(e.prob) if e.prob is not None else None,
            "edge": float(e.edge) if e.edge is not None else None,
            "model_source": e.model_source,
        }
        for e in edges_rows if (e.edge or 0) > 0
    ][:top_k_edges]

    return {"probs": probs, "edges": pos_edges, "source_used": used_source or "unknown"}

# -------------------------------------------------------------------
# Prompt builder (unchanged)
# -------------------------------------------------------------------

def _build_prompt_enriched(fixture: Fixture, form_data: dict, team_stats: dict, shared_text: str, probs: dict, n: int) -> str:
    home = fixture.home_team or "Home"
    away = fixture.away_team or "Away"
    comp = fixture.comp or fixture.sport or "match"

    hf, af = form_data.get("home", {}) or {}, form_data.get("away", {}) or {}
    h_w, h_d, h_l = int(hf.get("wins", 0)), int(hf.get("draws", 0)), int(hf.get("losses", 0))
    a_w, a_d, a_l = int(af.get("wins", 0)), int(af.get("draws", 0)), int(af.get("losses", 0))
    h_gf = float(hf.get("avg_goals_for") or 0.0)
    h_ga = float(hf.get("avg_goals_against") or 0.0)
    a_gf = float(af.get("avg_goals_for") or 0.0)
    a_ga = float(af.get("avg_goals_against") or 0.0)

    summary = team_stats.get("summary") or {}
    home_sum = summary.get("home") or {}
    away_sum = summary.get("away") or {}
    h_form = home_sum.get("form") or "?"
    a_form = away_sum.get("form") or "?"
    h_avg_gf = float(home_sum.get("avg_gf") or 0.0)
    h_avg_ga = float(home_sum.get("avg_ga") or 0.0)
    a_avg_gf = float(away_sum.get("avg_gf") or 0.0)
    a_avg_ga = float(away_sum.get("avg_ga") or 0.0)

    # normalize to %
    p_home, p_draw, p_away = probs.get("home"), probs.get("draw"), probs.get("away")
    vals = [v for v in (p_home, p_draw, p_away) if v is not None]
    if vals and sum(vals) > 0:
        total = sum(vals)
        scale = 100.0 / total
        if p_home is not None: p_home *= scale
        if p_draw is not None: p_draw *= scale
        if p_away is not None: p_away *= scale

    def _pct(v): return f"{v:.1f}%" if v is not None else "n/a"

    prob_line = f"Model probabilities — {home}: {_pct(p_home)}, Draw: {_pct(p_draw)}, {away}: {_pct(p_away)}."
    shared_sentence = ""
    if shared_text:
        bullets = [ln.strip() for ln in shared_text.splitlines() if ln.strip().startswith("- ")]
        parsed = []
        for ln in bullets[:4]:
            m = re.match(r"-\s*vs\s*(.+?):\s*(.+?)\s*\|\s*(.+)$", ln)
            if m:
                opp, home_res, away_res = m.groups()
                parsed.append(f"{opp} ({home_res} / {away_res})")
        if parsed:
            shared_sentence = "Shared opponents: " + ", ".join(parsed) + "."

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
End with one balanced 'what decides it' line."""\

# -------- League-only form helpers (for expert depth) -----------------

def _form_stats_from_rows(rows: List[dict], n: int) -> dict:
    """Compute W/D/L and avg GF/GA from API-Football 'recent results' rows."""
    take = rows[:n] if rows else []
    w = d = l = gf_sum = ga_sum = 0
    seq = []
    for r in take:
        res = (r.get("result") or "").lower()
        gf = float(r.get("goals_for") or 0)
        ga = float(r.get("goals_against") or 0)
        gf_sum += gf; ga_sum += ga
        if res.startswith("w"): w += 1; seq.append("W")
        elif res.startswith("d"): d += 1; seq.append("D")
        elif res.startswith("l"): l += 1; seq.append("L")
        else: seq.append("?")
    games = max(1, len(take))
    return {
        "wins": w, "draws": d, "losses": l,
        "avg_goals_for": round(gf_sum / games, 2),
        "avg_goals_against": round(ga_sum / games, 2),
        "sequence": " ".join(seq)
    }

def _league_form_pair(ctx: dict, n: int = 5) -> dict:
    """
    Get last-n league-only form for both sides using API-Football.
    ctx = {home_pid, away_pid, season, league_id}
    """
    lid = ctx.get("league_id"); season = ctx.get("season")
    hpid = ctx.get("home_pid"); apid = ctx.get("away_pid")
    if not (lid and season and hpid and apid):
        return {"home": {}, "away": {}}

    H = _af_recent(int(hpid), season=int(season), limit=max(10, n), league_id=int(lid)) or []
    A = _af_recent(int(apid), season=int(season), limit=max(10, n), league_id=int(lid)) or []
    return {
        "home": _form_stats_from_rows(H, n),
        "away": _form_stats_from_rows(A, n),
    }

# -------------------------------------------------------------------
# Routes — Generate Preview (team_form + strict probs)
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

    team_stats = _get_team_stats(fixture_id)
    ctx = _fixture_ctx_for_shared(fx)
    shared_text = _shared_opponents_text(ctx["home_pid"], ctx["away_pid"], ctx["season"], ctx["league_id"], n)

    mdl = _get_win_probs_and_edges_any(db, fixture_id, sources=["team_form"], top_k_edges=5)
    probs = mdl["probs"]

    prompt = _build_prompt_enriched(fx, form_data, team_stats, shared_text, probs, n)
    client = _openai_client()

    try:
        resp = client.chat_completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a neutral, sharp football analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=280,
        )
    except AttributeError:
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

# -------------------------------------------------------------------
# Debug helpers
# -------------------------------------------------------------------

@router.get("/winprobs")
def admin_win_probs(fixture_id: int, source: str = Query("team_form"), db: Session = Depends(get_db)):
    return _get_win_probs_strict(db, fixture_id, source=source)


@router.get("/edges")
def admin_edges(fixture_id: int, source: str = Query("team_form"), db: Session = Depends(get_db)):
    data = _get_win_probs_and_edges_any(db, fixture_id, sources=[source], top_k_edges=50)
    return data["edges"]


@router.get("/debug/probs")
def debug_preview_probs(fixture_id: int = Query(...), db: Session = Depends(get_db)):
    mdl = _get_win_probs_and_edges_any(db, fixture_id, sources=["team_form"])
    probs = mdl["probs"]
    p_home, p_draw, p_away = probs.get("home"), probs.get("draw"), probs.get("away")
    vals = [v for v in [p_home, p_draw, p_away] if v is not None]
    if vals and sum(vals) > 0:
        total = sum(vals)
        scale = 100.0 / total
        if p_home is not None: p_home *= scale
        if p_draw is not None: p_draw *= scale
        if p_away is not None: p_away *= scale

    def _pct(v): return f"{v:.1f}%" if v is not None else "n/a"

    return {
        "fixture_id": fixture_id,
        "source_used": mdl.get("source_used"),
        "normalized_probs": {"home": _pct(p_home), "draw": _pct(p_draw), "away": _pct(p_away)}
    }


@router.get("/debug/winprobs-rows")
def debug_winprob_rows(
    fixture_id: int = Query(...),
    source: str = Query("team_form"),
    db: Session = Depends(get_db),
):
    rows: List[ModelProb] = (
        db.query(ModelProb)
        .filter(ModelProb.fixture_id == fixture_id)
        .filter(ModelProb.source == source)
        .filter(ModelProb.market.in_(HDA_ONLY))
        .order_by(desc(ModelProb.as_of))
        .all()
    )
    return [
        {
            "market": r.market,
            "prob": float(r.prob),
            "as_of": r.as_of.isoformat()
        }
        for r in rows
    ]

# -------------------------------------------------------------------
# Batch (daily) generation  — FIXED fx.id
# -------------------------------------------------------------------

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

            mdl = _get_win_probs_and_edges_any(db, fx.id, sources=["team_form"], top_k_edges=5)
            probs = mdl["probs"]

            prompt = _build_prompt_enriched(fx, form_data, team_stats, shared_text, probs, n)
            client = _openai_client()

            try:
                resp = client.chat_completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a sharp, neutral football analyst."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.6,
                    max_tokens=280,
                )
            except AttributeError:
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
                    day=day_obj,
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

# -------------------------------------------------------------------
# Public reader
# -------------------------------------------------------------------

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

# -------------------------------------------------------------------
# Expert bettor analysis (uses same strict probs)
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Expert bettor analysis (now cup-aware + H2H context)
# -------------------------------------------------------------------

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

    # ---------- Core data ----------
    # All competitions form (from DB summary)
    form_all = _form_from_db(db, fixture_id, n=n)
    home_all = form_all.get("home", {}) or {}
    away_all = form_all.get("away", {}) or {}

    # Model probabilities
    model = _get_win_probs_and_edges_any(db, fixture_id, sources=["team_form"], top_k_edges=5)
    p_home = model["probs"]["home"]
    p_draw = model["probs"]["draw"]
    p_away = model["probs"]["away"]
    top_edges = model["edges"]

    # League positions
    league_positions = (
        db.query(LeagueStanding.team, LeagueStanding.position)
        .filter(LeagueStanding.league == fx.comp)
        .all()
    )
    home_pos = next((p for t, p in league_positions if t == fx.home_team), None)
    away_pos = next((p for t, p in league_positions if t == fx.away_team), None)

    # Cup/H2H + league-only form
    ctx = _fixture_ctx_for_shared(fx)
    cup_like = _is_cup_like(fx.comp)
    league_form = _league_form_pair(ctx, n=n)
    home_lg = league_form.get("home", {}) or {}
    away_lg = league_form.get("away", {}) or {}

    # Handle missing league data gracefully
    if home_lg.get("games", 0) == 0:
        home_lg["note"] = "No recent league matches recorded."
    if away_lg.get("games", 0) == 0:
        away_lg["note"] = "No recent league matches recorded."

    # -------- Corrected H2H logic --------
    h2h_note = ""
    if ctx.get("home_pid") and ctx.get("away_pid"):
        recent_h2h = _pick_most_recent_h2h(ctx["home_pid"], ctx["away_pid"], ctx["season"])
        if recent_h2h:
            days_ago_txt = ""
            try:
                dt = datetime.fromisoformat((recent_h2h.get("date") or "").replace("Z", ""))
                delta_days = (datetime.utcnow() - dt).days
                if 0 <= delta_days <= 14:
                    days_ago_txt = f" ({delta_days} days ago)"
            except Exception:
                pass

            comp_nm = recent_h2h.get("competition") or "league"
            score = recent_h2h.get("score") or "?"
            winner_pid = recent_h2h.get("winner_pid")

            home_name = fx.home_team or "Home"
            away_name = fx.away_team or "Away"

            if winner_pid is None:
                h2h_note = f"The sides met{days_ago_txt} in {comp_nm}: it finished {score}."
            else:
                winner_name = home_name if winner_pid == ctx["home_pid"] else away_name
                loser_name = away_name if winner_pid == ctx["home_pid"] else home_name
                h2h_note = f"The sides met{days_ago_txt} in {comp_nm}: {winner_name} beat {loser_name} {score}."

    # ---------- Prompt ----------
    prompt = f"""
You are a professional football betting analyst. Be concise and data-grounded.

Fixture: {fx.home_team} vs {fx.away_team} ({fx.comp or fx.sport})
League positions (if known): {fx.home_team}: {home_pos or "?"}, {fx.away_team}: {away_pos or "?"}

ALL competitions – last {n}:
- {fx.home_team}: {home_all}
- {fx.away_team}: {away_all}

LEAGUE ONLY – last {n}:
- {fx.home_team}: {home_lg}
- {fx.away_team}: {away_lg}

Recent H2H (if present): {h2h_note or "none"}

Model win probabilities:
- home: {p_home}
- draw: {p_draw}
- away: {p_away}

Edges (potential value bets):
{top_edges}

Respond with one JSON object containing:
- "paragraphs": 3–4 short paragraphs of analytical insight (no Markdown):
   1. Compare overall form across all competitions.
   2. Discuss league-only trends (W/D/L, avg GF/GA, strengths/weaknesses).
   3. Include tactical implications and recent H2H summary.
   4. Conclude with betting implications, referencing model probabilities and edges.
- "probabilities": percentages (home/draw/away).
- "best_bets": up to 3 suggested bets with rationale.
- "confidence": one of ["Low","Medium","High"].
- "disclaimer": short note on variance and responsible betting.
    """.strip()

    # ---------- LLM call ----------
    client = _openai_client()
    try:
        resp = client.chat_completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.45,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
    except AttributeError:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.45,
            max_tokens=600,
            response_format={"type": "json_object"},
        )

    try:
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        data = {
            "paragraphs": ["Analysis unavailable."],
            "probabilities": {"home": None, "draw": None, "away": None},
            "best_bets": [],
            "confidence": "Low",
            "disclaimer": "No bet if price/edge isn’t there.",
        }

    # ---------- Persist ----------
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
        "cup_context": _is_cup_like(fx.comp),
        "h2h_note": h2h_note,
        "league_form": league_form,
    }

@router.get("/debug/h2h")
def debug_h2h(fixture_id: int = Query(...), db: Session = Depends(get_db)):
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx:
        raise HTTPException(status_code=404, detail="Fixture not found")
    ctx = _fixture_ctx_for_shared(fx)
    if not (ctx["home_pid"] and ctx["away_pid"]):
        return {"fixture_id": fixture_id, "message": "Missing provider team ids", "ctx": ctx}
    h2h = _pick_most_recent_h2h(ctx["home_pid"], ctx["away_pid"], ctx["season"])
    return {"fixture_id": fixture_id, "ctx": ctx, "h2h": h2h}