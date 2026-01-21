from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc

from ..db import get_db
from ..models import Fixture, ModelProb
from ..services.form import get_hybrid_form_for_fixture
from ..services.utils import confidence_from_prob

router = APIRouter(prefix="/api", tags=["insights"])

EPS = 1e-6


def clamp(p: float) -> float:
    p = float(p)
    return max(EPS, min(1.0 - EPS, p))


def fair_odds(p: float) -> float:
    return round(1.0 / clamp(p), 2)


def _get(form: dict, *keys, default=None):
    """Safely get a value from form dict with fallback key names."""
    for k in keys:
        if k in form and form.get(k) is not None:
            return form.get(k)
    return default


def mk_blurb(market: str, home_form: dict, away_form: dict, p: float) -> str:
    """
    Short, scannable blurbs.
    Tries multiple key names because your hybrid form schema can vary.
    """
    # Common goal-rate keys (support multiple schemas)
    hg  = _get(home_form, "avg_gf", "avg_goals_for", "gf_avg", "avgGF", default=None)
    ag  = _get(away_form, "avg_gf", "avg_goals_for", "gf_avg", "avgGF", default=None)
    hga = _get(home_form, "avg_ga", "avg_goals_against", "ga_avg", "avgGA", default=None)
    aga = _get(away_form, "avg_ga", "avg_goals_against", "ga_avg", "avgGA", default=None)

    # Optional “hit rates” if present in your form payload
    home_btts = _get(home_form, "btts_last5", "btts_rate", "btts_hits", default=None)
    away_btts = _get(away_form, "btts_last5", "btts_rate", "btts_hits", default=None)
    home_o25  = _get(home_form, "o25_last5", "o25_rate", "over25_last5", default=None)
    away_o25  = _get(away_form, "o25_last5", "o25_rate", "over25_last5", default=None)

    # helpers
    def _fmt_rate(v):
        # allow "4/5" strings or numeric 0..1
        if v is None:
            return None
        if isinstance(v, str):
            return v
        try:
            fv = float(v)
            if 0.0 <= fv <= 1.0:
                return f"{round(fv*100):.0f}%"
            return f"{fv:.2f}"
        except Exception:
            return str(v)

    hb = _fmt_rate(home_btts)
    ab = _fmt_rate(away_btts)
    ho = _fmt_rate(home_o25)
    ao = _fmt_rate(away_o25)

    if market == "BTTS_Y":
        if hb is not None and ab is not None:
            return f"BTTS trend: Home {hb}, Away {ab}. CSB makes BTTS Yes ~{p*100:.0f}%."
        if hg is not None and ag is not None and hga is not None and aga is not None:
            return (
                f"Both teams score/concede (GF {float(hg):.1f}/{float(ag):.1f}, "
                f"GA {float(hga):.1f}/{float(aga):.1f}). BTTS Yes ~{p*100:.0f}%."
            )
        return f"CSB makes BTTS Yes ~{p*100:.0f}% based on recent form & goals profile."

    if market == "BTTS_N":
        return f"CSB makes BTTS No ~{p*100:.0f}% (one side likely blanked)."

    if market == "O2.5":
        if ho is not None and ao is not None:
            return f"Over 2.5 trend: Home {ho}, Away {ao}. CSB makes Over 2.5 ~{p*100:.0f}%."
        if hg is not None and ag is not None:
            return f"Attacking output points upward (avg GF {float(hg):.1f}+{float(ag):.1f}). Over 2.5 ~{p*100:.0f}%."
        return f"CSB makes Over 2.5 ~{p*100:.0f}% from recent goals form."

    if market == "U2.5":
        if hga is not None and aga is not None:
            return f"Defensive profile leans lower-scoring (avg GA {float(hga):.1f}+{float(aga):.1f}). Under 2.5 ~{p*100:.0f}%."
        return f"CSB makes Under 2.5 ~{p*100:.0f}% from recent goals form."

    if market in ("HOME_WIN", "AWAY_WIN", "DRAW"):
        return f"CSB fair price is based on form-weighted win/draw rates (p~{p*100:.0f}%)."

    if market in ("1X", "X2", "12"):
        return f"Double chance derived from 1X2 probabilities (p~{p*100:.0f}%)."

    return f"CSB fair odds are based on model probability of ~{p*100:.0f}%."


@router.get("/fixtures/{fixture_id}/insights")
def fixture_insights(
    fixture_id: int,
    model: str = "team_form",  # maps to ModelProb.source
    db: Session = Depends(get_db),
):
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx:
        raise HTTPException(status_code=404, detail="Fixture not found")

    # 1) Pull form (PASS FIXTURE OBJECT, not int)
    try:
        hybrid = get_hybrid_form_for_fixture(db, fx)
    except Exception:
        hybrid = None

    home_form = (hybrid or {}).get("home_form") or {}
    away_form = (hybrid or {}).get("away_form") or {}

    # 2) Pull latest probs for this model source
    rows = (
        db.query(ModelProb)
        .filter(ModelProb.fixture_id == fixture_id, ModelProb.source == model)
        .order_by(desc(ModelProb.as_of))
        .all()
    )

    # newest-first, so first time we see a market is the latest prob for it
    latest = {}
    for r in rows:
        if not r.market or r.market in latest:
            continue
        try:
            latest[r.market] = float(r.prob)
        except Exception:
            continue

    wanted = [
        "BTTS_Y", "BTTS_N",
        "O2.5", "U2.5",
        "HOME_WIN", "DRAW", "AWAY_WIN",
        "1X", "X2", "12",
    ]

    insights = []
    for m in wanted:
        p = latest.get(m)
        if p is None:
            continue
        p = clamp(p)
        insights.append(
            {
                "market": m,
                "prob": round(p, 4),
                "fair_odds": fair_odds(p),
                "confidence": confidence_from_prob(p),
                "blurb": mk_blurb(m, home_form, away_form, p),
            }
        )

    return {
        "fixture_id": fixture_id,
        "model": model,
        "insights": insights,
        "form": {"home": home_form, "away": away_form},
    }