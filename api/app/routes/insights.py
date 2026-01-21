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

def mk_blurb(market: str, home_form: dict, away_form: dict, p: float) -> str:
    # keep this deliberately short + consistent
    hg = home_form.get("avg_gf", None)
    ag = away_form.get("avg_gf", None)
    hga = home_form.get("avg_ga", None)
    aga = away_form.get("avg_ga", None)

    # fallback if any missing
    if hg is None or ag is None or hga is None or aga is None:
        return "Fair price is derived from CSB’s form-based model probabilities."

    if market == "BTTS_Y":
        return f"Both teams have been scoring/conceding recently (GF {hg:.1f}/{ag:.1f}, GA {hga:.1f}/{aga:.1f}). CSB makes BTTS Yes ~{p*100:.0f}%."
    if market == "O2.5":
        return f"Goals profile suggests a higher total (avg GF {hg:.1f}+{ag:.1f}). CSB makes Over 2.5 ~{p*100:.0f}%."
    if market == "U2.5":
        return f"Recent goal rates lean lower (avg GA {hga:.1f}+{aga:.1f}). CSB makes Under 2.5 ~{p*100:.0f}%."

    return f"CSB fair odds are based on model probability of ~{p*100:.0f}%."

@router.get("/fixtures/{fixture_id}/insights")
def fixture_insights(
    fixture_id: int,
    model: str = "team_form",   # or your default model
    db: Session = Depends(get_db),
):
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx:
        raise HTTPException(status_code=404, detail="Fixture not found")

    # 1) Pull form (you already use this in explain)
    hybrid = get_hybrid_form_for_fixture(db, fixture_id)
    home_form = (hybrid or {}).get("home_form") or {}
    away_form = (hybrid or {}).get("away_form") or {}

    # 2) Pull latest probs for this model
    rows = (
        db.query(ModelProb)
        .filter(ModelProb.fixture_id == fixture_id, ModelProb.model == model)
        .order_by(desc(ModelProb.created_at))
        .all()
    )

    latest = {}
    for r in rows:
        if r.market and isinstance(r.prob, (int, float)):
            latest.setdefault(r.market, float(r.prob))

    # 3) Choose which markets to show
    wanted = ["BTTS_Y", "BTTS_N", "O2.5", "U2.5", "HOME_WIN", "DRAW", "AWAY_WIN", "1X", "X2", "12"]

    insights = []
    for m in wanted:
        p = latest.get(m)
        if p is None:
            continue
        p = clamp(p)
        insights.append({
            "market": m,
            "prob": round(p, 4),
            "fair_odds": fair_odds(p),
            "confidence": confidence_from_prob(p),
            "blurb": mk_blurb(m, home_form, away_form, p),
        })

    return {
        "fixture_id": fixture_id,
        "model": model,
        "insights": insights,
        "form": {"home": home_form, "away": away_form},
    }