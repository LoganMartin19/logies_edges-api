# api/app/routes/insights.py

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


def _f(x, default=None):
    """Safe float conversion."""
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


def _i(x, default=None):
    """Safe int conversion."""
    if x is None:
        return default
    try:
        return int(x)
    except Exception:
        return default


def _wdl(form: dict):
    """
    Try to extract W/D/L from common keys.
    Supports multiple naming styles without breaking.
    """
    w = _i(form.get("wins") or form.get("w") or form.get("form_wins"), 0)
    d = _i(form.get("draws") or form.get("d") or form.get("form_draws"), 0)
    l = _i(form.get("losses") or form.get("l") or form.get("form_losses"), 0)
    # if all zero, treat as unknown
    if w == 0 and d == 0 and l == 0:
        return None
    return w, d, l


def mk_blurb(market: str, home_form: dict, away_form: dict, p: float) -> str:
    """
    Rich-but-short blurbs. Defensive on missing fields.
    Expects avg_gf/avg_ga where possible (your hybrid form already uses these).
    """
    hg = _f(home_form.get("avg_gf"))
    hga = _f(home_form.get("avg_ga"))
    ag = _f(away_form.get("avg_gf"))
    aga = _f(away_form.get("avg_ga"))

    hw = _wdl(home_form)
    aw = _wdl(away_form)

    # --- Generic fallback if form lacks key metrics ---
    if hg is None or hga is None or ag is None or aga is None:
        pct = f"{p*100:.0f}%"
        if market in ("BTTS_Y", "BTTS_N"):
            return f"Fair price is derived from CSB’s form model. CSB prices this at ~{pct}."
        if market.startswith("O") or market.startswith("U"):
            return f"Fair price is derived from CSB’s goals model. CSB prices this at ~{pct}."
        return f"Fair price is derived from CSB’s form-based model probabilities (~{pct})."

    # Useful derived signals
    total_gf = hg + ag
    total_ga = hga + aga
    goal_env = (total_gf + total_ga) / 2.0  # rough “game environment”
    home_goal_diff = hg - hga
    away_goal_diff = ag - aga

    # small helper strings
    wdl_home = f"{hw[0]}W-{hw[1]}D-{hw[2]}L" if hw else None
    wdl_away = f"{aw[0]}W-{aw[1]}D-{aw[2]}L" if aw else None

    # ---------------- BTTS ----------------
    if market == "BTTS_Y":
        bits = []
        bits.append(f"Both sides score {hg:.1f}/{ag:.1f} gpg")
        bits.append(f"and concede {hga:.1f}/{aga:.1f}.")
        if wdl_home and wdl_away:
            bits.append(f"Recent form: {wdl_home} vs {wdl_away}.")
        bits.append(f"CSB prices BTTS Yes at ~{p*100:.0f}%.")
        return " ".join(bits)

    if market == "BTTS_N":
        bits = []
        # We’ll lean on one-sided scoring / stronger defence signal
        bits.append(f"At least one side looks likely to be contained:")
        bits.append(f"GF {hg:.1f}/{ag:.1f}, GA {hga:.1f}/{aga:.1f}.")
        if wdl_home and wdl_away:
            bits.append(f"Form: {wdl_home} vs {wdl_away}.")
        bits.append(f"CSB prices BTTS No at ~{p*100:.0f}%.")
        return " ".join(bits)

    # -------------- Totals (O/U 2.5) --------------
    if market == "O2.5":
        bits = []
        bits.append(f"Goal environment is high (GF total {total_gf:.1f}, GA total {total_ga:.1f}).")
        bits.append(f"That points to chances at both ends.")
        bits.append(f"CSB prices Over 2.5 at ~{p*100:.0f}%.")
        return " ".join(bits)

    if market == "U2.5":
        bits = []
        bits.append(f"Goal environment leans lower (GF total {total_gf:.1f}, GA total {total_ga:.1f}).")
        bits.append(f"Defensive trend suggests fewer big chances.")
        bits.append(f"CSB prices Under 2.5 at ~{p*100:.0f}%.")
        return " ".join(bits)

    # ---------------- 1X2 ----------------
    if market == "HOME_WIN":
        bits = []
        bits.append(f"Home trend {home_goal_diff:+.1f} goal diff (GF {hg:.1f}, GA {hga:.1f})")
        bits.append(f"vs away {away_goal_diff:+.1f} (GF {ag:.1f}, GA {aga:.1f}).")
        if wdl_home and wdl_away:
            bits.append(f"Form: {wdl_home} vs {wdl_away}.")
        bits.append(f"CSB prices Home Win at ~{p*100:.0f}%.")
        return " ".join(bits)

    if market == "AWAY_WIN":
        bits = []
        bits.append(f"Away trend {away_goal_diff:+.1f} goal diff (GF {ag:.1f}, GA {aga:.1f})")
        bits.append(f"vs home {home_goal_diff:+.1f} (GF {hg:.1f}, GA {hga:.1f}).")
        if wdl_home and wdl_away:
            bits.append(f"Form: {wdl_home} vs {wdl_away}.")
        bits.append(f"CSB prices Away Win at ~{p*100:.0f}%.")
        return " ".join(bits)

    if market == "DRAW":
        bits = []
        bits.append(f"Both teams profile similarly (GF {hg:.1f}/{ag:.1f}, GA {hga:.1f}/{aga:.1f}).")
        bits.append(f"That often increases draw likelihood.")
        if wdl_home and wdl_away:
            bits.append(f"Form: {wdl_home} vs {wdl_away}.")
        bits.append(f"CSB prices Draw at ~{p*100:.0f}%.")
        return " ".join(bits)

    # ---------------- Double Chance ----------------
    if market == "1X":
        return (
            f"Double Chance covers Home or Draw. "
            f"Home trend {home_goal_diff:+.1f} vs away {away_goal_diff:+.1f} "
            f"(GF {hg:.1f}/{ag:.1f}, GA {hga:.1f}/{aga:.1f}). "
            f"CSB prices 1X at ~{p*100:.0f}%."
        )

    if market == "X2":
        return (
            f"Double Chance covers Away or Draw. "
            f"Away trend {away_goal_diff:+.1f} vs home {home_goal_diff:+.1f} "
            f"(GF {ag:.1f}/{hg:.1f}, GA {aga:.1f}/{hga:.1f}). "
            f"CSB prices X2 at ~{p*100:.0f}%."
        )

    if market == "12":
        return (
            f"Double Chance excludes Draw (either team wins). "
            f"Goals profile suggests decisive outcomes are plausible "
            f"(GF total {total_gf:.1f}, GA total {total_ga:.1f}). "
            f"CSB prices 12 at ~{p*100:.0f}%."
        )

    # Generic fallback for any other market keys
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

    # Pull form (pass FIXTURE object)
    try:
        hybrid = get_hybrid_form_for_fixture(db, fx)
    except Exception:
        hybrid = None

    home_form = (hybrid or {}).get("home_form") or {}
    away_form = (hybrid or {}).get("away_form") or {}

    # Pull latest probs for this model source
    rows = (
        db.query(ModelProb)
        .filter(ModelProb.fixture_id == fixture_id, ModelProb.source == model)
        .order_by(desc(ModelProb.as_of))
        .all()
    )

    # newest-first, so first time we see a market is the latest prob for it
    latest = {}
    for r in rows:
        if not r.market:
            continue
        if r.market in latest:
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