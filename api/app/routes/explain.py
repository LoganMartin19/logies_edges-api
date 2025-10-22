# api/app/routes/explain.py
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..db import get_db
from ..models import Fixture, ModelProb
from ..services.form import (
    get_recent_form,          # fallback only
    get_recent_fixtures,      # fallback only
    get_hybrid_form_for_fixture,  # ✅ API-backed hybrid
)
from ..services.league_strength import get_team_strength
from ..services.utils import confidence_from_prob

# Pull explanation helpers from edge.py
from ..edge import expected_goals_from_form as _exp_goals_from_form
from ..edge import build_why_from_form as _build_why_from_form

router = APIRouter(prefix="/explain", tags=["explain"])

# ---- small normalizer so FE can pass HOMEWIN / BTTSYES etc. ----
def _normalize_market(m: str) -> str:
    if not m:
        return ""
    x = m.strip().upper().replace(" ", "").replace("-", "")
    synonyms = {
        # BTTS
        "BTTSYES": "BTTS_Y", "BTTSY": "BTTS_Y", "BOTHTEAMSTOSCOREYES": "BTTS_Y",
        "BTTSNO": "BTTS_N",  "BTTSN": "BTTS_N", "BOTHTEAMSTOSCORENO":  "BTTS_N",
        # 1X2
        "HOMEWIN": "HOME_WIN",
        "AWAYWIN": "AWAY_WIN",
        # (DRAW stays DRAW)
    }
    # Already canonical like O2.5 / U2.5?
    if x.startswith("O") and x[1:].replace(".", "", 1).isdigit():
        return x
    if x.startswith("U") and x[1:].replace(".", "", 1).isdigit():
        return x
    return synonyms.get(x, x)


@router.get("/probability")
def explain_probability(
    fixture_id: int = Query(...),
    market: str = Query(...),  # e.g. "O2.5", "HOMEWIN", "BTTSYES"
    db: Session = Depends(get_db),
):
    fixture = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fixture:
        return {"error": "Fixture not found"}

    norm_market = _normalize_market(market)

    # ---------- Hybrid form (API-backed with DB fallback) ----------
    form_payload = get_hybrid_form_for_fixture(db, fixture, n=5, comp_scope=True)
    home_sum = form_payload["home"]["summary"]
    away_sum = form_payload["away"]["summary"]
    home_recent = form_payload["home"]["recent"]
    away_recent = form_payload["away"]["recent"]

    # Safety fallback if hybrid had no rows at all
    if not home_recent and not away_recent:
        home_sum = get_recent_form(db, fixture.home_team, fixture.kickoff_utc)
        away_sum = get_recent_form(db, fixture.away_team, fixture.kickoff_utc)
        home_recent = get_recent_fixtures(db, fixture.home_team, fixture.kickoff_utc, n=5)
        away_recent = get_recent_fixtures(db, fixture.away_team, fixture.kickoff_utc, n=5)

    # Strengths (kept — gives league-relative context)
    home_strength = get_team_strength(fixture.home_team, fixture.comp, db)
    away_strength = get_team_strength(fixture.away_team, fixture.comp, db)
    strength_delta = home_strength - away_strength

    # Latest model probability & fair price for THIS market
    prob_row = (
        db.query(ModelProb.prob)
        .filter(ModelProb.fixture_id == fixture.id, ModelProb.market == norm_market)
        .order_by(ModelProb.as_of.desc())
        .first()
    )
    prob = float(prob_row[0]) if prob_row else None
    fair_odds = (1 / prob) if prob and prob > 0 else None

    # Human readable WHY from hybrid form
    why_text = _build_why_from_form(fixture, form_payload)
    exp_goals = _exp_goals_from_form(home_sum, away_sum)  # {"home": x, "away": y, "total": z}

    explanation = {
        "fixture": f"{fixture.home_team} vs {fixture.away_team}",
        "market": norm_market,
        "home_strength": round(home_strength, 3),
        "away_strength": round(away_strength, 3),
        "strength_delta": round(strength_delta, 3),
        "explanation": why_text,                  # ✅ new: paragraph based on hybrid form
        "form": {
            "home": home_sum,                     # includes GFpg/GApg, W-D-L, etc.
            "away": away_sum,
            "expected_goals": exp_goals,         # ✅ blended EG from form
        },
        "notes": [],
    }

    if prob is not None:
        explanation["model_probability"] = round(prob * 100, 1)
        explanation["confidence"] = confidence_from_prob(prob)
        if fair_odds:
            explanation["fair_price"] = round(fair_odds, 2)

    # -------------------
    # Over/Under
    # -------------------
    if norm_market.startswith(("O", "U")):
        # Use our blended expected goals straight from form
        explanation["avg_total_goals"] = exp_goals["total"]
        explanation["notes"].append(
            f"Form-blended expected goals ≈ {exp_goals['home']:.2f} + {exp_goals['away']:.2f} = {exp_goals['total']:.2f}."
        )

        try:
            line = float(norm_market[1:])
        except Exception:
            line = 2.5

        if exp_goals["total"] > line + 0.3:
            explanation["notes"].append("Form leans high-scoring → Overs more plausible.")
        elif exp_goals["total"] < line - 0.3:
            explanation["notes"].append("Form leans low-scoring → Unders more plausible.")
        else:
            explanation["notes"].append("Form sits close to the line → marginal edge.")

        if prob and fair_odds:
            side = "Over" if norm_market.startswith("O") else "Under"
            explanation["recommendation"] = f"Bet {side} only if odds > {fair_odds:.2f}"

    # -------------------
       # -------------------
    # BTTS
    # -------------------
    elif market in ["BTTS_Y", "BTTS_N"]:
        # Use the exact same data path as the form widget (API-first with DB fallback),
        # and scope to the fixture's competition to keep things consistent.
        hybrid = get_hybrid_form_for_fixture(db, fixture, n=5, comp_scope=True)

        recent_home = hybrid.get("home", {}).get("recent", []) or []
        recent_away = hybrid.get("away", {}).get("recent", []) or []

        home_played = len(recent_home)
        away_played = len(recent_away)

        home_scored   = sum(1 for m in recent_home if (m.get("goals_for") or 0) > 0)
        away_scored   = sum(1 for m in recent_away if (m.get("goals_for") or 0) > 0)
        home_conceded = sum(1 for m in recent_home if (m.get("goals_against") or 0) > 0)
        away_conceded = sum(1 for m in recent_away if (m.get("goals_against") or 0) > 0)

        explanation["notes"].append(f"{fixture.home_team} scored in {home_scored}/{home_played} games.")
        explanation["notes"].append(f"{fixture.away_team} scored in {away_scored}/{away_played} games.")
        explanation["notes"].append(f"{fixture.home_team} conceded in {home_conceded}/{home_played} games.")
        explanation["notes"].append(f"{fixture.away_team} conceded in {away_conceded}/{away_played} games.")

        # Simple qualitative read:
        if home_scored and away_scored and home_conceded and away_conceded:
            explanation["notes"].append("Both teams frequently score and concede → BTTS Yes more plausible.")
        else:
            explanation["notes"].append("At least one team is inconsistent (scoring or conceding) → BTTS No more plausible.")

        # If the model prob was present at top, we already added fair_price & confidence.

    # -------------------
    # 1X2 (Win/Draw/Loss)
    # -------------------
    elif norm_market in {"HOME_WIN", "DRAW", "AWAY_WIN"}:
        explanation["notes"].append(
            f"Recent form (Home): {home_sum['wins']}W–{home_sum['draws']}D–{home_sum['losses']}L"
        )
        explanation["notes"].append(
            f"Recent form (Away): {away_sum['wins']}W–{away_sum['draws']}D–{away_sum['losses']}L"
        )

        if strength_delta > 0:
            explanation["notes"].append(f"League strength tilts toward {fixture.home_team}.")
        elif strength_delta < 0:
            explanation["notes"].append(f"League strength tilts toward {fixture.away_team}.")
        else:
            explanation["notes"].append("Teams are evenly matched by league strength.")

        # Show full distribution if available
        probs_1x2 = {}
        for mkt in ["HOME_WIN", "DRAW", "AWAY_WIN"]:
            row = (
                db.query(ModelProb.prob)
                .filter(ModelProb.fixture_id == fixture.id, ModelProb.market == mkt)
                .order_by(ModelProb.as_of.desc())
                .first()
            )
            if row:
                probs_1x2[mkt] = float(row[0])

        if len(probs_1x2) == 3:
            ph, pd, pa = probs_1x2["HOME_WIN"], probs_1x2["DRAW"], probs_1x2["AWAY_WIN"]
            explanation["notes"].append(
                f"Model distribution: Home {ph*100:.1f}%, Draw {pd*100:.1f}%, Away {pa*100:.1f}%."
            )
            # Market-specific fair
            target = {"HOME_WIN": ph, "DRAW": pd, "AWAY_WIN": pa}[norm_market]
            if target and target > 0:
                explanation["fair_price"] = round(1 / target, 2)
                explanation["recommendation"] = f"Bet {norm_market} only if odds > {explanation['fair_price']:.2f}"

    # -------------------
    # Double Chance
    # -------------------
    elif norm_market in {"1X", "12", "X2"}:
        probs_1x2 = {}
        for mkt in ["HOME_WIN", "DRAW", "AWAY_WIN"]:
            row = (
                db.query(ModelProb.prob)
                .filter(ModelProb.fixture_id == fixture.id, ModelProb.market == mkt)
                .order_by(ModelProb.as_of.desc())
                .first()
            )
            if row:
                probs_1x2[mkt] = float(row[0])

        if len(probs_1x2) == 3:
            ph, pd, pa = probs_1x2["HOME_WIN"], probs_1x2["DRAW"], probs_1x2["AWAY_WIN"]
            explanation["notes"].append(
                f"Model predicts: Home {ph*100:.1f}%, Draw {pd*100:.1f}%, Away {pa*100:.1f}%."
            )

            if norm_market == "1X":
                dc_prob = ph + pd
                explanation["notes"].append("Double Chance 1X = Home Win + Draw.")
            elif norm_market == "12":
                dc_prob = ph + pa
                explanation["notes"].append("Double Chance 12 = Home Win + Away Win.")
            else:  # X2
                dc_prob = pd + pa
                explanation["notes"].append("Double Chance X2 = Draw + Away Win.")

            explanation["notes"].append(f"Combined probability = {dc_prob*100:.1f}%.")
            if dc_prob:
                explanation["fair_price"] = round(1 / dc_prob, 2)
                explanation["recommendation"] = f"Bet {norm_market} only if odds > {explanation['fair_price']:.2f}"

    # Always include league-strength context
    explanation["notes"].append(
        "Strength values are league-relative: 1.0 = top of league, 0.05 = bottom."
    )

    return explanation