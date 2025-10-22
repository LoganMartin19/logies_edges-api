# api/app/routers/form.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Fixture
from ..services.form import (
    get_recent_form,              # DB-only (kept for HTML page & summary)
    get_recent_fixtures,          # DB-only (kept for HTML page & summary)
    get_recent_form_api_first,    # ⚽️ DB + API top-up
    get_hybrid_form_for_fixture,  # ✅ singular
)

router = APIRouter(prefix="/form", tags=["form"])


# ------------------------------- Safe helpers -------------------------------

_DEFAULT_SUMMARY: Dict[str, float | int] = {
    "played": 0,
    "wins": 0,
    "draws": 0,
    "losses": 0,
    "avg_goals_for": 0.0,
    "avg_goals_against": 0.0,
    "goals_for": 0.0,
    "goals_against": 0.0,
    "goal_diff": 0.0,
}

def _norm_summary(raw: Optional[Dict[str, Any]]) -> Dict[str, float | int]:
    """
    Normalize any summary (or None) to a consistent dict so the frontend doesn't crash.
    Accepts either avg_* or totals; computes missing values when possible.
    """
    if not isinstance(raw, dict):
        return dict(_DEFAULT_SUMMARY)

    out = dict(_DEFAULT_SUMMARY)

    played = int(raw.get("played") or raw.get("matches") or 0)
    wins = int(raw.get("wins") or 0)
    draws = int(raw.get("draws") or 0)
    losses = int(raw.get("losses") or 0)

    gf_total = float(raw.get("goals_for") or raw.get("gf") or 0.0)
    ga_total = float(raw.get("goals_against") or raw.get("ga") or 0.0)

    gf_avg = raw.get("avg_goals_for")
    ga_avg = raw.get("avg_goals_against")

    if gf_avg is None:
        gf_avg = (gf_total / played) if played else 0.0
    if ga_avg is None:
        ga_avg = (ga_total / played) if played else 0.0

    out.update({
        "played": played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "avg_goals_for": float(gf_avg or 0.0),
        "avg_goals_against": float(ga_avg or 0.0),
        "goals_for": gf_total if gf_total else float(gf_avg or 0.0) * played,
        "goals_against": ga_total if ga_total else float(ga_avg or 0.0) * played,
    })
    out["goal_diff"] = float(out["goals_for"]) - float(out["goals_against"])
    return out

def _is_grid_or_ice(comp: Optional[str]) -> bool:
    cu = (comp or "").upper()
    return any(x in cu for x in ("NFL", "CFB", "NCAA", "AMERICAN", "NHL", "HOCKEY", "ICE"))


# --- JSON: hybrid form for a fixture (primary) -------------------------------

@router.get("/fixture/json")
@router.get("/form/fixture", include_in_schema=False)  # back-compat alias
def get_fixture_form_json(
    fixture_id: int = Query(...),
    n: int = Query(5, ge=1, le=20),
    all_comps: int = Query(0, description="0 = scope to fixture competition, 1 = all competitions"),
    db: Session = Depends(get_db),
):
    """
    Unified hybrid form endpoint:
      - ⚽️ Football → DB + API (get_recent_form_api_first)
      - 🏈/🏒 Gridiron/Hockey → hybrid (provider fetch if wired, else DB)
    Hardened: never subscripts None, always returns normalized summaries.
    """
    fixture = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fixture:
        return {"error": "Fixture not found"}

    comp_scope = None if all_comps else (fixture.comp or None)
    as_of = fixture.kickoff_utc or datetime.now(timezone.utc)
    in_grid_or_ice = _is_grid_or_ice(fixture.comp)

    if in_grid_or_ice:
        # Hybrid path (safe)
        form_data = get_hybrid_form_for_fixture(db, fixture, n=n, comp_scope=(comp_scope is not None)) or {}
        home_blk = (form_data.get("home") or {})
        away_blk = (form_data.get("away") or {})

        home_summary = _norm_summary(home_blk.get("summary"))
        away_summary = _norm_summary(away_blk.get("summary"))
        home_recent = list(home_blk.get("recent") or [])
        away_recent = list(away_blk.get("recent") or [])

        return {
            "fixture_id": fixture.id,
            "home_team": fixture.home_team,
            "away_team": fixture.away_team,
            "competition": fixture.comp,
            "scope": "all" if comp_scope is None else "competition",
            "home_form": home_summary,
            "away_form": away_summary,
            "home_recent": home_recent,
            "away_recent": away_recent,
            "n": n,
        }

    # ⚽️ Football: API-first with DB fallback; normalize summaries
    home = get_recent_form_api_first(
        db=db,
        team_name=fixture.home_team,
        team_provider_id=getattr(fixture, "provider_home_team_id", None),
        before=as_of,
        n=n,
        comp=comp_scope,
    ) or {}
    away = get_recent_form_api_first(
        db=db,
        team_name=fixture.away_team,
        team_provider_id=getattr(fixture, "provider_away_team_id", None),
        before=as_of,
        n=n,
        comp=comp_scope,
    ) or {}

    return {
        "fixture_id": fixture.id,
        "home_team": fixture.home_team,
        "away_team": fixture.away_team,
        "competition": fixture.comp,
        "scope": "all" if comp_scope is None else "competition",
        "home_form": _norm_summary(home.get("summary")),
        "away_form": _norm_summary(away.get("summary")),
        "home_recent": list(home.get("recent") or []),
        "away_recent": list(away.get("recent") or []),
        "n": n,
    }


# --- HTML preview (DB-only for speed/debug) ----------------------------------

@router.get("/fixture/html", response_class=HTMLResponse)
def get_fixture_form_html(
    fixture_id: int = Query(...),
    n: int = Query(5, ge=1, le=20),
    all_comps: int = Query(0, description="0 = scope to fixture competition, 1 = all competitions"),
    db: Session = Depends(get_db),
):
    """
    Simple HTML preview (DB-only for speed/debug).
    """
    fixture = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fixture:
        return HTMLResponse("<i>Fixture not found.</i>")

    as_of = fixture.kickoff_utc or datetime.now(timezone.utc)
    comp_scope = None if all_comps else (fixture.comp or None)

    # Note: keep original call signature expected by your services.form helpers
    home_form = get_recent_form(db, fixture.home_team, as_of, n=n, comp=comp_scope)
    away_form = get_recent_form(db, fixture.away_team, as_of, n=n, comp=comp_scope)
    home_recent = get_recent_fixtures(db, fixture.home_team, as_of, n=n, comp=comp_scope)
    away_recent = get_recent_fixtures(db, fixture.away_team, as_of, n=n, comp=comp_scope)

    def render_team_form(team_name, form, recent):
        rows = ""
        for match in (recent or []):
            color = {"win": "#c8e6c9", "draw": "#fff9c4", "loss": "#ffcdd2"}.get(match.get("result"), "#eee")
            home_away = "vs" if match.get("is_home") else "@"
            opponent = match.get("opponent", "-")
            score = match.get("score", "-")
            date = (match.get("date") or "")[:10]
            rows += f"""
                <tr style="background:{color};text-align:center">
                    <td>{date}</td>
                    <td>{home_away} {opponent}</td>
                    <td>{score}</td>
                    <td>{(match.get('result') or '').capitalize()}</td>
                </tr>
            """

        f = _norm_summary(form)
        summary = f"{int(f['wins'])}W – {int(f['draws'])}D – {int(f['losses'])}L"
        gf_ga = f"{int(f['goals_for'])} GF / {int(f['goals_against'])} GA"
        avgs = f"{float(f['avg_goals_for']):.1f} GFpg / {float(f['avg_goals_against']):.1f} GApg"
        scope_label = "All competitions" if comp_scope is None else (comp_scope or "Competition")

        return f"""
            <div style="margin-bottom:20px">
                <h4 style="margin:5px 0">{team_name} — last {n} ({scope_label}): {summary}</h4>
                <p style="margin:4px 0;color:#555">{gf_ga} — {avgs}</p>
                <table style="width:100%;border-collapse:collapse;font-size:13px">
                    <thead>
                        <tr style="background:#eee;text-align:center">
                            <th style="padding:6px">Date</th>
                            <th>Opponent</th>
                            <th>Score</th>
                            <th>Result</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
        """

    html = f"""
    <div class='form-breakdown' style='padding: 10px; background: #f9f9f9; border: 1px solid #ccc; margin-top: 6px'>
        {render_team_form(fixture.home_team, home_form, home_recent)}
        {render_team_form(fixture.away_team, away_form, away_recent)}
    </div>
    """
    return HTMLResponse(html)


# --- DB-only summary ----------------------------------------------------------

@router.get("/summary")
def get_team_summary(
    team: str = Query(...),
    as_of: datetime = Query(datetime.utcnow()),
    n: int = Query(5, ge=1, le=20),
    comp: str | None = Query(None, description="If provided, restrict to this competition name"),
    db: Session = Depends(get_db),
):
    """
    Small helper endpoint (DB-only) for quick summaries.
    """
    form = get_recent_form(db, team, as_of, n=n, comp=comp)
    recent = get_recent_fixtures(db, team, as_of, n=n, comp=comp)

    f = _norm_summary(form)

    return {
        "team": team,
        "as_of": as_of.isoformat(),
        "last_n": n,
        "scope": "all" if comp is None else comp,
        "summary": {
            "wins": int(f["wins"]),
            "draws": int(f["draws"]),
            "losses": int(f["losses"]),
            "goals_for": float(f["goals_for"]),
            "goals_against": float(f["goals_against"]),
            "goal_diff": float(f["goal_diff"]),
            "avg_goals_for": float(f["avg_goals_for"]),
            "avg_goals_against": float(f["avg_goals_against"]),
        },
        "recent_matches": recent or [],
    }


# --- Computed standings (any comp using settled fixtures) ---------------------

@router.get("/standings/computed")
def get_computed_standings(
    comp: str = Query(..., description="Competition name e.g. 'NHL' or 'NCAA'"),
    as_of: str | None = Query(None, description="Optional ISO date"),
    db: Session = Depends(get_db),
):
    """
    Generate computed standings for NHL / CFB (or any comp) using settled fixtures.
    """
    cutoff = datetime.fromisoformat(as_of) if as_of else datetime.now(timezone.utc)
    rows = (
        db.query(Fixture)
        .filter(Fixture.comp == comp)
        .filter(Fixture.result_settled == True)
        .filter(Fixture.kickoff_utc < cutoff)
        .all()
    )

    if not rows:
        return {"comp": comp, "table": []}

    table: Dict[str, Dict[str, Any]] = {}
    for f in rows:
        for side, is_home in [(f.home_team, True), (f.away_team, False)]:
            if side not in table:
                table[side] = {
                    "team": side, "played": 0, "wins": 0, "losses": 0, "draws": 0,
                    "gf": 0, "ga": 0
                }

            scored = f.full_time_home if is_home else f.full_time_away
            conceded = f.full_time_away if is_home else f.full_time_home

            table[side]["played"] += 1
            table[side]["gf"] += scored
            table[side]["ga"] += conceded
            if scored > conceded:
                table[side]["wins"] += 1
            elif scored == conceded:
                table[side]["draws"] += 1
            else:
                table[side]["losses"] += 1

    # Compute points
    for v in table.values():
        if comp.upper() == "NHL":
            v["points"] = v["wins"] * 2
        elif comp.upper() in ("CFB", "NCAA"):
            v["points"] = v["wins"]
        else:
            v["points"] = v["wins"] * 3 + v["draws"]
        v["gd"] = v["gf"] - v["ga"]

    sorted_table = sorted(
        table.values(),
        key=lambda x: (x["points"], x["gd"], x["gf"]),
        reverse=True,
    )
    for i, row in enumerate(sorted_table, start=1):
        row["position"] = i

    return {"comp": comp, "as_of": cutoff.isoformat(), "table": sorted_table}

@router.get("/_debug/hybrid")
def debug_hybrid(fixture_id: int, db: Session = Depends(get_db)):
    from ..services.form import get_hybrid_form_for_fixture
    from ..services import api_ice
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx:
        return {"error": "fixture not found"}
    ice_fixture = api_ice.get_fixture(int(getattr(fx, "provider_fixture_id", 0))) if getattr(fx, "provider_fixture_id", None) else None
    return {
        "fixture_row": {
            "id": fx.id,
            "comp": fx.comp,
            "sport": getattr(fx, "sport", None),
            "provider_fixture_id": getattr(fx, "provider_fixture_id", None),
            "provider_home_team_id": getattr(fx, "provider_home_team_id", None),
            "provider_away_team_id": getattr(fx, "provider_away_team_id", None),
        },
        "ice_get_fixture_snip": (ice_fixture or {}) if ice_fixture else None,
        "hybrid_result": get_hybrid_form_for_fixture(db, fx, n=5, comp_scope=True),
        "ice_last_http": getattr(api_ice, "LAST_HTTP", None),
    }