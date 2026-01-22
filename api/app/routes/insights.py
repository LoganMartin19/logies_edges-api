# api/app/routes/insights.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_, and_
from datetime import datetime, timezone

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
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


def _i(x, default=None):
    if x is None:
        return default
    try:
        return int(x)
    except Exception:
        return default


def _wdl_from_fixtures(team: str, fixtures: list[Fixture]):
    w = d = l = 0
    for fx in fixtures:
        if fx.full_time_home is None or fx.full_time_away is None:
            continue
        is_home = (fx.home_team == team)
        gf = fx.full_time_home if is_home else fx.full_time_away
        ga = fx.full_time_away if is_home else fx.full_time_home
        if gf > ga:
            w += 1
        elif gf == ga:
            d += 1
        else:
            l += 1
    return {"w": w, "d": d, "l": l}


def _team_last_n_fixtures(
    db: Session,
    team: str,
    comp: str | None,
    before_dt: datetime | None,
    n: int = 5,
):
    """
    Pull last N settled fixtures for a team. Uses Fixture table only.
    If comp is provided, restrict to that comp.
    Only fixtures with a recorded full-time score are included.
    """
    q = db.query(Fixture).filter(
        or_(Fixture.home_team == team, Fixture.away_team == team),
        Fixture.full_time_home.isnot(None),
        Fixture.full_time_away.isnot(None),
    )

    if comp:
        q = q.filter(Fixture.comp == comp)

    if before_dt:
        q = q.filter(Fixture.kickoff_utc < before_dt)

    return (
        q.order_by(desc(Fixture.kickoff_utc))
         .limit(n)
         .all()
    )


def _compute_team_stats_from_fixtures(team: str, fixtures: list[Fixture]):
    """
    Build last-N stats:
    - BTTS Yes count
    - Over 2.5 count
    - GF/GA totals + averages
    - W/D/L
    """
    played = 0
    gf_total = 0
    ga_total = 0
    btts_yes = 0
    over_2_5 = 0

    for fx in fixtures:
        if fx.full_time_home is None or fx.full_time_away is None:
            continue

        played += 1
        is_home = (fx.home_team == team)
        gf = fx.full_time_home if is_home else fx.full_time_away
        ga = fx.full_time_away if is_home else fx.full_time_home

        gf_total += gf
        ga_total += ga

        # BTTS in match (both teams score at least 1)
        if fx.full_time_home > 0 and fx.full_time_away > 0:
            btts_yes += 1

        # Over 2.5
        if (fx.full_time_home + fx.full_time_away) >= 3:
            over_2_5 += 1

    if played == 0:
        return None

    wdl = _wdl_from_fixtures(team, fixtures)

    return {
        "played": played,
        "gf": gf_total,
        "ga": ga_total,
        "avg_gf": round(gf_total / played, 2),
        "avg_ga": round(ga_total / played, 2),
        "btts_yes": btts_yes,
        "btts_yes_rate": round(btts_yes / played, 3),
        "over_2_5": over_2_5,
        "over_2_5_rate": round(over_2_5 / played, 3),
        "wdl": wdl,
    }


def mk_blurb(market: str, stats_home: dict | None, stats_away: dict | None, p: float) -> str:
    """
    Rich blurbs powered by *stats*, not the hybrid form (because yours can be empty).
    """
    pct = f"{p*100:.0f}%"

    if not stats_home or not stats_away:
        # clean apostrophe to avoid CSBâ€™s issue
        if market in ("BTTS_Y", "BTTS_N"):
            return f"CSB form model prices this at ~{pct}."
        if market in ("O2.5", "U2.5"):
            return f"CSB goals model prices this at ~{pct}."
        return f"CSB model prices this at ~{pct}."

    hp = stats_home["played"]
    ap = stats_away["played"]

    hg = stats_home["avg_gf"]
    hga = stats_home["avg_ga"]
    ag = stats_away["avg_gf"]
    aga = stats_away["avg_ga"]

    hbtts = stats_home["btts_yes"]
    abtts = stats_away["btts_yes"]
    hover = stats_home["over_2_5"]
    aover = stats_away["over_2_5"]

    hw = stats_home["wdl"]["w"]; hd = stats_home["wdl"]["d"]; hl = stats_home["wdl"]["l"]
    aw = stats_away["wdl"]["w"]; ad = stats_away["wdl"]["d"]; al = stats_away["wdl"]["l"]

    if market == "BTTS_Y":
        return (
            f"Last {hp}/{ap}: BTTS hit {hbtts}/{hp} (home) and {abtts}/{ap} (away). "
            f"Scoring/conceding: GF {hg:.1f}/{ag:.1f}, GA {hga:.1f}/{aga:.1f}. "
            f"CSB prices BTTS Yes at ~{pct}."
        )

    if market == "BTTS_N":
        return (
            f"Last {hp}/{ap}: BTTS missed {hp-hbtts}/{hp} (home) and {ap-abtts}/{ap} (away). "
            f"Defensive profile: GA {hga:.1f}/{aga:.1f}. "
            f"CSB prices BTTS No at ~{pct}."
        )

    if market == "O2.5":
        return (
            f"Last {hp}/{ap}: Over 2.5 hit {hover}/{hp} (home) and {aover}/{ap} (away). "
            f"Goal env: GF {hg:.1f}+{ag:.1f} per game (combined). "
            f"CSB prices Over 2.5 at ~{pct}."
        )

    if market == "U2.5":
        return (
            f"Last {hp}/{ap}: Under 2.5 hit {hp-hover}/{hp} (home) and {ap-aover}/{ap} (away). "
            f"Goal env: GA {hga:.1f}+{aga:.1f} (combined conceded). "
            f"CSB prices Under 2.5 at ~{pct}."
        )

    if market == "HOME_WIN":
        return (
            f"Form last {hp}/{ap}: {hw}W-{hd}D-{hl}L (home) vs {aw}W-{ad}D-{al}L (away). "
            f"Goal diff trends: {hg-hga:+.1f} vs {ag-aga:+.1f}. "
            f"CSB prices Home Win at ~{pct}."
        )

    if market == "AWAY_WIN":
        return (
            f"Form last {hp}/{ap}: {aw}W-{ad}D-{al}L (away) vs {hw}W-{hd}D-{hl}L (home). "
            f"Goal diff trends: {ag-aga:+.1f} vs {hg-hga:+.1f}. "
            f"CSB prices Away Win at ~{pct}."
        )

    if market == "DRAW":
        return (
            f"Profiles are close: GF {hg:.1f}/{ag:.1f}, GA {hga:.1f}/{aga:.1f}. "
            f"Recent form: {hw}W-{hd}D-{hl}L vs {aw}W-{ad}D-{al}L. "
            f"CSB prices Draw at ~{pct}."
        )

    if market == "1X":
        return f"1X covers Home or Draw. Home trend {hg-hga:+.1f} vs away {ag-aga:+.1f}. CSB prices 1X at ~{pct}."
    if market == "X2":
        return f"X2 covers Away or Draw. Away trend {ag-aga:+.1f} vs home {hg-hga:+.1f}. CSB prices X2 at ~{pct}."
    if market == "12":
        return f"12 excludes Draw. Recent goal profile suggests decisive outcomes are plausible. CSB prices 12 at ~{pct}."

    return f"CSB model prices this at ~{pct}."


@router.get("/fixtures/{fixture_id}/insights")
def fixture_insights(
    fixture_id: int,
    model: str = "team_form",  # maps to ModelProb.source
    n: int = 5,                # last-N window for stats
    db: Session = Depends(get_db),
):
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx:
        raise HTTPException(status_code=404, detail="Fixture not found")

    # 1) Try hybrid form first (may be empty)
    try:
        hybrid = get_hybrid_form_for_fixture(db, fx)
    except Exception:
        hybrid = None

    home_form = (hybrid or {}).get("home_form") or {}
    away_form = (hybrid or {}).get("away_form") or {}

    # 2) Compute stats from fixtures table as fallback (and also useful generally)
    before_dt = fx.kickoff_utc or datetime.now(timezone.utc)

    home_fx = _team_last_n_fixtures(db, fx.home_team, fx.comp, before_dt, n=n)
    away_fx = _team_last_n_fixtures(db, fx.away_team, fx.comp, before_dt, n=n)

    stats_home = _compute_team_stats_from_fixtures(fx.home_team, home_fx)
    stats_away = _compute_team_stats_from_fixtures(fx.away_team, away_fx)

    # If hybrid form is empty, fill it using computed stats so UI can still show "form"
    if not home_form and stats_home:
        home_form = {
            "avg_gf": stats_home["avg_gf"],
            "avg_ga": stats_home["avg_ga"],
            "wins": stats_home["wdl"]["w"],
            "draws": stats_home["wdl"]["d"],
            "losses": stats_home["wdl"]["l"],
        }
    if not away_form and stats_away:
        away_form = {
            "avg_gf": stats_away["avg_gf"],
            "avg_ga": stats_away["avg_ga"],
            "wins": stats_away["wdl"]["w"],
            "draws": stats_away["wdl"]["d"],
            "losses": stats_away["wdl"]["l"],
        }

    # 3) Pull latest probs for this model source
    rows = (
        db.query(ModelProb)
        .filter(ModelProb.fixture_id == fixture_id, ModelProb.source == model)
        .order_by(desc(ModelProb.as_of))
        .all()
    )

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
        insights.append({
            "market": m,
            "prob": round(p, 4),
            "fair_odds": fair_odds(p),
            "confidence": confidence_from_prob(p),
            "blurb": mk_blurb(m, stats_home, stats_away, p),
        })

    return {
        "fixture_id": fixture_id,
        "model": model,
        "n": n,
        "insights": insights,
        "form": {"home": home_form, "away": away_form},
        "stats": {"home": stats_home, "away": stats_away},
    }