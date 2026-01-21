# api/app/routes/insights.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc

from ..db import get_db
from ..models import Fixture, ModelProb, TeamForm
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


def _wdl(form: dict):
    w = _i(form.get("wins") or form.get("w") or form.get("form_wins"), 0)
    d = _i(form.get("draws") or form.get("d") or form.get("form_draws"), 0)
    l = _i(form.get("losses") or form.get("l") or form.get("form_losses"), 0)
    if w == 0 and d == 0 and l == 0:
        return None
    return w, d, l


def _parse_csv_ints(v):
    if v is None:
        return None
    if isinstance(v, list):
        out = []
        for x in v:
            try:
                out.append(int(x))
            except Exception:
                return None
        return out

    if not isinstance(v, str):
        return None

    parts = [p.strip() for p in v.split(",") if p.strip() != ""]
    if not parts:
        return None

    out = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            return None
    return out


def _compute_last5_stats_from_strings(last_5_gf: str | None, last_5_ga: str | None):
    gf = _parse_csv_ints(last_5_gf)
    ga = _parse_csv_ints(last_5_ga)
    if not gf or not ga:
        return None

    n = min(len(gf), len(ga), 5)
    if n <= 0:
        return None

    btts = 0
    o25 = 0
    u25 = 0
    for i in range(n):
        gfi = gf[i]
        gai = ga[i]
        if gfi > 0 and gai > 0:
            btts += 1
        if (gfi + gai) >= 3:
            o25 += 1
        else:
            u25 += 1

    return {"n": n, "btts": btts, "o25": o25, "u25": u25}


def _team_stats_from_formdict(form: dict):
    """
    Try to compute stats from whatever the hybrid form dict contains.
    Supports common keys; if none exist, returns None.
    """
    gf = (
        form.get("last_5_goals_for")
        or form.get("last5_goals_for")
        or form.get("goals_for_last5")
    )
    ga = (
        form.get("last_5_goals_against")
        or form.get("last5_goals_against")
        or form.get("goals_against_last5")
    )
    return _compute_last5_stats_from_strings(gf, ga)


def _team_stats_from_db(db: Session, team: str, comp: str | None):
    """
    Guaranteed fallback: use TeamForm row from DB.
    """
    q = db.query(TeamForm).filter(TeamForm.team == team)
    if comp:
        q = q.filter(TeamForm.comp == comp)

    row = q.order_by(desc(TeamForm.updated_at)).first()
    if not row:
        return None

    return _compute_last5_stats_from_strings(row.last_5_goals_for, row.last_5_goals_against)


def _stat_line(market: str, home_s: dict | None, away_s: dict | None):
    if not home_s and not away_s:
        return None

    def fmt_pair(label, key):
        if home_s and away_s:
            return f"{label}: {home_s[key]}/{home_s['n']} (home), {away_s[key]}/{away_s['n']} (away)."
        if home_s:
            return f"{label}: {home_s[key]}/{home_s['n']} in home’s last {home_s['n']}."
        return f"{label}: {away_s[key]}/{away_s['n']} in away’s last {away_s['n']}."

    if market in ("BTTS_Y", "BTTS_N"):
        return "Stat: " + fmt_pair("BTTS hit", "btts")
    if market == "O2.5":
        return "Stat: " + fmt_pair("Over 2.5 hit", "o25")
    if market == "U2.5":
        return "Stat: " + fmt_pair("Under 2.5 hit", "u25")

    return None


def mk_blurb(market: str, home_form: dict, away_form: dict, p: float, stat_line: str | None) -> str:
    hg = _f(home_form.get("avg_gf"))
    hga = _f(home_form.get("avg_ga"))
    ag = _f(away_form.get("avg_gf"))
    aga = _f(away_form.get("avg_ga"))

    hw = _wdl(home_form)
    aw = _wdl(away_form)
    wdl_home = f"{hw[0]}W-{hw[1]}D-{hw[2]}L" if hw else None
    wdl_away = f"{aw[0]}W-{aw[1]}D-{aw[2]}L" if aw else None

    pct = f"{p*100:.0f}%"

    # fallback if averages missing
    if hg is None or hga is None or ag is None or aga is None:
        if market in ("BTTS_Y", "BTTS_N"):
            base = f"CSB’s form model prices this at ~{pct}."
        elif market.startswith("O") or market.startswith("U"):
            base = f"CSB’s goals model prices this at ~{pct}."
        else:
            base = f"CSB’s model prices this at ~{pct}."
        return f"{base} {stat_line}" if stat_line else base

    total_gf = hg + ag
    total_ga = hga + aga
    home_diff = hg - hga
    away_diff = ag - aga

    if market == "BTTS_Y":
        base = (
            f"Both sides are active at both ends: score {hg:.1f}/{ag:.1f} gpg, concede {hga:.1f}/{aga:.1f}. "
            f"{('Form: ' + wdl_home + ' vs ' + wdl_away + '. ') if (wdl_home and wdl_away) else ''}"
            f"CSB prices BTTS Yes at ~{pct}."
        )
        return f"{base} {stat_line}" if stat_line else base

    if market == "BTTS_N":
        base = (
            f"At least one side profiles as containable: GF {hg:.1f}/{ag:.1f}, GA {hga:.1f}/{aga:.1f}. "
            f"{('Form: ' + wdl_home + ' vs ' + wdl_away + '. ') if (wdl_home and wdl_away) else ''}"
            f"CSB prices BTTS No at ~{pct}."
        )
        return f"{base} {stat_line}" if stat_line else base

    if market == "O2.5":
        base = (
            f"Goals environment looks high (combined GF {total_gf:.1f}, combined GA {total_ga:.1f}); "
            f"that often correlates with higher totals. CSB prices Over 2.5 at ~{pct}."
        )
        return f"{base} {stat_line}" if stat_line else base

    if market == "U2.5":
        base = (
            f"Goals environment leans lower (combined GF {total_gf:.1f}, combined GA {total_ga:.1f}); "
            f"often fewer clear chances. CSB prices Under 2.5 at ~{pct}."
        )
        return f"{base} {stat_line}" if stat_line else base

    if market == "HOME_WIN":
        base = (
            f"Home trend {home_diff:+.1f} (GF {hg:.1f}, GA {hga:.1f}) vs away {away_diff:+.1f} (GF {ag:.1f}, GA {aga:.1f}). "
            f"{('Form: ' + wdl_home + ' vs ' + wdl_away + '. ') if (wdl_home and wdl_away) else ''}"
            f"CSB prices Home Win at ~{pct}."
        )
        return base

    if market == "AWAY_WIN":
        base = (
            f"Away trend {away_diff:+.1f} (GF {ag:.1f}, GA {aga:.1f}) vs home {home_diff:+.1f} (GF {hg:.1f}, GA {hga:.1f}). "
            f"{('Form: ' + wdl_home + ' vs ' + wdl_away + '. ') if (wdl_home and wdl_away) else ''}"
            f"CSB prices Away Win at ~{pct}."
        )
        return base

    if market == "DRAW":
        base = (
            f"Profiles are fairly close (GF {hg:.1f}/{ag:.1f}, GA {hga:.1f}/{aga:.1f}). "
            f"{('Form: ' + wdl_home + ' vs ' + wdl_away + '. ') if (wdl_home and wdl_away) else ''}"
            f"CSB prices Draw at ~{pct}."
        )
        return base

    return f"CSB’s model prices this at ~{pct}."


@router.get("/fixtures/{fixture_id}/insights")
def fixture_insights(
    fixture_id: int,
    model: str = "team_form",
    db: Session = Depends(get_db),
):
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx:
        raise HTTPException(status_code=404, detail="Fixture not found")

    # hybrid forms (best effort)
    try:
        hybrid = get_hybrid_form_for_fixture(db, fx)
    except Exception:
        hybrid = None

    home_form = (hybrid or {}).get("home_form") or {}
    away_form = (hybrid or {}).get("away_form") or {}

    # --- Guaranteed stats: try hybrid -> DB fallback ---
    home_stats = _team_stats_from_formdict(home_form) or _team_stats_from_db(db, fx.home_team, fx.comp)
    away_stats = _team_stats_from_formdict(away_form) or _team_stats_from_db(db, fx.away_team, fx.comp)

    # probs
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

        sline = _stat_line(m, home_stats, away_stats)

        insights.append(
            {
                "market": m,
                "prob": round(p, 4),
                "fair_odds": fair_odds(p),
                "confidence": confidence_from_prob(p),
                "blurb": mk_blurb(m, home_form, away_form, p, sline),
            }
        )

    return {
        "fixture_id": fixture_id,
        "model": model,
        "insights": insights,
        "form": {"home": home_form, "away": away_form},
        "stats": {
            "home": home_stats,
            "away": away_stats,
        },
    }