# api/app/routes/insights.py

from fastapi import APIRouter, Depends, HTTPException, Query
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


def _parse_score_obj(obj):
    """
    Accepts a recent fixture row:
      - {"goals_for":2,"goals_against":1}
      - {"gf":2,"ga":1}
      - {"score":"2-1"}
    Returns (gf, ga) or (None,None)
    """
    if obj is None or not isinstance(obj, dict):
        return None, None

    s = obj.get("score")
    if isinstance(s, str) and "-" in s:
        try:
            a, b = s.split("-", 1)
            return int(a.strip()), int(b.strip())
        except Exception:
            return None, None

    for k1, k2 in [("goals_for", "goals_against"), ("gf", "ga")]:
        if k1 in obj and k2 in obj:
            try:
                return int(obj.get(k1)), int(obj.get(k2))
            except Exception:
                return None, None

    return None, None


def _compute_stats_from_recent(recent_rows):
    """
    Compute counts from a list of recent rows with score fields.
    Returns dict: {n, btts, o25, u25} or None
    """
    if not isinstance(recent_rows, list) or not recent_rows:
        return None

    btts = 0
    o25 = 0
    u25 = 0
    n = 0

    for r in recent_rows:
        gf, ga = _parse_score_obj(r)
        if gf is None or ga is None:
            continue

        n += 1
        if gf > 0 and ga > 0:
            btts += 1
        if (gf + ga) >= 3:
            o25 += 1
        else:
            u25 += 1

    if n == 0:
        return None

    return {"n": n, "btts": btts, "o25": o25, "u25": u25}


def _parse_csv_ints(v):
    """
    Parses '2,0,1,3,2' -> [2,0,1,3,2]
    Accepts list already -> list[int]
    """
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


def _compute_stats_from_last5_strings(form: dict):
    """
    Uses TeamForm-style fields if present:
      - last_5_goals_for:  "2,0,1,3,2"
      - last_5_goals_against: "1,1,0,1,0"
    Returns dict: {n, btts, o25, u25} or None
    """
    gf = _parse_csv_ints(
        form.get("last_5_goals_for")
        or form.get("last5_goals_for")
        or form.get("goals_for_last5")
    )
    ga = _parse_csv_ints(
        form.get("last_5_goals_against")
        or form.get("last5_goals_against")
        or form.get("goals_against_last5")
    )

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


def _team_stats(form: dict):
    """
    Try multiple ways to get stats:
      1) form["recent"] list (preferred)
      2) last_5_goals_for/against strings
    """
    recent = form.get("recent") or form.get("home_recent") or form.get("away_recent")
    s = _compute_stats_from_recent(recent)
    if s:
        return s

    s2 = _compute_stats_from_last5_strings(form)
    if s2:
        return s2

    return None


def _stat_line(market: str, home_form: dict, away_form: dict):
    hs = _team_stats(home_form)
    a_s = _team_stats(away_form)

    if hs is None and a_s is None:
        return None

    if market in ("BTTS_Y", "BTTS_N"):
        if hs and a_s:
            return f"Stat: BTTS landed {hs['btts']}/{hs['n']} (home), {a_s['btts']}/{a_s['n']} (away)."
        if hs:
            return f"Stat: BTTS landed {hs['btts']}/{hs['n']} in home’s last {hs['n']}."
        if a_s:
            return f"Stat: BTTS landed {a_s['btts']}/{a_s['n']} in away’s last {a_s['n']}."

    if market == "O2.5":
        if hs and a_s:
            return f"Stat: Over 2.5 hit {hs['o25']}/{hs['n']} (home), {a_s['o25']}/{a_s['n']} (away)."
        if hs:
            return f"Stat: Over 2.5 hit {hs['o25']}/{hs['n']} in home’s last {hs['n']}."
        if a_s:
            return f"Stat: Over 2.5 hit {a_s['o25']}/{a_s['n']} in away’s last {a_s['n']}."

    if market == "U2.5":
        if hs and a_s:
            return f"Stat: Under 2.5 hit {hs['u25']}/{hs['n']} (home), {a_s['u25']}/{a_s['n']} (away)."
        if hs:
            return f"Stat: Under 2.5 hit {hs['u25']}/{hs['n']} in home’s last {hs['n']}."
        if a_s:
            return f"Stat: Under 2.5 hit {a_s['u25']}/{a_s['n']} in away’s last {a_s['n']}."

    return None


def mk_blurb(market: str, home_form: dict, away_form: dict, p: float) -> str:
    hg = _f(home_form.get("avg_gf"))
    hga = _f(home_form.get("avg_ga"))
    ag = _f(away_form.get("avg_gf"))
    aga = _f(away_form.get("avg_ga"))

    hw = _wdl(home_form)
    aw = _wdl(away_form)

    wdl_home = f"{hw[0]}W-{hw[1]}D-{hw[2]}L" if hw else None
    wdl_away = f"{aw[0]}W-{aw[1]}D-{aw[2]}L" if aw else None

    stat = _stat_line(market, home_form, away_form)
    pct = f"{p*100:.0f}%"

    # fallback (no avg_gf/avg_ga)
    if hg is None or hga is None or ag is None or aga is None:
        if market in ("BTTS_Y", "BTTS_N"):
            base = f"Fair price is derived from CSB's form model. CSB prices this at ~{pct}."
        elif market.startswith("O") or market.startswith("U"):
            base = f"Fair price is derived from CSB's goals model. CSB prices this at ~{pct}."
        else:
            base = f"Fair price is derived from CSB's form-based model probabilities (~{pct})."
        return f"{base} {stat}" if stat else base

    total_gf = hg + ag
    total_ga = hga + aga
    home_goal_diff = hg - hga
    away_goal_diff = ag - aga

    if market == "BTTS_Y":
        bits = [
            f"Both sides are active at both ends: scoring {hg:.1f}/{ag:.1f} gpg and conceding {hga:.1f}/{aga:.1f}.",
        ]
        if wdl_home and wdl_away:
            bits.append(f"Form: {wdl_home} vs {wdl_away}.")
        bits.append(f"CSB prices BTTS Yes at ~{pct}.")
        if stat:
            bits.append(stat)
        return " ".join(bits)

    if market == "BTTS_N":
        bits = [
            f"At least one side profiles as containable: GF {hg:.1f}/{ag:.1f}, GA {hga:.1f}/{aga:.1f}.",
        ]
        if wdl_home and wdl_away:
            bits.append(f"Form: {wdl_home} vs {wdl_away}.")
        bits.append(f"CSB prices BTTS No at ~{pct}.")
        if stat:
            bits.append(stat)
        return " ".join(bits)

    if market == "O2.5":
        bits = [
            f"Goals environment looks high (combined GF {total_gf:.1f}, combined GA {total_ga:.1f}).",
            "That usually correlates with more chances + higher totals.",
            f"CSB prices Over 2.5 at ~{pct}.",
        ]
        if stat:
            bits.append(stat)
        return " ".join(bits)

    if market == "U2.5":
        bits = [
            f"Goals environment leans lower (combined GF {total_gf:.1f}, combined GA {total_ga:.1f}).",
            "This often means fewer clear chances and a tighter game state.",
            f"CSB prices Under 2.5 at ~{pct}.",
        ]
        if stat:
            bits.append(stat)
        return " ".join(bits)

    if market == "HOME_WIN":
        bits = [
            f"Home goal trend {home_goal_diff:+.1f} (GF {hg:.1f}, GA {hga:.1f}) vs away {away_goal_diff:+.1f} (GF {ag:.1f}, GA {aga:.1f})."
        ]
        if wdl_home and wdl_away:
            bits.append(f"Form: {wdl_home} vs {wdl_away}.")
        bits.append(f"CSB prices Home Win at ~{pct}.")
        return " ".join(bits)

    if market == "AWAY_WIN":
        bits = [
            f"Away goal trend {away_goal_diff:+.1f} (GF {ag:.1f}, GA {aga:.1f}) vs home {home_goal_diff:+.1f} (GF {hg:.1f}, GA {hga:.1f})."
        ]
        if wdl_home and wdl_away:
            bits.append(f"Form: {wdl_home} vs {wdl_away}.")
        bits.append(f"CSB prices Away Win at ~{pct}.")
        return " ".join(bits)

    if market == "DRAW":
        bits = [
            f"Profiles are fairly close (GF {hg:.1f}/{ag:.1f}, GA {hga:.1f}/{aga:.1f}).",
            "That tends to lift draw probability in the model.",
        ]
        if wdl_home and wdl_away:
            bits.append(f"Form: {wdl_home} vs {wdl_away}.")
        bits.append(f"CSB prices Draw at ~{pct}.")
        return " ".join(bits)

    if market == "1X":
        return (
            f"Double Chance covers Home or Draw. "
            f"Home trend {home_goal_diff:+.1f} vs away {away_goal_diff:+.1f} "
            f"(GF {hg:.1f}/{ag:.1f}, GA {hga:.1f}/{aga:.1f}). "
            f"CSB prices 1X at ~{pct}."
        )

    if market == "X2":
        return (
            f"Double Chance covers Away or Draw. "
            f"Away trend {away_goal_diff:+.1f} vs home {home_goal_diff:+.1f} "
            f"(GF {ag:.1f}/{hg:.1f}, GA {aga:.1f}/{hga:.1f}). "
            f"CSB prices X2 at ~{pct}."
        )

    if market == "12":
        return (
            f"Double Chance excludes Draw (either team wins). "
            f"Combined scoring/conceding suggests decisive outcomes are plausible "
            f"(GF {total_gf:.1f}, GA {total_ga:.1f}). "
            f"CSB prices 12 at ~{pct}."
        )

    base = f"CSB fair odds are based on model probability of ~{pct}."
    return f"{base} {stat}" if stat else base


@router.get("/fixtures/{fixture_id}/insights")
def fixture_insights(
    fixture_id: int,
    model: str = "team_form",  # maps to ModelProb.source
    n: int = Query(5, ge=1, le=10),
    include_recent: bool = False,
    db: Session = Depends(get_db),
):
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx:
        raise HTTPException(status_code=404, detail="Fixture not found")

    try:
        hybrid = get_hybrid_form_for_fixture(db, fx, n=n)
    except Exception:
        hybrid = None

    # ✅ CORRECT KEYS from get_hybrid_form_for_fixture:
    # hybrid["home"]["summary"], hybrid["home"]["recent"]
    home_block = (hybrid or {}).get("home") or {}
    away_block = (hybrid or {}).get("away") or {}

    home_summary = home_block.get("summary") or {}
    away_summary = away_block.get("summary") or {}

    home_recent = home_block.get("recent") or []
    away_recent = away_block.get("recent") or []

    # Create "form" dicts your mk_blurb expects, and inject recent for stat calc
    home_form = dict(home_summary)
    away_form = dict(away_summary)
    home_form["recent"] = home_recent
    away_form["recent"] = away_recent

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
            "blurb": mk_blurb(m, home_form, away_form, p),
        })

    # Helpful explicit stats object (so you can see instantly if recent is empty)
    stats = {
        "home": _team_stats(home_form),
        "away": _team_stats(away_form),
    }

    out = {
        "fixture_id": fixture_id,
        "model": model,
        "n": n,
        "insights": insights,
        "form": {"home": home_summary, "away": away_summary},
        "stats": stats,
    }

    if include_recent:
        out["recent"] = {"home": home_recent, "away": away_recent}

    return out