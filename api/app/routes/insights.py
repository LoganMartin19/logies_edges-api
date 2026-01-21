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
    if w == 0 and d == 0 and l == 0:
        return None
    return w, d, l


def _parse_score_obj(obj):
    """
    Accepts a 'recent fixture' row from hybrid form (you might have):
      - {"goals_for":2,"goals_against":1}
      - {"gf":2,"ga":1}
      - {"score":"2-1"}
    Returns (gf, ga) or (None,None)
    """
    if obj is None:
        return None, None

    # score string
    s = None
    if isinstance(obj, dict):
        s = obj.get("score")

    if isinstance(s, str) and "-" in s:
        try:
            a, b = s.split("-", 1)
            return int(a.strip()), int(b.strip())
        except Exception:
            return None, None

    if isinstance(obj, dict):
        for k1, k2 in [("goals_for", "goals_against"), ("gf", "ga")]:
            if k1 in obj and k2 in obj:
                try:
                    return int(obj.get(k1)), int(obj.get(k2))
                except Exception:
                    return None, None

    return None, None


def _compute_stats_from_recent(recent_rows):
    """
    Computes basic counts from recent fixtures:
      - btts_count
      - over25_count
      - under25_count
      - n
    recent_rows: list[dict]
    """
    if not isinstance(recent_rows, list) or not recent_rows:
        return None

    btts = 0
    o25 = 0
    u25 = 0
    n = 0

    for r in recent_rows:
        gf, ga = _parse_score_obj(r if isinstance(r, dict) else None)
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

    return {
        "n": n,
        "btts": btts,
        "o25": o25,
        "u25": u25,
    }


def _stat_line(market: str, home_form: dict, away_form: dict):
    """
    Builds the extra "Stat:" line if we can.
    Prefers explicit keys if present; otherwise computes from recent lists if available.
    """
    # If your hybrid form ever adds explicit counts, we’ll use them:
    # (safe lookups; won’t break)
    # Examples (optional future keys):
    #  home_form["btts_last_n"] = {"hit": 3, "n": 5}
    #  home_form["o25_last_n"] = {"hit": 4, "n": 5}
    def _extract_explicit(form, key):
        v = form.get(key)
        if isinstance(v, dict):
            hit = _i(v.get("hit"))
            n = _i(v.get("n"))
            if hit is not None and n:
                return hit, n
        return None

    # fallback: compute from recent lists if present
    home_recent = home_form.get("recent") or home_form.get("home_recent") or []
    away_recent = away_form.get("recent") or away_form.get("away_recent") or []

    hs = _compute_stats_from_recent(home_recent)
    as_ = _compute_stats_from_recent(away_recent)

    # Nothing to show
    if hs is None and as_ is None:
        return None

    if market in ("BTTS_Y", "BTTS_N"):
        if hs and as_:
            return f"Stat: BTTS landed {hs['btts']}/{hs['n']} (home), {as_['btts']}/{as_['n']} (away)."
        if hs:
            return f"Stat: BTTS landed {hs['btts']}/{hs['n']} in home’s recent games."
        if as_:
            return f"Stat: BTTS landed {as_['btts']}/{as_['n']} in away’s recent games."

    if market == "O2.5":
        if hs and as_:
            return f"Stat: Over 2.5 hit {hs['o25']}/{hs['n']} (home), {as_['o25']}/{as_['n']} (away)."
        if hs:
            return f"Stat: Over 2.5 hit {hs['o25']}/{hs['n']} in home’s recent games."
        if as_:
            return f"Stat: Over 2.5 hit {as_['o25']}/{as_['n']} in away’s recent games."

    if market == "U2.5":
        if hs and as_:
            return f"Stat: Under 2.5 hit {hs['u25']}/{hs['n']} (home), {as_['u25']}/{as_['n']} (away)."
        if hs:
            return f"Stat: Under 2.5 hit {hs['u25']}/{hs['n']} in home’s recent games."
        if as_:
            return f"Stat: Under 2.5 hit {as_['u25']}/{as_['n']} in away’s recent games."

    # For 1X2/DC we’ll skip the stat (unless you later add explicit results keys).
    return None


def mk_blurb(market: str, home_form: dict, away_form: dict, p: float) -> str:
    """
    Rich-but-short blurbs with one concrete stat line where available.
    Defensive on missing fields.
    """
    hg = _f(home_form.get("avg_gf"))
    hga = _f(home_form.get("avg_ga"))
    ag = _f(away_form.get("avg_gf"))
    aga = _f(away_form.get("avg_ga"))

    hw = _wdl(home_form)
    aw = _wdl(away_form)

    wdl_home = f"{hw[0]}W-{hw[1]}D-{hw[2]}L" if hw else None
    wdl_away = f"{aw[0]}W-{aw[1]}D-{aw[2]}L" if aw else None

    stat = _stat_line(market, home_form, away_form)

    # --- Generic fallback if form lacks key metrics ---
    if hg is None or hga is None or ag is None or aga is None:
        pct = f"{p*100:.0f}%"
        base = None
        if market in ("BTTS_Y", "BTTS_N"):
            base = f"Fair price is derived from CSB’s form model. CSB prices this at ~{pct}."
        elif market.startswith("O") or market.startswith("U"):
            base = f"Fair price is derived from CSB’s goals model. CSB prices this at ~{pct}."
        else:
            base = f"Fair price is derived from CSB’s form-based model probabilities (~{pct})."

        if stat:
            return f"{base} {stat}"
        return base

    # Useful derived signals
    total_gf = hg + ag
    total_ga = hga + aga
    home_goal_diff = hg - hga
    away_goal_diff = ag - aga

    # ---------------- BTTS ----------------
    if market == "BTTS_Y":
        bits = [
            f"Both sides are active at both ends: scoring {hg:.1f}/{ag:.1f} gpg and conceding {hga:.1f}/{aga:.1f}.",
        ]
        if wdl_home and wdl_away:
            bits.append(f"Form: {wdl_home} vs {wdl_away}.")
        bits.append(f"CSB prices BTTS Yes at ~{p*100:.0f}%.")
        if stat:
            bits.append(stat)
        return " ".join(bits)

    if market == "BTTS_N":
        bits = [
            f"At least one side profiles as containable: GF {hg:.1f}/{ag:.1f}, GA {hga:.1f}/{aga:.1f}.",
        ]
        if wdl_home and wdl_away:
            bits.append(f"Form: {wdl_home} vs {wdl_away}.")
        bits.append(f"CSB prices BTTS No at ~{p*100:.0f}%.")
        if stat:
            bits.append(stat)
        return " ".join(bits)

    # -------------- Totals (O/U 2.5) --------------
    if market == "O2.5":
        bits = [
            f"Goals environment looks high (combined GF {total_gf:.1f}, combined GA {total_ga:.1f}).",
            "That usually correlates with more chances + higher totals.",
            f"CSB prices Over 2.5 at ~{p*100:.0f}%.",
        ]
        if stat:
            bits.append(stat)
        return " ".join(bits)

    if market == "U2.5":
        bits = [
            f"Goals environment leans lower (combined GF {total_gf:.1f}, combined GA {total_ga:.1f}).",
            "This often means fewer clear chances and a tighter game state.",
            f"CSB prices Under 2.5 at ~{p*100:.0f}%.",
        ]
        if stat:
            bits.append(stat)
        return " ".join(bits)

    # ---------------- 1X2 ----------------
    if market == "HOME_WIN":
        bits = [
            f"Home goal trend {home_goal_diff:+.1f} (GF {hg:.1f}, GA {hga:.1f}) vs away {away_goal_diff:+.1f} (GF {ag:.1f}, GA {aga:.1f})."
        ]
        if wdl_home and wdl_away:
            bits.append(f"Form: {wdl_home} vs {wdl_away}.")
        bits.append(f"CSB prices Home Win at ~{p*100:.0f}%.")
        return " ".join(bits)

    if market == "AWAY_WIN":
        bits = [
            f"Away goal trend {away_goal_diff:+.1f} (GF {ag:.1f}, GA {aga:.1f}) vs home {home_goal_diff:+.1f} (GF {hg:.1f}, GA {hga:.1f})."
        ]
        if wdl_home and wdl_away:
            bits.append(f"Form: {wdl_home} vs {wdl_away}.")
        bits.append(f"CSB prices Away Win at ~{p*100:.0f}%.")
        return " ".join(bits)

    if market == "DRAW":
        bits = [
            f"Profiles are fairly close (GF {hg:.1f}/{ag:.1f}, GA {hga:.1f}/{aga:.1f}).",
            "That tends to lift draw probability in the model.",
        ]
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
            f"Combined scoring/conceding suggests decisive outcomes are plausible "
            f"(GF {total_gf:.1f}, GA {total_ga:.1f}). "
            f"CSB prices 12 at ~{p*100:.0f}%."
        )

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

    # Pull hybrid form (pass FIXTURE object)
    try:
        hybrid = get_hybrid_form_for_fixture(db, fx)
    except Exception:
        hybrid = None

    # Some versions of hybrid might return {home_form, away_form, home_recent, away_recent}
    home_form = (hybrid or {}).get("home_form") or {}
    away_form = (hybrid or {}).get("away_form") or {}

    # If recent lists exist at top-level, attach them so mk_blurb can compute stats
    if isinstance((hybrid or {}).get("home_recent"), list) and "recent" not in home_form:
        home_form = dict(home_form)
        home_form["recent"] = (hybrid or {}).get("home_recent") or []

    if isinstance((hybrid or {}).get("away_recent"), list) and "recent" not in away_form:
        away_form = dict(away_form)
        away_form["recent"] = (hybrid or {}).get("away_recent") or []

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