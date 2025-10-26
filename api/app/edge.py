# api/app/edge.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import median
from collections import defaultdict
from typing import Dict, Tuple, List, Optional, Any
import re
import math

from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from sqlalchemy import or_

from .models import Odds, ModelProb, Edge, Fixture, ClosingOdds, Prediction
from .services.league_strength import get_team_strength  # (kept if you use it elsewhere)
from .services.utils import confidence_from_prob
# ✅ API-backed hybrid form (DB fallback)
from .services.form import get_hybrid_form_for_fixture

# ----------------------------
# Config
# ----------------------------
HOURS_AHEAD = 96
STALE_ODDS_HOURS = 72
MODEL_SOURCE = "team_form"  # ✅ switched from consensus_v2

# Which book’s historical calibration we shrink toward
CAL_BOOK = "bet365"

PREFERRED_BOOK: Optional[str] = None

# Excluded from MODEL consensus (can still appear in edges if you want)
MODEL_BOOK_BLACKLIST = {"1xbet"}

# Sanity guards
MIN_BOOKS_FOR_MODEL = 2
MIN_DEC_ODDS = 1.02
MAX_DEC_ODDS = 6.0   # tighter cap to kill wild edges

# ----------------------------
# Helpers
# ----------------------------
def _norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def _is_blacklisted_book(name: str | None) -> bool:
    return _norm(name) in MODEL_BOOK_BLACKLIST

def _ok_price(x: float | None) -> bool:
    try:
        v = float(x)
        return (v >= MIN_DEC_ODDS) and (v <= MAX_DEC_ODDS)
    except Exception:
        return False

def _implied(p: float) -> float:
    return 1.0 / p if p and p > 0 else 0.0

def _devig_pair(price_a: float, price_b: float) -> Tuple[float, float]:
    ia = _implied(price_a); ib = _implied(price_b)
    s = ia + ib
    if s <= 0: return (0.0, 0.0)
    return (ia / s, ib / s)

def _devig_three(price1: float, price2: float, price3: float) -> Tuple[float, float, float]:
    i1, i2, i3 = _implied(price1), _implied(price2), _implied(price3)
    s = i1 + i2 + i3
    if s <= 0: return (0.0, 0.0, 0.0)
    return (i1 / s, i2 / s, i3 / s)

def _median_consensus(vals: List[float]) -> float:
    vals = [v for v in vals if 0.0 < v < 1.0]
    return float(median(vals)) if vals else 0.0

def _is_gridiron(comp: str | None) -> bool:
    cu = (comp or "").upper()
    return ("NFL" in cu) or ("CFB" in cu) or ("NCAA" in cu) or ("AMERICAN" in cu)

def _is_ice(comp: str | None) -> bool:
    cu = (comp or "").upper()
    return ("NHL" in cu) or ("ICE" in cu) or ("HOCKEY" in cu)

# --- Market canonicalization (single source of truth) ------------------------
def _canon_market(m: str | None) -> str:
    if not m:
        return ""
    x = m.strip().upper().replace(" ", "").replace("-", "")
    # Totals like O2.5/U2.5 keep their numeric
    if x.startswith("O") and x[1:].replace(".", "", 1).isdigit():
        return x
    if x.startswith("U") and x[1:].replace(".", "", 1).isdigit():
        return x
    SYN = {
        # 1X2 synonyms
        "1": "HOME_WIN", "HOME": "HOME_WIN", "HOMEWIN": "HOME_WIN",
        "MATCHWINNERHOME": "HOME_WIN", "TEAM1": "HOME_WIN",
        "X": "DRAW", "DRAW": "DRAW",
        "2": "AWAY_WIN", "AWAY": "AWAY_WIN", "AWAYWIN": "AWAY_WIN",
        "MATCHWINNERAWAY": "AWAY_WIN", "TEAM2": "AWAY_WIN",

        # Double chance (usually already canonical)
        "1X": "1X", "12": "12", "X2": "X2",

        # BTTS
        "BTTSYES": "BTTS_Y", "BTTSY": "BTTS_Y", "BOTHTEAMSTOSCOREYES": "BTTS_Y",
        "BTTSNO": "BTTS_N",  "BTTSN": "BTTS_N", "BOTHTEAMSTOSCORENO":  "BTTS_N",
    }
    return SYN.get(x, x)

# --- SAFE FORM HELPERS -------------------------------------------------------

DEFAULT_SUMMARY: Dict[str, float | int] = {
    # counts
    "played": 0,
    "wins": 0,
    "draws": 0,
    "losses": 0,
    # averages per game
    "avg_goals_for": 0.0,
    "avg_goals_against": 0.0,
    # totals (optional – computed if missing)
    "goals_for": 0.0,
    "goals_against": 0.0,
}

def _coerce_summary(raw: Optional[Dict[str, Any]]) -> Dict[str, float | int]:
    """
    Accepts a 'summary' dict (or None) and returns a normalized summary
    with keys we use everywhere. Handles both 'avg_*' and raw totals.
    """
    if not isinstance(raw, dict):
        return dict(DEFAULT_SUMMARY)

    out = dict(DEFAULT_SUMMARY)

    # Prefer averages if present, else compute from totals/played
    played = int(raw.get("played") or raw.get("matches") or 0)
    out["played"] = played
    out["wins"] = int(raw.get("wins") or 0)
    out["draws"] = int(raw.get("draws") or 0)
    out["losses"] = int(raw.get("losses") or 0)

    # Read both avg keys and total keys tolerantly
    gf_total = float(raw.get("goals_for") or raw.get("gf") or 0.0)
    ga_total = float(raw.get("goals_against") or raw.get("ga") or 0.0)

    gf_avg = raw.get("avg_goals_for")
    ga_avg = raw.get("avg_goals_against")

    if gf_avg is None:
        gf_avg = (gf_total / played) if played else 0.0
    if ga_avg is None:
        ga_avg = (ga_total / played) if played else 0.0

    out["avg_goals_for"] = float(gf_avg or 0.0)
    out["avg_goals_against"] = float(ga_avg or 0.0)
    out["goals_for"] = float(gf_total or 0.0) if gf_total else float(out["avg_goals_for"]) * played
    out["goals_against"] = float(ga_total or 0.0) if ga_total else float(out["avg_goals_against"]) * played

    return out

def _fallback_summary_from_db_recent(db: Session, team: str, comp: str, ko: datetime, limit: int = 5) -> Dict[str, float | int]:
    """
    Build a form-like summary using DB fixtures when hybrid payload is missing.
    """
    rows = (
        db.query(Fixture)
        .filter(
            Fixture.kickoff_utc < ko,
            Fixture.result_settled == True,
            Fixture.comp == comp,
            or_(Fixture.home_team == team, Fixture.away_team == team)
        )
        .order_by(Fixture.kickoff_utc.desc())
        .limit(limit)
        .all()
    )
    played = len(rows)
    wins = draws = losses = 0
    gf_total = ga_total = 0

    for f in rows:
        is_home = (f.home_team == team)
        gf = (f.full_time_home if is_home else f.full_time_away) or 0
        ga = (f.full_time_away if is_home else f.full_time_home) or 0

        gf_total += gf
        ga_total += ga

        if gf > ga:
            wins += 1
        elif gf == ga:
            draws += 1
        else:
            losses += 1

    return {
        "played": played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "avg_goals_for": (gf_total / played) if played else 0.0,
        "avg_goals_against": (ga_total / played) if played else 0.0,
        "goals_for": float(gf_total),
        "goals_against": float(ga_total),
    }

def _safe_get_forms(db: Session, f: Fixture, n: int = 5) -> tuple[dict, dict]:
    """
    Returns (home_summary, away_summary) safely, with DB fallback if hybrid form missing.
    """
    forms = {}
    try:
        forms = get_hybrid_form_for_fixture(db, f, n=n, comp_scope=True) or {}
    except Exception:
        forms = {}

    home_raw = (forms.get("home") or {}).get("summary")
    away_raw = (forms.get("away") or {}).get("summary")

    if not home_raw:
        home_raw = _fallback_summary_from_db_recent(db, f.home_team, f.comp or "", f.kickoff_utc, limit=n)
    if not away_raw:
        away_raw = _fallback_summary_from_db_recent(db, f.away_team, f.comp or "", f.kickoff_utc, limit=n)

    return _coerce_summary(home_raw), _coerce_summary(away_raw)

# ----------------------------
# Build model_probs
# ----------------------------
# Accept whole and fractional lines (e.g., O2, U2.5, O2.75)
_OU_RE = re.compile(r"^(O|U)(\d+(?:\.\d+)?)$")

def _confidence_label(prob: float) -> str:
    """Convert raw probability into a confidence label."""
    if prob >= 0.70:
        return "High"
    elif prob >= 0.55:
        return "Medium"
    return "Low"


def ensure_baseline_probs(
    db: Session,
    now: datetime,
    hours_ahead: int = HOURS_AHEAD,
    staleness_hours: int = STALE_ODDS_HOURS,
    source: str = "team_form",
    time_window: tuple[datetime, datetime] | None = None,
    league_comp_filter: list[str] | None = None,
    use_closing_when_past: bool = True,
) -> None:
    if time_window:
        start_dt, end_dt = time_window
    else:
        start_dt = now
        end_dt = now + timedelta(hours=hours_ahead)

    use_closing = use_closing_when_past and (end_dt <= now)

    fq = db.query(Fixture).filter(
        Fixture.kickoff_utc >= start_dt,
        Fixture.kickoff_utc < end_dt,
    )
    if league_comp_filter:
        fq = fq.filter(Fixture.comp.in_(league_comp_filter))

    if use_closing:
        fq = fq.join(ClosingOdds, ClosingOdds.fixture_id == Fixture.id).distinct()
    else:
        cutoff_seen = now - timedelta(hours=staleness_hours)
        fq = fq.join(Odds, Odds.fixture_id == Fixture.id).filter(Odds.last_seen >= cutoff_seen).distinct()

    fixtures: List[Fixture] = fq.all()
    if not fixtures:
        return

    # wipe existing model_probs
    db.query(ModelProb).filter(
        and_(ModelProb.source == source, ModelProb.fixture_id.in_([f.id for f in fixtures]))
    ).delete(synchronize_session=False)
    db.flush()

    def _collect_prices_live(fx_id: int) -> Dict[str, List[float]]:
        cutoff_seen = now - timedelta(hours=staleness_hours)
        rows: List[Odds] = db.query(Odds).filter(Odds.fixture_id == fx_id, Odds.last_seen >= cutoff_seen).all()
        out: Dict[str, List[float]] = defaultdict(list)
        for o in rows:
            if _is_blacklisted_book(o.bookmaker):
                continue
            try:
                price = float(o.price)
            except Exception:
                continue
            if _ok_price(price):
                out[_canon_market(o.market)].append(price)  # ← normalize key
        return out

    def _collect_prices_closing(fx_id: int) -> Dict[str, List[float]]:
        rows: List[ClosingOdds] = db.query(ClosingOdds).filter(ClosingOdds.fixture_id == fx_id).all()
        out: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            try:
                price = float(r.price)
            except Exception:
                continue
            if _ok_price(price):
                out[_canon_market(r.market)].append(price)  # ← normalize key
        return out

    for f in fixtures:
        prices = _collect_prices_closing(f.id) if use_closing else _collect_prices_live(f.id)
        if not prices:
            continue

        # ✅ SAFE form summaries with DB fallback
        home_form, away_form = _safe_get_forms(db, f, n=5)
        is_grid = _is_gridiron(f.comp)
        is_ice  = _is_ice(f.comp)

        def median_price(market: str, min_books: int = MIN_BOOKS_FOR_MODEL) -> Optional[float]:
            arr = [p for p in prices.get(market, []) if _ok_price(p)]
            if len(arr) < min_books:
                return None
            return float(median(arr))

        # -------------------------
        # Totals (All sports)
        # -------------------------
        totals_by_line: Dict[str, Dict[str, str]] = defaultdict(dict)
        for mkt in prices.keys():
            m = _OU_RE.match(mkt or "")
            if not m:
                continue
            side, line = m.group(1), m.group(2)
            totals_by_line[line][side] = mkt

        for line, sides in totals_by_line.items():
            om, um = sides.get("O"), sides.get("U")
            if not om or not um:
                continue
            po, pu = median_price(om), median_price(um)
            if not (po and pu):
                continue

            try:
                float(line)  # parsed but unused (kept for clarity)
            except Exception:
                continue

            cons_over, cons_under = _devig_pair(po, pu)
            form_over = adjust_prob_for_goals(home_form, away_form)  # robust (handles avg vs totals)
            p_over = (0.6 * form_over) + (0.4 * cons_over)
            p_under = 1.0 - p_over

            p_over  = _apply_calibration(db, om, CAL_BOOK, p_over)
            p_under = _apply_calibration(db, um, CAL_BOOK, p_under)

            if 0.0 < p_over < 1.0 and 0.0 < p_under < 1.0:
                db.add(ModelProb(fixture_id=f.id, source=source, market=om, prob=p_over,  as_of=now))
                db.add(ModelProb(fixture_id=f.id, source=source, market=um, prob=p_under, as_of=now))

                # ✅ Prediction row with clear market label
                top_side, top_prob = (om, p_over) if p_over >= p_under else (um, p_under)
                db.add(Prediction(
                    fixture_id=f.id,
                    market=f"TOTALS_{line}",
                    predicted_side=top_side,
                    prob=top_prob,
                    fair_price=(1 / top_prob if top_prob > 0 else None),
                    model_source=source,
                    confidence=confidence_from_prob(top_prob)
                ))

        # -------------------------
        # Gridiron & Ice: Moneyline (2-way)
        # -------------------------
        if is_grid or is_ice:
            ph_price = median_price("HOME_WIN", min_books=1)
            pa_price = median_price("AWAY_WIN", min_books=1)
            if ph_price and pa_price:
                ph, pa = _devig_pair(ph_price, pa_price)

                if 0.0 < ph < 1.0 and 0.0 < pa < 1.0:
                    db.add(ModelProb(fixture_id=f.id, source=source, market="HOME_WIN", prob=ph, as_of=now))
                    db.add(ModelProb(fixture_id=f.id, source=source, market="AWAY_WIN", prob=pa, as_of=now))

                    top_side, top_prob = ("HOME_WIN", ph) if ph >= pa else ("AWAY_WIN", pa)
                    db.add(Prediction(
                        fixture_id=f.id,
                        market="MONEYLINE",
                        predicted_side=top_side,
                        prob=top_prob,
                        fair_price=(1 / top_prob if top_prob > 0 else None),
                        model_source=source,
                        confidence=confidence_from_prob(top_prob),
                    ))

        # -------------------------
        # BTTS  (soccer only)
        # -------------------------
        if not is_grid and not is_ice:
            pa, pb = median_price("BTTS_Y"), median_price("BTTS_N")
            if pa and pb:
                cons_yes, cons_no = _devig_pair(pa, pb)

                gf_home = float(home_form.get("avg_goals_for", 0.0) or 0.0)
                ga_home = float(home_form.get("avg_goals_against", 0.0) or 0.0)
                gf_away = float(away_form.get("avg_goals_for", 0.0) or 0.0)
                ga_away = float(away_form.get("avg_goals_against", 0.0) or 0.0)

                offensive = (gf_home + gf_away) / 2
                defensive = (ga_home + ga_away) / 2
                avg_goals_potential = (0.6 * offensive) + (0.4 * defensive)
                goals_factor = max(0.5, min(avg_goals_potential / 2.5, 1.5))

                form_yes = min(0.90, max(0.10, cons_yes * goals_factor))
                p_yes = (0.6 * form_yes) + (0.4 * cons_yes)
                p_no = 1.0 - p_yes

                p_yes = _apply_calibration(db, "BTTS_Y", CAL_BOOK, p_yes)
                p_no  = _apply_calibration(db, "BTTS_N", CAL_BOOK, p_no)

                if 0.0 < p_yes < 1.0 and 0.0 < p_no < 1.0:
                    db.add(ModelProb(fixture_id=f.id, source=source, market="BTTS_Y", prob=p_yes, as_of=now))
                    db.add(ModelProb(fixture_id=f.id, source=source, market="BTTS_N", prob=p_no,  as_of=now))

                    top_side, top_prob = ("BTTS_Y", p_yes) if p_yes >= p_no else ("BTTS_N", p_no)
                    db.add(Prediction(
                        fixture_id=f.id,
                        market="BTTS",
                        predicted_side=top_side,
                        prob=top_prob,
                        fair_price=(1 / top_prob if top_prob > 0 else None),
                        model_source=source,
                        confidence=confidence_from_prob(top_prob)
                    ))

        # -------------------------
        # Soccer 1X2 (3-way)
        # -------------------------
        if not _is_gridiron(f.comp) and not _is_ice(f.comp):
            # allow 1 book to avoid missing legs when feeds are sparse
            p_home = median_price("HOME_WIN", min_books=1)
            p_draw = median_price("DRAW",     min_books=1)
            p_away = median_price("AWAY_WIN", min_books=1)
            if p_home and p_draw and p_away:
                ph, pd, pa = _devig_three(p_home, p_draw, p_away)

                if 0.0 < ph < 1.0 and 0.0 < pd < 1.0 and 0.0 < pa < 1.0:
                    db.add(ModelProb(fixture_id=f.id, source=source, market="HOME_WIN", prob=ph, as_of=now))
                    db.add(ModelProb(fixture_id=f.id, source=source, market="DRAW",     prob=pd, as_of=now))
                    db.add(ModelProb(fixture_id=f.id, source=source, market="AWAY_WIN", prob=pa, as_of=now))

                    # --- Double Chance derived ---
                    def _cap(x: float) -> float:
                        return min(max(x, 0.0), 1.0)

                    db.add(ModelProb(fixture_id=f.id, source=source, market="1X", prob=_cap(ph + pd), as_of=now))
                    db.add(ModelProb(fixture_id=f.id, source=source, market="12", prob=_cap(ph + pa), as_of=now))
                    db.add(ModelProb(fixture_id=f.id, source=source, market="X2", prob=_cap(pd + pa), as_of=now))

                    # Top prediction (optional)
                    top_side, top_prob = max(
                        [("HOME_WIN", ph), ("DRAW", pd), ("AWAY_WIN", pa)],
                        key=lambda x: x[1]
                    )
                    db.add(Prediction(
                        fixture_id=f.id,
                        market="1X2",
                        predicted_side=top_side,
                        prob=top_prob,
                        fair_price=(1 / top_prob if top_prob > 0 else None),
                        model_source=source,
                        confidence=confidence_from_prob(top_prob)
                    ))

    db.commit()


def get_recent_form(
    db: Session,
    team: str,
    comp: str,
    current_ko: datetime,
    limit: int = 5,
) -> dict:
    """Returns recent form (last N results) for a team in the same comp before current_ko."""
    past_matches = db.query(Fixture).filter(
        Fixture.comp == comp,
        Fixture.kickoff_utc < current_ko,
        Fixture.result_settled == True,
        ((Fixture.home_team == team) | (Fixture.away_team == team))
    ).order_by(Fixture.kickoff_utc.desc()).limit(limit).all()

    played = 0
    won = 0
    drawn = 0
    lost = 0
    goals_scored = 0
    goals_conceded = 0

    for match in past_matches:
        played += 1
        is_home = match.home_team == team
        scored = match.full_time_home if is_home else match.full_time_away
        conceded = match.full_time_away if is_home else match.full_time_home
        goals_scored += scored
        goals_conceded += conceded

        if scored > conceded:
            won += 1
        elif scored == conceded:
            drawn += 1
        else:
            lost += 1

    return {
        "played": played,
        "won": won,
        "drawn": drawn,
        "lost": lost,
        "avg_scored": goals_scored / played if played else 0,
        "avg_conceded": goals_conceded / played if played else 0,
        "goal_diff": goals_scored - goals_conceded,
        "points": won * 3 + drawn,
    }

# ----------------------------
# Compute edges
# ----------------------------
def compute_edges(
    db: Session,
    now: datetime,
    since: Optional[datetime] = None,
    fixture_id: Optional[int] = None,
    min_edge: float = 0.00,
    source: str = MODEL_SOURCE,
    hours_ahead: int = HOURS_AHEAD,
    staleness_hours: int = STALE_ODDS_HOURS,
    prefer_book: Optional[str] = None,
) -> int:
    # Reuse global canonicalizer
    def normalize_market(m: str) -> str:
        return _canon_market(m)

    cutoff_ko = now + timedelta(hours=hours_ahead)
    cutoff_seen = now - timedelta(hours=staleness_hours)

    # Delete existing edges
    if fixture_id:
        db.query(Edge).filter(
            Edge.model_source == source,
            Edge.fixture_id == fixture_id
        ).delete(synchronize_session=False)
    elif since:
        fixture_ids = (
            db.query(Fixture.id)
            .filter(Fixture.kickoff_utc >= since)
            .subquery()
        )
        db.query(Edge).filter(
            Edge.model_source == source,
            Edge.fixture_id.in_(fixture_ids)
        ).delete(synchronize_session=False)
    else:
        db.query(Edge).filter(Edge.model_source == source).delete(synchronize_session=False)

    db.flush()

    # Latest model probs
    latest_prob = (
        db.query(
            ModelProb.fixture_id,
            ModelProb.market,
            func.max(ModelProb.as_of).label("latest"),
        )
        .filter(ModelProb.source == source)
        .group_by(ModelProb.fixture_id, ModelProb.market)
        .subquery()
    )

    probs_q = (
        db.query(ModelProb)
        .join(latest_prob, and_(
            ModelProb.fixture_id == latest_prob.c.fixture_id,
            ModelProb.market == latest_prob.c.market,
            ModelProb.as_of == latest_prob.c.latest,
        ))
        .join(Fixture, Fixture.id == ModelProb.fixture_id)
        .filter(Fixture.kickoff_utc <= cutoff_ko)
    )

    if fixture_id:
        probs_q = probs_q.filter(ModelProb.fixture_id == fixture_id)
    else:
        probs_q = probs_q.filter(Fixture.kickoff_utc >= (since or now))

    probs = probs_q.all()

    # Normalize probs
    p_map: Dict[Tuple[int, str], float] = {
        (p.fixture_id, normalize_market(p.market)): float(p.prob)
        for p in probs if p.market and p.prob is not None
    }

    eff_book = prefer_book if prefer_book else PREFERRED_BOOK

    # Odds query
    price_q = db.query(Odds).join(Fixture, Fixture.id == Odds.fixture_id)

    if fixture_id:
        price_q = price_q.filter(Fixture.id == fixture_id)
    else:
        price_q = price_q.filter(Fixture.kickoff_utc >= (since or now))

    price_q = price_q.filter(
        Fixture.kickoff_utc <= cutoff_ko,
        Odds.last_seen >= cutoff_seen,
    )

    count = 0

    for o in price_q.all():
        try:
            price = float(o.price)
        except Exception:
            continue

        if not _ok_price(price):
            continue

        norm_market = normalize_market(o.market)
        key = (o.fixture_id, norm_market)
        p = p_map.get(key)

        if p is None or p <= 0:
            continue

        ev = p * price - 1.0
        if ev > 1.0:
            ev = 1.0

        if ev >= min_edge:
            db.add(Edge(
                fixture_id=o.fixture_id,
                market=norm_market,
                bookmaker=o.bookmaker,
                price=price,
                model_source=source,
                prob=p,
                edge=ev,
                created_at=now,
            ))
            count += 1

    db.commit()
    return count

# ----------------------------
# Calibration shrink helper
# ----------------------------
def _apply_calibration(
    db: Session,
    market: str,
    book: str,
    p: float,
    scope: str | None = None,   # pass Fixture.comp if you want per-league calibration
) -> float:
    """
    Apply logistic calibration: p' = sigmoid(alpha + beta * logit(p)).
    Falls back: scope-specific -> global -> raw p.
    """
    from math import log
    from .models import Calibration

    def _clip(x: float, lo: float = 1e-6, hi: float = 1 - 1e-6) -> float:
        return lo if x < lo else hi if x > hi else x

    def _sigmoid(z: float) -> float:
        # numerically stable
        if z >= 0:
            ez = (2.718281828459045 ** (-z))
            return 1.0 / (1.0 + ez)
        else:
            ez = (2.718281828459045 ** (z))
            return ez / (1.0 + ez)

    def _logit(x: float) -> float:
        x = _clip(x)
        return math.log(x / (1.0 - x))

    if not (0.0 < p < 1.0):
        return _clip(p)

    # Try scope-specific first (e.g., scope="League One"), then global
    scopes_to_try = []
    if scope:
        scopes_to_try.append(scope)
    scopes_to_try.append("global")

    for sc in scopes_to_try:
        row = (
            db.query(Calibration)
              .filter(
                  Calibration.market == market,
                  Calibration.book == book,
                  Calibration.scope == sc
              )
              .one_or_none()
        )
        if row:
            try:
                z = row.alpha + row.beta * _logit(p)
                return _clip(_sigmoid(z))
            except Exception:
                break

    return _clip(p)

# ----------------------------
# Team form helper (DB-only legacy – kept for reference)
# ----------------------------
def get_team_form_features(
    db: Session,
    team: str,
    kickoff: datetime,
    n: int = 5,
) -> dict:
    rows = (
        db.query(Fixture)
        .filter(
            Fixture.kickoff_utc < kickoff,
            Fixture.result_settled == True,
            or_(Fixture.home_team == team, Fixture.away_team == team)
        )
        .order_by(Fixture.kickoff_utc.desc())
        .limit(n)
        .all()
    )

    played = 0
    wins = draws = losses = 0
    goals_for = goals_against = 0

    for f in rows:
        played += 1
        is_home = f.home_team == team
        gf = f.full_time_home if is_home else f.full_time_away
        ga = f.full_time_away if is_home else f.full_time_home

        goals_for += gf
        goals_against += ga

        if gf > ga:
            wins += 1
        elif gf == ga:
            draws += 1
        else:
            losses += 1

    return {
        "played": played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_diff": goals_for - goals_against,
    }

def adjust_prob_for_form(p: float, team_form: dict, opp_form: dict, market: str | None = None) -> float:
    """
    Adjust probability based on form.
    - For HOME_WIN / AWAY_WIN: uses W/D/L score.
    - For BTTS: uses average goals scored.
    - For O/U: uses combined avg goals scored + conceded.
    """
    def _clip(x: float, lo: float = 0.01, hi: float = 0.99) -> float:
        return max(lo, min(hi, x))

    if not team_form or not opp_form:
        return p

    if market in {"HOME_WIN", "AWAY_WIN"}:
        def form_score(f: dict) -> float:
            return float(f.get("wins", 0)) * 0.3 + float(f.get("draws", 0)) * 0.0 + float(f.get("losses", 0)) * -0.2

        delta = form_score(team_form) - form_score(opp_form)
        adjustment = 0.02 * delta
        return _clip(p + max(-0.1, min(0.1, adjustment)))

    elif market == "BTTS_Y":
        avg_goals = float(team_form.get("avg_goals_for", 0.0)) + float(opp_form.get("avg_goals_for", 0.0))
        # historical center ≈ 2.5 goals
        boost = 0.03 * ((avg_goals / 2.0) - 2.5)
        return _clip(p + max(-0.08, min(0.08, boost)))

    elif market == "BTTS_N":
        avg_goals = float(team_form.get("avg_goals_for", 0.0)) + float(opp_form.get("avg_goals_for", 0.0))
        drop = -0.03 * ((avg_goals / 2.0) - 2.5)
        return _clip(p + max(-0.08, min(0.08, drop)))

    elif market and (market.startswith("O") or market.startswith("U")):
        avg_gf = float(team_form.get("avg_goals_for", 0.0)) + float(opp_form.get("avg_goals_for", 0.0))
        avg_ga = float(team_form.get("avg_goals_against", 0.0)) + float(opp_form.get("avg_goals_against", 0.0))
        avg_total = (avg_gf + avg_ga) / 2.0

        try:
            line = float(market[1:])
        except Exception:
            return _clip(p)

        if market.startswith("O"):
            return _clip(p + max(-0.10, min(0.10, 0.03 * (avg_total - line))))
        else:
            return _clip(p + max(-0.10, min(0.10, -0.03 * (avg_total - line))))

    return _clip(p)

def adjust_prob_for_goals(home_form: dict, away_form: dict) -> float:
    """
    Adjust Over/Under probability using form-based average goals.
    Works whether summary has avg_* keys or only totals+played.
    """
    # derive averages (robustly)
    def _avg(f: dict, key_avg: str, key_total: str) -> float:
        if key_avg in f:
            try:
                return float(f[key_avg] or 0.0)
            except Exception:
                return 0.0
        tot = float(f.get(key_total, 0.0) or 0.0)
        n = int(f.get("played", 0) or 0)
        return (tot / n) if n else 0.0

    h_gf = _avg(home_form, "avg_goals_for", "goals_for")
    h_ga = _avg(home_form, "avg_goals_against", "goals_against")
    a_gf = _avg(away_form, "avg_goals_for", "goals_for")
    a_ga = _avg(away_form, "avg_goals_against", "goals_against")

    avg_total_goals = (h_gf + h_ga + a_gf + a_ga) / 2.0
    avg_total_goals = max(0.5, min(avg_total_goals, 5.0))  # clamp

    # Sigmoid mapping around 2.5
    base_prob = 1 / (1 + math.exp(-0.7 * (avg_total_goals - 2.5)))
    linear_adjustment = max(-0.03, min(0.03, (avg_total_goals - 2.5) * 0.02))

    adjusted_prob = base_prob + linear_adjustment
    return max(0.05, min(0.80, adjusted_prob))

# --- Form-driven explanation helpers -----------------------------------------

def _blend_avg(x: float, y: float) -> float:
    """Blend a team's GF with opponent GA (robust when samples are small)."""
    try:
        return round(0.6 * float(x or 0) + 0.4 * float(y or 0), 2)
    except Exception:
        return 0.0

def expected_goals_from_form(home_summary: dict, away_summary: dict) -> dict:
    """
    Compute blended team expectations + match total directly from hybrid form.

    Formula weights offensive vs defensive form dynamically:
    - Home expected goals = (0.65 * home_avg_GF) + (0.35 * away_avg_GA)
    - Away expected goals = (0.65 * away_avg_GF) + (0.35 * home_avg_GA)
    Adjusted slightly by how many games each side has played.
    """

    # Safely extract averages
    h_gf = float(home_summary.get("avg_goals_for", 0.0) or 0.0)
    h_ga = float(home_summary.get("avg_goals_against", 0.0) or 0.0)
    a_gf = float(away_summary.get("avg_goals_for", 0.0) or 0.0)
    a_ga = float(away_summary.get("avg_goals_against", 0.0) or 0.0)

    # Number of matches played in form sample
    n_home = max(1, int(home_summary.get("played", 0) or 0))
    n_away = max(1, int(away_summary.get("played", 0) or 0))

    # Dynamic weighting: teams with more recent matches get slightly higher confidence
    weight_home = min(0.7, 0.5 + (n_home / 20))
    weight_away = min(0.7, 0.5 + (n_away / 20))

    # Blend attack vs opponent defence
    exp_home = (weight_home * h_gf) + ((1 - weight_home) * a_ga)
    exp_away = (weight_away * a_gf) + ((1 - weight_away) * h_ga)

    # Round + sanity bounds
    exp_home = round(max(0.2, min(exp_home, 3.5)), 2)
    exp_away = round(max(0.2, min(exp_away, 3.5)), 2)
    total = round(exp_home + exp_away, 2)
    total = max(0.8, min(total, 6.5))

    # 🪄 Debug print for verification
    print(
        f"[EXPECTED GOALS DEBUG] "
        f"Home GFpg={h_gf:.2f}, GApg={h_ga:.2f}, Away GFpg={a_gf:.2f}, GApg={a_ga:.2f} | "
        f"n_home={n_home}, n_away={n_away} | "
        f"exp_home={exp_home}, exp_away={exp_away}, total={total}"
    )

    return {
        "home": exp_home,
        "away": exp_away,
        "total": total,
    }

def build_why_from_form(fixture, form_payload: dict) -> str:
    """
    Human-readable explanation string entirely from the hybrid form payload.
    Avoids '0.0 avg' unless there is truly no form data.
    """
    home = (form_payload.get("home") or {}).get("summary") or {}
    away = (form_payload.get("away") or {}).get("summary") or {}

    home = _coerce_summary(home)
    away = _coerce_summary(away)

    n_home = int(home.get("played", 0) or 0)
    n_away = int(away.get("played", 0) or 0)
    scope = (fixture.comp or "all competitions")

    if not n_home and not n_away:
        return (
            f"Form-based projection unavailable for {fixture.home_team} vs {fixture.away_team} "
            f"(no recent matches in scope: {scope}). Edge based on pricing only."
        )

    exp = expected_goals_from_form(home, away)

    def fmt(x):
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "0.00"

    parts = [
        f"{fixture.home_team} vs {fixture.away_team} — form scope: {scope}.",
        f"{fixture.home_team}: {int(home['wins'])}W-{int(home['draws'])}D-{int(home['losses'])}L over last {n_home} "
        f"(GFpg {fmt(home['avg_goals_for'])}, GApg {fmt(home['avg_goals_against'])}).",
        f"{fixture.away_team}: {int(away['wins'])}W-{int(away['draws'])}D-{int(away['losses'])}L over last {n_away} "
        f"(GFpg {fmt(away['avg_goals_for'])}, GApg {fmt(away['avg_goals_against'])}).",
        f"Blended expectation ≈ {fixture.home_team} {fmt(exp['home'])} + "
        f"{fixture.away_team} {fmt(exp['away'])} = {fmt(exp['total'])} match goals.",
    ]
    return " ".join(parts)