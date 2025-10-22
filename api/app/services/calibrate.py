# api/app/services/calibrate.py
from __future__ import annotations
from typing import Optional, List, Tuple
import math
from math import log, isfinite
from datetime import datetime, timedelta, timezone
import re

from sqlalchemy.orm import Session
from sqlalchemy import and_

from ..models import Fixture, ModelProb, ClosingOdds, Calibration

# ---------- helpers

def _clip(x: float, lo: float = 1e-6, hi: float = 1 - 1e-6) -> float:
    return lo if x < lo else hi if x > hi else x

def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(1.0 - eps, max(eps, p))
    return math.log(p / (1.0 - p))

def _sigmoid(z: float) -> float:
    # stable sigmoid
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

def _market_win(mkt: str, ft_h: Optional[int], ft_a: Optional[int]) -> Optional[int]:
    """Return 1/0 outcome for the given market, or None if unknown."""
    if ft_h is None or ft_a is None:
        return None
    total = ft_h + ft_a
    m = (mkt or "").upper()

    if m == "HOME_WIN": return 1 if ft_h > ft_a else 0
    if m == "DRAW":     return 1 if ft_h == ft_a else 0
    if m == "AWAY_WIN": return 1 if ft_a > ft_h else 0
    if m == "BTTS_Y":   return 1 if (ft_h > 0 and ft_a > 0) else 0
    if m == "BTTS_N":   return 1 if not (ft_h > 0 and ft_a > 0) else 0

    mo = re.fullmatch(r"([OU])\s*([0-9]+(?:\.[0-9]+)?)", m)
    if mo:
        side = mo.group(1)
        line = float(mo.group(2))
        if side == "O": return 1 if total > line else 0
        if side == "U": return 1 if total < line else 0
    return None

def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(1.0 - eps, max(eps, p))
    return math.log(p / (1.0 - p))

def _fit_platt(probs, labels, l2: float = 3.0, max_iter: int = 50, eps: float = 1e-6):
    """
    Fit calibrated q = sigmoid(A * logit(p) + B) with L2 regularization on A.
    Returns (B, A, logloss) so it maps to (alpha, beta, logloss).
    """
    n = len(probs)
    if n == 0:
        return 0.0, 1.0, 0.0  # identity

    xs = [_logit(p, eps) for p in probs]
    ys = [1 if y else 0 for y in labels]

    # Guardrails
    if sum(ys) in (0, n) or len(set(round(x, 6) for x in xs)) < 2:
        # identity
        ll = -sum(
            y * math.log(min(1 - eps, max(eps, p))) +
            (1 - y) * math.log(min(1 - eps, max(eps, 1 - p)))
            for y, p in zip(ys, probs)
        ) / n
        return 0.0, 1.0, ll

    A, B = 1.0, 0.0

    for _ in range(max_iter):
        z = [A * x + B for x in xs]
        p_old = [_sigmoid(t) for t in z]

        # gradients (L2 on A)
        gA = sum((pi - yi) * xi for pi, yi, xi in zip(p_old, ys, xs)) + l2 * A
        gB = sum(pi - yi for pi, yi in zip(p_old, ys))

        # Hessian
        w = [pi * (1.0 - pi) for pi in p_old]
        hAA = sum(wi * xi * xi for wi, xi in zip(w, xs)) + l2
        hAB = sum(wi * xi for wi, xi in zip(w, xs))
        hBB = sum(w)

        det = hAA * hBB - hAB * hAB
        if abs(det) < 1e-9:
            break

        dA = -( hBB * gA - hAB * gB) / det
        dB = -(-hAB * gA + hAA * gB) / det

        # backtracking line search
        step = 1.0
        for _try in range(5):
            A_new = A + step * dA
            B_new = B + step * dB

            z_new = [A_new * x + B_new for x in xs]
            p_new = [_sigmoid(t) for t in z_new]

            ll_old = -sum(
                yi * math.log(min(1 - eps, max(eps, pi))) +
                (1 - yi) * math.log(min(1 - eps, max(eps, 1 - pi)))
                for yi, pi in zip(ys, p_old)
            )
            ll_new = -sum(
                yi * math.log(min(1 - eps, max(eps, pj))) +
                (1 - yi) * math.log(min(1 - eps, max(eps, 1 - pj)))
                for yi, pj in zip(ys, p_new)
            )
            if ll_new <= ll_old + 1e-6:
                A, B = A_new, B_new
                break
            step *= 0.5
        else:
            break

        if abs(dA) < 1e-6 and abs(dB) < 1e-6:
            break

    # final logloss
    z = [A * x + B for x in xs]
    p_fin = [_sigmoid(t) for t in z]
    ll = -sum(
        yi * math.log(min(1 - eps, max(eps, pi))) +
        (1 - yi) * math.log(min(1 - eps, max(eps, 1 - pi)))
        for yi, pi in zip(ys, p_fin)
    ) / n

    # alpha=intercept=B, beta=slope=A
    return float(B), float(A), float(ll)
# ---------- public API used by your admin routes

def train_and_store_calibration(
    db: Session,
    market: str,
    book: str,
    days_back: int,
    source: str,
    comp_list: Optional[List[str]],
    scope: str,
    min_samples: int = 10,            # NEW - default to 10
    use_all_books: bool = False       # NEW - ignore for now unless you implement
):
    """
    Train logistic calibration for (market, book, scope).
    """
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days_back)

    q = (
        db.query(
            ModelProb.prob,
            Fixture.full_time_home,
            Fixture.full_time_away,
            Fixture.comp,
        )
        .join(Fixture, Fixture.id == ModelProb.fixture_id)
        .join(ClosingOdds, and_(
            ClosingOdds.fixture_id == Fixture.id,
            ClosingOdds.market == market,
            ClosingOdds.bookmaker == book,
        ))
        .filter(ModelProb.market == market, ModelProb.source == source)
        .filter(Fixture.kickoff_utc >= since, Fixture.kickoff_utc <= now)
        .filter(Fixture.result_settled == True)
    )
    if comp_list:
        q = q.filter(Fixture.comp.in_(comp_list))
    if scope != "global":
        q = q.filter(Fixture.comp == scope)

    rows = q.all()
    probs, labels = [], []

    for p, ft_h, ft_a, _comp in rows:
        try:
            pf = float(p)
        except Exception:
            continue
        if not (isfinite(pf) and 0.0 < pf < 1.0):
            continue
        y = _market_win(market, ft_h, ft_a)
        if y is None:
            continue
        probs.append(pf)
        labels.append(int(y))

    n = len(probs)
    if n < min_samples:   # now respects param
        return {"ok": False, "message": f"Not enough samples to calibrate ({n} < {min_samples})", "market": market, "book": book, "scope": scope}

    alpha, beta, logloss = _fit_platt(probs, labels)

    row = db.query(Calibration).filter(
        Calibration.market == market,
        Calibration.book == book,
        Calibration.scope == scope
    ).one_or_none()

    now_ts = datetime.now(timezone.utc)
    if row:
        row.alpha = float(alpha)
        row.beta = float(beta)
        row.n_train = int(n)
        row.logloss = float(logloss)
        row.created_at = now_ts
    else:
        db.add(Calibration(
            market=market,
            book=book,
            scope=scope,
            alpha=float(alpha),
            beta=float(beta),
            n_train=int(n),
            logloss=float(logloss),
            created_at=now_ts,
        ))
    db.commit()

    return {"ok": True, "market": market, "book": book, "scope": scope, "n_train": n, "alpha": alpha, "beta": beta, "logloss": logloss}

def _apply_row_calibration(db: Session, market: str, book: str, p: float, scope: str) -> float:
    """Apply (alpha,beta) for a scope; fallback to global; fallback to raw p."""
    if not (0.0 < p < 1.0):
        return _clip(p)

    for sc in ([scope] if scope != "global" else []) + ["global"]:
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
            z = row.alpha + row.beta * _logit(p)
            return _clip(_sigmoid(z))
    return _clip(p)

def apply_calibration_to_upcoming(
    db: Session,
    source_in: str,
    source_out: str,
    book: str,
    hours_ahead: int,
    comp_list: Optional[List[str]],
    scope: str,   # "global" or specific Fixture.comp
):
    """
    Read upcoming ModelProb (source_in), write calibrated probs to ModelProb (source_out).
    """
    now = datetime.now(timezone.utc)
    until = now + timedelta(hours=hours_ahead)

    # Pull probs + fixture info
    q = (
        db.query(
            ModelProb.id,
            ModelProb.fixture_id,
            ModelProb.market,
            ModelProb.prob,
            Fixture.comp,
            Fixture.kickoff_utc,
        )
        .join(Fixture, Fixture.id == ModelProb.fixture_id)
        .filter(ModelProb.source == source_in)
        .filter(Fixture.kickoff_utc >= now, Fixture.kickoff_utc <= until)
    )
    if comp_list:
        q = q.filter(Fixture.comp.in_(comp_list))

    rows = q.all()
    written = 0

    # Overwrite existing output probs for these fixtures to keep it clean
    fx_ids = list({r.fixture_id for r in rows})
    if fx_ids:
        db.query(ModelProb).filter(
            and_(ModelProb.source == source_out, ModelProb.fixture_id.in_(fx_ids))
        ).delete(synchronize_session=False)
        db.flush()

    for _id, fx_id, market, p, comp, ko in rows:
        try:
            pf = float(p)
        except Exception:
            continue
        if not (0.0 < pf < 1.0):
            continue

        sc = comp if scope != "global" else "global"
        pcal = _apply_row_calibration(db, market, book, pf, sc)

        db.add(ModelProb(
            fixture_id=fx_id,
            source=source_out,
            market=market,
            prob=pcal,
            as_of=now,
        ))
        written += 1

    db.commit()
    return {"ok": True, "source_in": source_in, "source_out": source_out, "hours_ahead": hours_ahead, "written": written}