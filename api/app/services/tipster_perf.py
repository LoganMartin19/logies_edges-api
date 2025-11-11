# api/app/services/tipster_perf.py
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from ..models import TipsterPick, ModelProb, Fixture

SETTLED_STATES = {"WIN", "LOSE", "PUSH", "VOID", "VOD"}  # treat VOID/VOD as push

# ---------- utils ----------

def _to_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        # handles Decimal/str/int/float
        return float(x)
    except Exception:
        return None

def _norm_result(s: Optional[str]) -> str:
    return (s or "").strip().upper()

def _canon_market(m: Optional[str]) -> str:
    if not m:
        return ""
    x = m.strip().upper().replace(" ", "").replace("-", "")
    if x.startswith("O") and x[1:].replace(".", "", 1).isdigit():
        return x
    if x.startswith("U") and x[1:].replace(".", "", 1).isdigit():
        return x
    syn = {
        "HOMEWIN": "HOME_WIN", "AWAYWIN": "AWAY_WIN",
        "BTTSYES": "BTTS_Y", "BTTSNO": "BTTS_N",
        "1": "HOME_WIN", "2": "AWAY_WIN", "X": "DRAW",
    }
    return syn.get(x, x)

def _settled_profit(result: str, stake: float, price: float) -> float:
    r = _norm_result(result)
    if r == "WIN":
        return stake * max(0.0, price - 1.0)
    if r == "LOSE":
        return -stake
    # PUSH / VOID / VOD -> 0
    return 0.0

# ---------- core stats ----------

def compute_tipster_rolling_stats(db: Session, tipster_id: int, days: int = 30) -> Dict[str, Any]:
    """
    Computes rolling stats over the last `days` by created_at.
    Only settled picks (WIN/LOSE/PUSH/VOID/VOD) affect ROI & winrate.
    """
    since = datetime.utcnow() - timedelta(days=days)

    rows: List[TipsterPick] = (
        db.query(TipsterPick)
          .filter(
              TipsterPick.tipster_id == tipster_id,
              TipsterPick.created_at >= since,
          ).all()
    )

    total_picks = len(rows)

    # settled only for perf
    settled: List[TipsterPick] = [r for r in rows if _norm_result(r.result) in SETTLED_STATES]

    # compute stake/profit from data (don’t trust stored profit if missing)
    stake_sum = 0.0
    profit_sum = 0.0
    wins = 0
    losses = 0

    for r in settled:
        stake = _to_float(getattr(r, "stake", 1.0)) or 1.0
        price = _to_float(getattr(r, "price", 0.0)) or 0.0
        res = _norm_result(getattr(r, "result", ""))

        stake_sum += stake
        profit_sum += _settled_profit(res, stake, price)
        if res == "WIN":
            wins += 1
        elif res == "LOSE":
            losses += 1

    settled_count = len(settled)
    roi = (profit_sum / stake_sum) if stake_sum > 0 else 0.0
    winrate = (wins / (wins + losses)) if (wins + losses) > 0 else 0.0

    return {
        "picks": total_picks,          # total created in window (incl. unsettled)
        "settled": settled_count,      # settled in window
        "wins": wins,
        "losses": losses,
        "profit": round(profit_sum, 4),
        "roi": round(roi, 4),          # 0.1234 => 12.34%
        "winrate": round(winrate, 4),  # 0.5625 => 56.25%
        "stake_units": round(stake_sum, 4),
    }

# ---------- model edge helper ----------

def model_edge_for_pick(db: Session, fixture_id: int, market: str, price) -> Optional[float]:
    """
    edge = prob * price - 1
    Uses most recent ModelProb regardless of source; add a `source` filter if you need.
    """
    mkt = _canon_market(market)
    mp = (
        db.query(ModelProb)
          .filter(ModelProb.fixture_id == fixture_id, ModelProb.market == mkt)
          .order_by(ModelProb.as_of.desc())
          .first()
    )
    if not mp:
        return None

    p = _to_float(mp.prob)
    q = _to_float(price)
    if p is None or q is None or not (0.0 < p < 1.0):
        return None

    return round(p * q - 1.0, 4)

# ---------- optional: leaderboard helper ----------

def get_tipster_leaderboard(db: Session, days: int = 30, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Returns simple leaderboard sorted by ROI then profit over the window.
    You can join Tipster table here if you want names/handles.
    """
    # Pull ids that had picks in window to limit work
    since = datetime.utcnow() - timedelta(days=days)
    ids = [tid for (tid,) in db.query(TipsterPick.tipster_id)
                                .filter(TipsterPick.created_at >= since)
                                .distinct().all()]

    out: List[Tuple[float, float, int, int]] = []  # (roi, profit, tipster_id, settled)
    rows: List[Dict[str, Any]] = []

    for tid in ids:
        s = compute_tipster_rolling_stats(db, tid, days=days)
        rows.append({"tipster_id": tid, **s})

    # sort: ROI desc, then profit desc, then settled desc
    rows.sort(key=lambda r: (r["roi"], r["profit"], r["settled"]), reverse=True)
    return rows[:limit]