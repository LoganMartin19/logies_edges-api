from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func
from ..models import TipsterPick, ModelProb, Fixture

def settle_profit(result: str, stake: float, price: float) -> float:
    if result == "WIN":
        return stake * (price - 1.0)
    if result == "LOSE":
        return -stake
    return 0.0  # PUSH/None

def compute_tipster_rolling_stats(db: Session, tipster_id: int, days: int = 30) -> dict:
    since = datetime.utcnow() - timedelta(days=days)
    rows = (db.query(TipsterPick)
            .filter(TipsterPick.tipster_id == tipster_id,
                    TipsterPick.created_at >= since)
            .all())
    picks = len(rows)
    profit = sum((r.profit if r.result else 0.0) for r in rows)
    stake_sum = sum((r.stake or 0.0) for r in rows if r.result)
    roi = (profit / stake_sum) if stake_sum > 0 else 0.0
    wins = sum(1 for r in rows if r.result == "WIN")
    settled = sum(1 for r in rows if r.result in ("WIN","LOSE","PUSH"))
    winrate = (wins / settled) if settled > 0 else 0.0
    return {"picks": picks, "profit": profit, "roi": roi, "winrate": winrate}

# api/app/services/tipster_perf.py

from decimal import Decimal
from typing import Optional

def _to_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        # handles Decimal, str, int, float
        return float(x)
    except Exception:
        return None

def model_edge_for_pick(db: Session, fixture_id: int, market: str, price) -> float | None:
    """
    edge = prob * price - 1
    Cast Decimals to float to avoid Decimal*float TypeError.
    """
    mp = (db.query(ModelProb)
          .filter(ModelProb.fixture_id == fixture_id, ModelProb.market == market)
          .order_by(ModelProb.as_of.desc())
          .first())
    if not mp:
        return None

    p = _to_float(mp.prob)
    q = _to_float(price)
    if p is None or q is None or not (0 < p < 1):
        return None

    return round(p * q - 1.0, 4)