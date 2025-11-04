from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func
from ..models import CreatorPick, ModelProb, Fixture

def settle_profit(result: str, stake: float, price: float) -> float:
    if result == "WIN":
        return stake * (price - 1.0)
    if result == "LOSE":
        return -stake
    return 0.0  # PUSH/None

def compute_creator_rolling_stats(db: Session, creator_id: int, days: int = 30) -> dict:
    since = datetime.utcnow() - timedelta(days=days)
    rows = (db.query(CreatorPick)
            .filter(CreatorPick.creator_id == creator_id,
                    CreatorPick.created_at >= since)
            .all())
    picks = len(rows)
    profit = sum((r.profit if r.result else 0.0) for r in rows)
    stake_sum = sum((r.stake or 0.0) for r in rows if r.result)
    roi = (profit / stake_sum) if stake_sum > 0 else 0.0
    wins = sum(1 for r in rows if r.result == "WIN")
    settled = sum(1 for r in rows if r.result in ("WIN","LOSE","PUSH"))
    winrate = (wins / settled) if settled > 0 else 0.0
    return {"picks": picks, "profit": profit, "roi": roi, "winrate": winrate}

def model_edge_for_pick(db: Session, fixture_id: int, market: str, price: float) -> float | None:
    # Look up latest model prob for that market on fixture
    mp = (db.query(ModelProb)
          .filter(ModelProb.fixture_id == fixture_id, ModelProb.market == market)
          .order_by(ModelProb.as_of.desc())
          .first())
    if not mp or not (0 < (mp.prob or 0) < 1):
        return None
    return (mp.prob * price) - 1.0