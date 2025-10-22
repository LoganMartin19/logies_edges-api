from sqlalchemy.orm import Session
from datetime import datetime
from . import models

# -------- Fixtures --------
def get_fixture(db: Session, fixture_id: int):
    return db.query(models.Fixture).filter(models.Fixture.id == fixture_id).first()

def get_fixtures(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Fixture).offset(skip).limit(limit).all()

def create_fixture(db: Session, pfid: str, comp: str, home: str, away: str, ko: datetime, country: str = ""):
    f = models.Fixture(
        provider_fixture_id=pfid,
        comp=comp,
        home_team=home,
        away_team=away,
        kickoff_utc=ko,
        country=country,
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    return f

# -------- Odds --------
def get_odds_for_fixture(db: Session, fixture_id: int):
    return db.query(models.Odds).filter(models.Odds.fixture_id == fixture_id).all()

def upsert_odds(db: Session, fixture_id: int, bookmaker: str, market: str, price: float):
    o = (
        db.query(models.Odds)
        .filter_by(fixture_id=fixture_id, bookmaker=bookmaker, market=market)
        .one_or_none()
    )
    if o:
        o.price = price
        o.last_seen = datetime.utcnow()
    else:
        o = models.Odds(
            fixture_id=fixture_id,
            bookmaker=bookmaker,
            market=market,
            price=price,
        )
        db.add(o)
    db.commit()
    return o

# -------- Model Probabilities --------
def set_model_prob(db: Session, fixture_id: int, source: str, market: str, prob: float):
    p = (
        db.query(models.ModelProb)
        .filter_by(fixture_id=fixture_id, source=source, market=market)
        .one_or_none()
    )
    if p:
        p.prob = prob
        p.as_of = datetime.utcnow()
    else:
        p = models.ModelProb(
            fixture_id=fixture_id,
            source=source,
            market=market,
            prob=prob,
        )
        db.add(p)
    db.commit()
    return p

def get_model_probs(db: Session, fixture_id: int):
    return db.query(models.ModelProb).filter(models.ModelProb.fixture_id == fixture_id).all()

# -------- Edges --------
def get_edges_for_fixture(db: Session, fixture_id: int):
    return db.query(models.Edge).filter(models.Edge.fixture_id == fixture_id).all()

# -------- Bets --------
def create_bet(db: Session, fixture_id: int, market: str, bookmaker: str, price: float, stake: float):
    b = models.Bet(
        fixture_id=fixture_id,
        market=market,
        bookmaker=bookmaker,
        price=price,
        stake=stake,
    )
    db.add(b)
    db.commit()
    db.refresh(b)
    return b

def settle_bet(db: Session, bet_id: int, result: str, ret: float):
    b = db.query(models.Bet).filter(models.Bet.id == bet_id).first()
    if not b:
        return None
    b.result = result
    b.ret = ret
    b.pnl = ret - b.stake
    db.commit()
    return b

# -------- Bankroll --------
def get_latest_bankroll(db: Session):
    return db.query(models.Bankroll).order_by(models.Bankroll.as_of.desc()).first()

def update_bankroll(db: Session, balance: float):
    br = models.Bankroll(balance=balance)
    db.add(br)
    db.commit()
    return br