import csv
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

from .db import Base
from .models import Fixture, Odds, Bet, Bankroll
from .ingest import upsert_fixture
from .crud import create_bet, settle_bet, get_latest_bankroll, update_bankroll

PGHOST=os.getenv("PGHOST","db")
PGPORT=os.getenv("PGPORT","5432")
PGUSER=os.getenv("PGUSER","vb_user")
PGPASSWORD=os.getenv("PGPASSWORD","vb_pass")
PGDATABASE=os.getenv("PGDATABASE","vb_db")

DATABASE_URL = f"postgresql://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def ensure_bankroll(db: Session, starting: float = 0.0):
    latest = get_latest_bankroll(db)
    if not latest:
        update_bankroll(db, starting)

def seed_from_csv(csv_path: str, starting_bankroll: float = 0.0):
    with SessionLocal() as db:
        ensure_bankroll(db, starting=starting_bankroll)

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pfid = row["provider_fixture_id"]
                comp = row["comp"]
                home = row["home_team"]
                away = row["away_team"]
                ko = datetime.fromisoformat(row["kickoff_utc"].replace("Z","+00:00"))
                market = row["market"]
                bookmaker = row["bookmaker"]
                price = float(row["price"])
                stake = float(row["stake"])
                result = row.get("result","PENDING").upper()
                ret = float(row.get("ret", "0") or 0)

                # ensure fixture exists
                fx = db.query(Fixture).filter(Fixture.provider_fixture_id == pfid).one_or_none()
                if not fx:
                    fx = upsert_fixture(db, pfid, comp, home, away, ko)

                # place bet
                b = create_bet(db, fx.id, market, bookmaker, price, stake)

                # settle if result supplied (WON/LOST/VOID)
                if result in ("WON","LOST","VOID"):
                    settle_bet(db, b.id, result, ret)

        print("Seeding complete")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m app.seed_bets /app/app/seed/bets_seed.csv [starting_bankroll]")
        sys.exit(1)
    csv_path = sys.argv[1]
    starting = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    seed_from_csv(csv_path, starting)