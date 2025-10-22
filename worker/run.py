import time
from datetime import datetime, timezone
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import os

PGHOST=os.getenv("PGHOST","localhost"); PGPORT=os.getenv("PGPORT","5432")
PGUSER=os.getenv("PGUSER","postgres"); PGPASSWORD=os.getenv("PGPASSWORD","postgres")
PGDATABASE=os.getenv("PGDATABASE","postgres")

DATABASE_URL = f"postgresql://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# In a real worker: fetch new fixtures/odds regularly and recompute edges.
def tick():
    with engine.connect() as conn:
        now = datetime.now(timezone.utc).isoformat()
        conn.exec_driver_sql("select 1;")
        print(f"[worker] alive {now}")

if __name__ == "__main__":
    while True:
        try:
            tick()
        except Exception as e:
            print("worker error:", e)
        time.sleep(60)