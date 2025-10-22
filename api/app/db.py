# api/app/db.py
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")  # defensive load

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from .settings import settings

# Debug (safe) – confirm which creds are used on startup
print(
    "DB CONFIG ->",
    f"user={settings.PGUSER} host={settings.PGHOST} port={settings.PGPORT} db={settings.PGDATABASE}"
)

DATABASE_URL = (
    f"postgresql://{settings.PGUSER}:{settings.PGPASSWORD}"
    f"@{settings.PGHOST}:{settings.PGPORT}/{settings.PGDATABASE}"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()