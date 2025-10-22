# api/app/db.py
import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Load .env locally; on Render you won't have one (that's fine)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

# Prefer a full DATABASE_URL (Render injects this). Fallback to individual parts for local dev.
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # Local fallback (what you had before via settings)
    from .settings import settings
    print(
        "DB CONFIG (local fallback) ->",
        f"user={settings.PGUSER} host={settings.PGHOST} port={settings.PGPORT} db={settings.PGDATABASE}"
    )
    DATABASE_URL = (
        f"postgresql://{settings.PGUSER}:{settings.PGPASSWORD}"
        f"@{settings.PGHOST}:{settings.PGPORT}/{settings.PGDATABASE}"
    )

# If it's a remote DB (not localhost), enforce SSL (Render Postgres requires it)
connect_args = {}
try:
    parsed = urlparse(DATABASE_URL)
    is_local = parsed.hostname in {"localhost", "127.0.0.1"} or parsed.hostname is None
except Exception:
    is_local = False

if not is_local:
    # psycopg2 uses sslmode
    connect_args["sslmode"] = "require"

engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()