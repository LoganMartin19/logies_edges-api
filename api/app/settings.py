# api/app/settings.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Always load the repo-root .env (…/logies_edges/.env), regardless of CWD
ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ROOT_ENV)

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


class Settings:
    # ----------------------------------------------------------------------
    # Database
    # ----------------------------------------------------------------------
    PGHOST = os.getenv("PGHOST", "localhost")
    PGPORT = int(os.getenv("PGPORT", "5432"))
    PGUSER = os.getenv("PGUSER", "vb_user")
    PGPASSWORD = os.getenv("PGPASSWORD", "vb_pass")
    PGDATABASE = os.getenv("PGDATABASE", "vb_db")

    # ----------------------------------------------------------------------
    # Model config
    # ----------------------------------------------------------------------
    EDGE_MIN = float(os.getenv("EDGE_MIN", "0.05"))

    # ----------------------------------------------------------------------
    # API keys
    # ----------------------------------------------------------------------
    API_FOOTBALL_KEY = os.getenv("FOOTBALL_API_KEY")
    API_NFL_KEY = os.getenv("GRIDIRON_API_KEY")

    # ----------------------------------------------------------------------
    # Stripe config (NEW)
    # ----------------------------------------------------------------------
    STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
    STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    
    # IMPORTANT: match the name you already added in Render/.env
    STRIPE_PREMIUM_PRICE_ID_PREMIUM = os.getenv("STRIPE_PREMIUM_PRICE_ID", "")
    
    # You can add additional pricing tiers later:
    # STRIPE_PRICE_ID_TIPSTER_MONTHLY = os.getenv("STRIPE_PRICE_ID_TIPSTER_MONTHLY", "")
    # STRIPE_PRICE_ID_TIPSTER_YEARLY = os.getenv("STRIPE_PRICE_ID_TIPSTER_YEARLY", "")


settings = Settings()