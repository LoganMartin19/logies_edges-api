# api/app/main.py
from __future__ import annotations

import os
from datetime import datetime, timezone
from fastapi import FastAPI, Depends
from fastapi.routing import APIRoute
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

from .db import Base, engine, get_db
from .edge import ensure_baseline_probs, compute_edges
from .settings import settings

# --- Routers / modules ---
from .routers import bets as bets_router
from .routers import ingest as ingest_router
from .routers import fixtures as fixtures_router
from .routers import basketball as basketball_router
from .routers.football_admin import admin as football_admin_router
from .routers import auth as auth_router
from .routers import tipsters as tipsters_router
from .routers import billing as billing_router 
from .routers import tipster_subscriptions as tipster_subscriptions_router
from .routers import notifications as notifications_router

from .routes import (
    pages as pages_router,
    admin_ops as admin_ops_router,
    diagnostics as diagnostics_router,
    shortlist as shortlist_router,
    historic as historic_router,
    backfill as backfill_router,
    performance as performance_router,
    admin_calibration as admin_calibration_router,
    form as form_router,
    explain as explain_router,
    poll as poll_router,
    football_extra as football_extra_router,
    player_props as player_props_router,
    tennis as tennis_router,
    picks as picks_router,
    public as public_router,
    preview as preview_router,              # ⚽ sport-specific (football) private/admin
    accas as accas_router,
    user_bets as user_bets_router,
    email_admin as email_admin_router,
    preferences as preferences_router,
)

# --- NEW: sport-aware preview dispatch + other sports (NFL/NHL) ---
from .routes.preview_dispatch import router as preview_dispatch_router, pub as preview_dispatch_pub
from .routes.preview_gridiron import router as preview_gridiron_router
from .routes.preview_hockey import router as preview_hockey_router

from .services import league_strength

# --- App init ---
app = FastAPI(title="Value Betting API", version="0.3.0")

# --- CORS for frontend + live site ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://logies-edges-site.vercel.app",
        "https://logies-edges-dashboard.vercel.app",
        "https://charteredsportsbetting.com",
        "https://www.charteredsportsbetting.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# If you also want all Vercel preview URLs, uncomment:
# app.add_middleware(
#     CORSMiddleware,
#     allow_origin_regex=r"https://.*\.vercel\.app$",
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# --- Health Check ---
@app.get("/health")
def health():
    """Lightweight health check for Render."""
    return {"ok": True}

@app.get("/api/health")
def api_health():
    """Mirror endpoint for dashboard/API checks."""
    return {"ok": True}

# --- Startup ---
@app.on_event("startup")
def startup():
    # Only auto-create tables locally; use Alembic in production
    env = os.getenv("ENV", "local")
    if env != "production":
        Base.metadata.create_all(bind=engine)

# --- Optional compute route ---
@app.post("/compute")
def compute(db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc)
    ensure_baseline_probs(db, now, source="team_form")
    compute_edges(db, now, settings.EDGE_MIN, source="team_form")
    return {"ok": True}

# --- Include routers ---
app.include_router(bets_router.router)
app.include_router(ingest_router.router)
app.include_router(fixtures_router.router)
app.include_router(basketball_router.router)
app.include_router(football_admin_router)
app.include_router(auth_router.router)
app.include_router(billing_router.router, prefix="/api")
app.include_router(tipsters_router.router)
app.include_router(tipster_subscriptions_router.router)
app.include_router(notifications_router.router)


# Split routes
app.include_router(pages_router.router)
app.include_router(admin_ops_router.router)
app.include_router(diagnostics_router.router)
app.include_router(shortlist_router.router)
app.include_router(historic_router.router)
app.include_router(backfill_router.router)
app.include_router(performance_router.router)
app.include_router(admin_calibration_router.router)
app.include_router(form_router.router)
app.include_router(league_strength.router)
app.include_router(explain_router.router)
app.include_router(poll_router.router)
app.include_router(football_extra_router.router)
app.include_router(player_props_router.router)
app.include_router(tennis_router.router)
app.include_router(picks_router.router)
app.include_router(accas_router.router)
app.include_router(email_admin_router.router, prefix="/api")
app.include_router(preferences_router.router, prefix="/api")

# ✅ Public + AI Preview routes
app.include_router(public_router.pub, prefix="/api")                 # existing public pages
# Keep football-specific private/admin preview routes available:
app.include_router(preview_router.router, prefix="/api")             # /ai/preview (football-only, private)
# ⛔️ Do NOT include preview_router.pub here to avoid path clash
# app.include_router(preview_router.pub, prefix="/api")              # (removed; superseded by dispatcher)

# NEW: include sport modules (private/admin endpoints for nfl/nhl, if any)
app.include_router(preview_gridiron_router, prefix="/api")           # /ai/preview/gridiron/*
app.include_router(preview_hockey_router, prefix="/api")             # /ai/preview/hockey/*

# NEW: include dispatcher (auto routes + the unified public endpoint)
app.include_router(preview_dispatch_router, prefix="/api")           # /ai/preview/generate/auto etc.
app.include_router(preview_dispatch_pub, prefix="/api")              # /public/ai/preview/by-fixture (sport-agnostic)

# --- Public: Featured Picks + ACCA ---
from .routes.picks import pub as picks_public_router
from .routes.accas import pub as accas_public_router

app.include_router(picks_public_router, prefix="/api")   # keep this (its pub prefix starts with /public/...)
app.include_router(accas_public_router)                  # <-- remove prefix to avoid /api/api/...
app.include_router(user_bets_router.router, prefix="/api")

# --- Debug route for visibility ---
@app.get("/debug/routes")
def list_routes():
    """List all registered API routes."""
    return [
        {
            "path": route.path,
            "name": route.name,
            "methods": list(route.methods),
        }
        for route in app.routes
        if isinstance(route, APIRoute)
    ]