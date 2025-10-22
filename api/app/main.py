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

# Split route groups
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
    preview as preview_router,   # ✅ AI previews
)
from .services import league_strength

# --- App init ---
app = FastAPI(title="Value Betting API", version="0.2.0")

# --- CORS for frontend + live site ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://logies-edges-site.vercel.app",  # ✅ your live site
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Startup ---
@app.on_event("startup")
def startup():
    if os.getenv("ENV") != "production":
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
app.include_router(picks_router.pub)

# ✅ Public + AI Preview routes
app.include_router(public_router.pub, prefix="/api")
app.include_router(preview_router.router, prefix="/api")
app.include_router(preview_router.pub, prefix="/api")

# --- Debug route for visibility ---
@app.get("/debug/routes")
def list_routes():
    return [
        {
            "path": route.path,
            "name": route.name,
            "methods": list(route.methods),
        }
        for route in app.routes
        if isinstance(route, APIRoute)
    ]