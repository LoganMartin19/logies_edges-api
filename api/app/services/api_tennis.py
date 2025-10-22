from __future__ import annotations
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import requests
import os

API_BASE = os.getenv("TENNIS_API_BASE", "https://your-tennis-api.example.com")
API_KEY  = os.getenv("TENNIS_API_KEY", "REPLACE_ME")

LAST_HTTP: Dict[str, Any] = {}

def _get(path: str, params: dict) -> Any:
    url = f"{API_BASE.rstrip('/')}/{path.lstrip('/')}"
    hdr = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    r = requests.get(url, params=params, headers=hdr, timeout=30)
    LAST_HTTP.update({"url": r.url, "status": r.status_code})
    r.raise_for_status()
    return r.json()

def fetch_fixtures_by_date(day_iso: str) -> List[Dict[str, Any]]:
    """
    Expected provider response per match (example):
    {
      "id": 12345,
      "tour": {"name": "ATP Basel", "country": "Switzerland"},
      "scheduled_utc": "2025-10-18T12:30:00Z",
      "player_a": {"id": 111, "name": "Alcaraz C."},
      "player_b": {"id": 222, "name": "Sinner J."}
    }
    """
    return _get("/tennis/fixtures", {"date": day_iso}) or []

def fetch_odds_for_fixture(provider_fixture_id: int) -> List[Dict[str, Any]]:
    """
    Expected odds rows:
    [{"book":"bet365","market":"PLAYER_A","price":1.80},
     {"book":"bet365","market":"PLAYER_B","price":2.05}, ...]
    """
    return _get(f"/tennis/odds/{provider_fixture_id}", {})