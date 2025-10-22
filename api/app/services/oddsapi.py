from __future__ import annotations
import os, requests
from datetime import datetime, timezone
from dateutil import parser as dtparse

BASE = "https://api.the-odds-api.com/v4"
API_KEY = os.getenv("ODDS_API_KEY")
REGION = os.getenv("ODDS_API_REGION", "uk")

def _get(path: str, **params):
    if not API_KEY:
        raise RuntimeError("ODDS_API_KEY missing")
    p = {"apiKey": API_KEY, "regions": REGION, **params}
    r = requests.get(f"{BASE}{path}", params=p, timeout=20)
    r.raise_for_status()
    return r.json()

def fetch_events(sport_key: str):
    # Example sport keys: "soccer_epl", "soccer_spain_la_liga", "soccer_uefa_champs"
    return _get(f"/sports/{sport_key}/odds", markets="h2h,totals,btts")

def parse_event(ev: dict):
    # Returns: (comp, home, away, kickoff_utc, bookmaker, prices[market])
    comp = ev.get("sport_key","").upper()
    commence = dtparse.isoparse(ev["commence_time"]).astimezone(timezone.utc)
    home, away = ev["home_team"], ev["away_team"]

    prices = {}  # { "O2.5": 1.83, "BTTS_YES": 1.72, ... }
    for book in ev.get("bookmakers", []):
        bm = book["title"].lower().replace(" ", "")
        # pick the first reputable book’s lines; you can dedupe later
        if bm not in ("bet365","paddypower","betfair","williamhill"): 
            continue
        for mk in book.get("markets", []):
            key = mk["key"]
            if key == "totals":  # over/under 2.5
                for o in mk["outcomes"]:
                    if o.get("point") == 2.5 and o["name"].lower() == "over":
                        prices["O2.5"] = (bm, float(o["price"]))
                    if o.get("point") == 2.5 and o["name"].lower() == "under":
                        prices["U2.5"] = (bm, float(o["price"]))
            elif key == "btts":
                for o in mk["outcomes"]:
                    if o["name"].lower() == "yes":
                        prices["BTTS_YES"] = (bm, float(o["price"]))
                    if o["name"].lower() == "no":
                        prices["BTTS_NO"] = (bm, float(o["price"]))
    return comp, home, away, commence, prices