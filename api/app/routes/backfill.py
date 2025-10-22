from __future__ import annotations
from datetime import datetime, timedelta, timezone, date
from typing import List, Dict

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Fixture
from ..services import apifootball

router = APIRouter(prefix="/backfill", tags=["backfill"])

def _dt_utc(d: date, t: str = "00:00") -> datetime:
    # helper: build a UTC-aware datetime from a date + "HH:MM"
    iso = f"{d.isoformat()}T{t}"
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt

@router.post("/soccer/dates")
def backfill_soccer_by_dates(
    days: str = Query(..., description="CSV of YYYY-MM-DD, e.g. 2025-08-17,2025-08-18"),
    leagues: str = Query("", description="Optional CSV of league keys (your SOCCER_LEAGUE_MAP keys); empty = all from provider for that day"),
    db: Session = Depends(get_db),
):
    """
    For each day given, pull fixtures from API-Football (by date) and upsert into `fixtures`.
    If a game is finished, also write full_time scores + result_settled=True.
    This does NOT fetch historic odds (providers rarely expose them), but prepares DB for analysis.
    """
    day_list = [datetime.fromisoformat(d.strip()).date() for d in days.split(",") if d.strip()]
    wanted_keys = [s.strip() for s in leagues.split(",") if s.strip()] if leagues else []

    # If you want to filter by provider league ids, we can map via apifootball.LEAGUE_MAP
    wanted_provider_ids = set(apifootball.LEAGUE_MAP[k] for k in wanted_keys if k in apifootball.LEAGUE_MAP)

    total_fixtures = 0
    updated_scores = 0
    upserts = 0

    for d in day_list:
        # raw list of fixtures from provider for that date
        raw = apifootball._get(apifootball.API_URL, {"date": d.isoformat()}) or []

        # optional filter to specific comps
        if wanted_provider_ids:
            raw = [fx for fx in raw if (fx.get("league") or {}).get("id") in wanted_provider_ids]

        for fx in raw:
            total_fixtures += 1

            # provider basics
            fixture = fx.get("fixture") or {}
            teams   = fx.get("teams") or {}
            league  = fx.get("league") or {}
            goals   = fx.get("goals") or {}

            provider_id = str(fixture.get("id") or "")
            comp = league.get("name") or ""
            home = (teams.get("home") or {}).get("name") or "Home"
            away = (teams.get("away") or {}).get("name") or "Away"

            # kickoff (API-Football gives ISO with timezone)
            dt_txt = fixture.get("date") or f"{d.isoformat()}T00:00:00+00:00"
            try:
                ko = datetime.fromisoformat(dt_txt.replace("Z", "+00:00"))
                if ko.tzinfo is None:
                    ko = ko.replace(tzinfo=timezone.utc)
                else:
                    ko = ko.astimezone(timezone.utc)
            except Exception:
                ko = _dt_utc(d, "00:00")

            # upsert fixture by provider_fixture_id
            f = db.query(Fixture).filter(Fixture.provider_fixture_id == provider_id).one_or_none()
            if f:
                f.comp = comp
                f.home_team = home
                f.away_team = away
                f.kickoff_utc = ko
            else:
                f = Fixture(
                    provider_fixture_id=provider_id,
                    comp=comp,
                    home_team=home,
                    away_team=away,
                    kickoff_utc=ko,
                )
                db.add(f)
                upserts += 1

            # if finished, write result fields
            status_short = ((fixture.get("status") or {}).get("short") or "").upper()
            if status_short in {"FT", "AET", "PEN"}:
                try:
                    fh = int(goals.get("home")) if goals.get("home") is not None else None
                    fa = int(goals.get("away")) if goals.get("away") is not None else None
                except Exception:
                    fh = fa = None

                # fallback from "score.fulltime"
                if fh is None or fa is None:
                    ft = (fx.get("score") or {}).get("fulltime") or {}
                    try:
                        fh = int(ft.get("home")) if ft.get("home") is not None else fh
                        fa = int(ft.get("away")) if ft.get("away") is not None else fa
                    except Exception:
                        pass

                if isinstance(fh, int) and isinstance(fa, int):
                    f.full_time_home = fh
                    f.full_time_away = fa
                    f.result_settled = True
                    updated_scores += 1

        db.commit()

    return {
        "ok": True,
        "days": [d.isoformat() for d in day_list],
        "filtered_to_leagues": wanted_keys,
        "provider_fixtures_seen": total_fixtures,
        "fixtures_upserted": upserts,
        "finished_with_scores_written": updated_scores,
    }