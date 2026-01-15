from __future__ import annotations

from datetime import datetime, timedelta, timezone, date
import re
import json
import time
import os
import requests
from hashlib import sha256
from typing import List, Dict

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import and_,func, or_

from ..db import get_db
from ..models import Fixture, Edge, Odds, ModelProb, Bet, ClosingOdds, LeagueStanding
from ..edge import ensure_baseline_probs, compute_edges, MODEL_SOURCE
from ..settings import settings
from ..services import apifootball
from ..telegram_alert import send_telegram_alert

# soccer + gridiron + ice ingestion helpers
from ..services.fixtures import ingest_fixtures_and_odds as ingest_soccer
from ..services.fixtures_gridiron import ingest_gridiron_and_odds as ingest_gridiron
from ..services.fixtures_ice import ingest_ice_and_odds as ingest_ice
from ..services.fixtures_nba import ingest_nba_and_odds   # ✅ new import
from ..services.fixtures import ingest_odds_for_fixture_id as refresh_soccer_fixture

# league maps for partitioning
from ..services.apifootball import LEAGUE_MAP as SOCCER_LEAGUE_MAP, get_standings_for_league
from ..services.api_gridiron import LEAGUE_MAP as GRIDIRON_LEAGUE_MAP
from ..services.api_ice import LEAGUE_MAP as ICE_LEAGUE_MAP  # ⬅️ NHL keys
from ..services.apifootball import fetch_odds_for_fixture

router = APIRouter(prefix="/admin", tags=["admin"])

LEAGUE_KEY_FILTER = {
    "EPL": ("Premier League", "England"),
    "CHAMP": ("Championship", "England"),
    "LG1": ("League One", "England"),
    "LG2": ("League Two", "England"),
    "SCO_PREM": ("Premiership", "Scotland"),
    "SCO_CHAMP": ("Championship", "Scotland"),
    "SCO1": ("League One", "Scotland"),
    "SCO2": ("League Two", "Scotland"),
    "SCO_CUP": ("Scottish Cup", "Scotland"),
    "LA_LIGA": ("La Liga", "Spain"),
    "BUNDES": ("Bundesliga", "Germany"),
    "BUNDES2": ("2. Bundesliga", "Germany"),
    "SERIE_A": ("Serie A", "Italy"),
    "SERIE_B": ("Serie B", "Italy"),
    "LIGUE1": ("Ligue 1", "France"),
    "UCL": ("UEFA Champions League", "World"),
    "UEL": ("UEFA Europa League", "World"),
    "UECL": ("UEFA Europa Conference League", "World"),
    "WCQ_EUR": ("World Cup - Qualification Europe", "World"),
}

def _partition_leagues(leagues_csv: str) -> Dict[str, List[str]]:
    """
    Split a CSV of league keys into sport buckets.
    - soccer:   keys in SOCCER_LEAGUE_MAP
    - gridiron: keys in GRIDIRON_LEAGUE_MAP (NFL, CFB)
    - ice:      keys in ICE_LEAGUE_MAP (NHL, ...)
    - nba:      "NBA"
    Unknown keys are ignored.
    """
    # normalize and keep original token for returning
    raw = [s.strip() for s in (leagues_csv or "").split(",") if s.strip()]
    keys_up = [s.upper() for s in raw]

    soccer_keys   = [raw[i] for i, k in enumerate(keys_up) if k in SOCCER_LEAGUE_MAP]
    gridiron_keys = [raw[i] for i, k in enumerate(keys_up) if k in GRIDIRON_LEAGUE_MAP]
    ice_keys      = [raw[i] for i, k in enumerate(keys_up) if k in ICE_LEAGUE_MAP]

    # NBA: simple allow-list (API uses one top tier)
    NBA_KEYS = {"NBA"}
    nba_keys = [raw[i] for i, k in enumerate(keys_up) if k in NBA_KEYS]

    return {
        "soccer": soccer_keys,
        "gridiron": gridiron_keys,
        "ice": ice_keys,
        "nba": nba_keys,
    }

@router.post("/run-daily")
def run_daily(
    db: Session = Depends(get_db),
    leagues: str = Query(
        "EPL,CHAMP,LG1,LG2,SCO_PREM,SCO_CHAMP,SCO1,SCO2,SCO_CHAL,"
        "LA_LIGA,BUNDES,BUNDES2,SERIE_A,SERIE_B,LIGUE1,"
        "UCL,UEL,UECL,WCQ_EUR,MLS,NFL,NHL"  # ⬅️ added MLS here (was in other endpoints) + NHL
    ),
    days: int = Query(2, ge=1, le=3),
    max_fixtures: int = Query(500, ge=10, le=1500),
    odds_delay_sec: float = Query(0.25, ge=0.0, le=2.0),
    nfl_uk_only: int = Query(0),
    cfb_uk_only: int = Query(1),
):
    today = datetime.now(timezone.utc).date()
    parts = _partition_leagues(leagues)
    ingested = []

    for d in range(days):
        day = today + timedelta(days=d)

        if parts["soccer"]:
            ingested.append(
                ingest_soccer(
                    db, day, parts["soccer"], prefer_book=None,
                    max_fixtures=max_fixtures, odds_delay_sec=odds_delay_sec
                )
            )

        if parts["gridiron"]:
            ingested.append(
                ingest_gridiron(
                    db, day, parts["gridiron"], prefer_book=None,
                    max_fixtures=max_fixtures, odds_delay_sec=odds_delay_sec,
                    require_uk_books_for_cfb=bool(cfb_uk_only),
                )
            )

        if parts["ice"]:
            ingested.append(
                ingest_ice(
                    db, day, parts["ice"], prefer_book=None,
                    max_games=None, odds_delay_sec=odds_delay_sec
                )
            )

    now = datetime.now(timezone.utc)
    ensure_baseline_probs(db, now, source="team_form")
    compute_edges(db, now, settings.EDGE_MIN, source="team_form")

    return {"ok": True, "ingested": ingested, "computed": True}

@router.post("/recompute")
def admin_recompute(
    db: Session = Depends(get_db),
    min_edge: float = Query(0.00),
):
    now = datetime.now(timezone.utc)
    ensure_baseline_probs(db, now, source="team_form")
    compute_edges(db, now, min_edge=min_edge, source="team_form")
    edges_ct = db.query(func.count()).select_from(Edge).scalar()
    probs_ct = db.query(func.count()).select_from(ModelProb).scalar()
    return {"ok": True, "edges": edges_ct, "model_probs": probs_ct}

@router.post("/refresh/{fixture_id}")
def admin_refresh_fixture(
    fixture_id: int,
    db: Session = Depends(get_db),
    compute: int = 1,
    prefer_book: str | None = Query(None),
    allow_fallback: int = Query(1),
):
    f = db.query(Fixture).filter(Fixture.id == fixture_id).one_or_none()
    if not f:
        raise HTTPException(404, "Fixture not found")

    comp_upper = (f.comp or "").upper()
    updated = {}

    # Only soccer supports per-fixture odds refresh via API-Football
    if not any(k in comp_upper for k in ("NFL", "CFB", "NCAA", "AMERICAN", "NHL", "ICE", "HOCKEY")):
        try:
            updated = refresh_soccer_fixture(
                db,
                provider_fixture_id=f.provider_fixture_id,
                prefer_book=prefer_book or "bet365",
                allow_fallback=bool(allow_fallback),
            )
        except Exception as e:
            updated = {"fixture_found": True, "error": repr(e)}
    else:
        updated = {"note": "Non-soccer fixture — use the daily/range ingest endpoints for fresh odds."}

    if compute:
        now = datetime.now(timezone.utc)
        ensure_baseline_probs(db, now, source="team_form")
        compute_edges(db, now, settings.EDGE_MIN, source="team_form")

    return {"ok": True, "fixture_id": fixture_id, "updated": updated}

@router.post("/run-dates")
def admin_run_dates(
    db: Session = Depends(get_db),
    days: str = Query(..., description="CSV of YYYY-MM-DD"),
    leagues: str = Query(
        # England + Cups
        "EPL,CHAMP,LG1,LG2,ENG_FA,EFL_CUP,EFL_TROPHY,NAT_LEAGUE,NAT_NORTH,NAT_SOUTH,"
        # Scotland + Cups
        "SCO_PREM,SCO_CHAMP,SCO1,SCO2,SCO_SC,SCO_LC,SCO_CHAL,"
        # Spain + Cup
        "LA_LIGA,LA_LIGA2,ESP_CDR,"
        # Germany + Cup
        "BUNDES,BUNDES2,GER_POKAL,"
        # Italy + Cup
        "SERIE_A,SERIE_B,ITA_COPPA,"
        # France + Cup
        "LIGUE1,LIGUE2,FRA_CDF,"
        # Portugal + Cup
        "POR_LIGA,POR_TACA,"
        # Netherlands + Cup
        "NED_ERED,NED_EERST,NED_KNVB,"
        # Belgium + Cup
        "BEL_PRO,BEL_CUP,"
        # Norway + Cup
        "NOR_ELI,NOR_CUP,"
        # Denmark + Cup
        "DEN_SL,DEN_CUP,"
        # Sweden + Cup
        "SWE_ALLS,SWE_SUPER,SWE_CUP,"
        # Argentina + Cups
        "ARG_LP,ARG_CDL,ARG_CUP,"
        # Brazil
        "BR_SERIE_A,BR_SERIE_B,"
        # USA
        "MLS,"
        # Europe comps
        "UCL,UEL,UECL,UWCL,WCQ_EUR,AFCON,"
        #Rest of the World
        "AUS_A_LEAGUE,"
        # Other sports you already run through here
        "NFL,NCAA,NHL,NBA",
        description="CSV of league keys to ingest"
    ),
    prefer_book: str = Query(None),
    compute_after: int = Query(1),
    cfb_uk_only: int = Query(1),
):
    # --- parse day list ---
    day_list = [
        datetime.fromisoformat(d.strip()).date()
        for d in (days or "").split(",")
        if d.strip()
    ]
    if not day_list:
        return {"ok": False, "error": "No valid days provided"}

    parts = _partition_leagues(leagues)
    ingested = []

    # --- ingest per day / league group ---
    for d in day_list:
        if parts["soccer"]:
            ingested.append(
                ingest_soccer(db, d, parts["soccer"], prefer_book=prefer_book)
            )
        if parts["gridiron"]:
            ingested.append(
                ingest_gridiron(
                    db, d, parts["gridiron"],
                    prefer_book=prefer_book,
                    require_uk_books_for_cfb=bool(cfb_uk_only),
                )
            )
        if parts["ice"]:
            ingested.append(
                ingest_ice(
                    db, d, parts["ice"],
                    prefer_book=prefer_book,
                    max_games=None,
                    odds_delay_sec=0.30,
                )
            )
        if parts["nba"]:
            ingested.append(
                ingest_nba_and_odds(
                    db, d, parts["nba"],
                    prefer_book=prefer_book,
                    max_games=None,
                    odds_delay_sec=0.35,
                )
            )

    computed = False

    if compute_after:
        now = datetime.now(timezone.utc)

        # --- restrict modelprob + edges to just around these days ---
        window_start_date = min(day_list) - timedelta(days=1)
        window_end_date = max(day_list) + timedelta(days=1)

        window_start = datetime.combine(
            window_start_date, datetime.min.time(), tzinfo=timezone.utc
        )
        window_end = datetime.combine(
            window_end_date, datetime.max.time(), tzinfo=timezone.utc
        )

        # soccer comps for team_form – adjust if you want others
        soccer_leagues = list(parts["soccer"]) if parts["soccer"] else None

        # 1) ensure baseline model probabilities only for this window
        ensure_baseline_probs(
            db=db,
            now=now,
            source="team_form",
            time_window=(window_start, window_end),
            league_comp_filter=soccer_leagues,
            use_closing_when_past=True,
        )

        # 2) compute edges only for fixtures from this window onwards
        compute_edges(
            db=db,
            now=now,
            since=window_start,            # ✅ proper datetime window
            min_edge=settings.EDGE_MIN,    # ✅ passed as the correct param
            source="team_form",
            # you can also override hours_ahead / staleness_hours here if desired:
            # hours_ahead=HOURS_AHEAD,
            # staleness_hours=STALE_ODDS_HOURS,
        )

        computed = True

    return {"ok": True, "ingested": ingested, "computed": computed}

@router.post("/run-range")
def admin_run_range(
    db: Session = Depends(get_db),
    start_day: str = Query(..., description="YYYY-MM-DD"),
    ndays: int = Query(2, ge=1, le=7),
    leagues: str = Query(
        "EPL,CHAMP,LG1,LG2,SCO_PREM,SCO_CHAMP,SCO1,SCO2,LA_LIGA,BUNDES,BUNDES2,SERIE_A,SERIE_B,LIGUE1,UCL,UEL,UECL,MLS,NFL,CFB,NHL"
    ),
    prefer_book: str = Query("bet365"),
    compute_after: int = Query(1),
    cfb_uk_only: int = Query(1),
):
    start = datetime.fromisoformat(start_day).date()
    parts = _partition_leagues(leagues)

    ingested = []
    for i in range(ndays):
        day = start + timedelta(days=i)
        if parts["soccer"]:
            ingested.append(ingest_soccer(db, day, parts["soccer"], prefer_book=prefer_book))
        if parts["gridiron"]:
            ingested.append(
                ingest_gridiron(db, day, parts["gridiron"], prefer_book=prefer_book,
                                require_uk_books_for_cfb=bool(cfb_uk_only))
            )
        if parts["ice"]:
            ingested.append(
                ingest_ice(db, day, parts["ice"], prefer_book=prefer_book, max_games=None, odds_delay_sec=0.30)
            )

    if compute_after:
        now = datetime.now(timezone.utc)
        ensure_baseline_probs(db, now, source="team_form")
        compute_edges(db, now, settings.EDGE_MIN, source="team_form")

    return {"ok": True, "ingested": ingested, "computed": bool(compute_after)}

# Housekeeping
@router.post("/clear-edges")
def clear_edges(db: Session = Depends(get_db)):
    n = db.query(Edge).delete(synchronize_session=False)
    db.commit()
    return {"deleted_edges": n}

@router.post("/keep-book")
def keep_book(book: str = Query(..., min_length=2), db: Session = Depends(get_db)):
    del_odds = db.query(Odds).filter(Odds.bookmaker != book).delete(synchronize_session=False)
    del_edges = db.query(Edge).filter(Edge.bookmaker != book).delete(synchronize_session=False)
    db.commit()
    return {"kept_book": book, "deleted_odds": del_odds, "deleted_edges": del_edges}

@router.post("/clear-odds")
def admin_clear_odds(db: Session = Depends(get_db)):
    n = db.query(Odds).delete(synchronize_session=False)
    db.commit()
    return {"deleted_odds": n}

@router.post("/full-reset")
def admin_full_reset(
    day: str | None = Query(None, description="YYYY-MM-DD"),
    leagues: str | None = Query(None, description="CSV e.g. EPL,CHAMP,UCL,NFL,CFB"),
    compute_after: int = 1,
    include_fixtures: int = Query(1),
    db: Session = Depends(get_db),
):
    out = {}
    out["deleted_edges"] = db.query(Edge).delete(synchronize_session=False)
    out["deleted_odds"]  = db.query(Odds).delete(synchronize_session=False)
    out["deleted_probs"] = db.query(ModelProb).delete(synchronize_session=False)
    if include_fixtures:
        out["deleted_fixtures"] = db.query(Fixture).delete(synchronize_session=False)
    db.commit()

    if day and leagues:
        d = datetime.fromisoformat(day).date()
        parts = _partition_leagues(leagues)
        ing = []
        if parts["soccer"]:
            ing.append(ingest_soccer(db, d, parts["soccer"], prefer_book="bet365"))
        if parts["gridiron"]:
            ing.append(ingest_gridiron(db, d, parts["gridiron"], prefer_book="bet365"))
        if parts["ice"]:
            ing.append(ingest_ice(db, d, parts["ice"], prefer_book="bet365"))
        out["ingest"] = ing

    if compute_after:
        ts = datetime.now(timezone.utc)
        ensure_baseline_probs(db, ts, source="team_form")
        compute_edges(db, ts, settings.EDGE_MIN, source="team_form")
        out["computed"] = True
    return out

@router.post("/cleanup-past")
def cleanup_past_fixtures(
    db: Session = Depends(get_db),
    older_than_days: int = Query(30, ge=1, description="Only delete fixtures older than this many days"),
    require_no_odds: int = Query(1, description="1=only delete fixtures with ZERO odds rows"),
    only_unsettled: int = Query(0, description="1=only delete fixtures that are not result_settled"),
    exclude_top_leagues: int = Query(1, description="1=protect curated top leagues"),
    dry_run: int = Query(1, description="1=show counts only; 0=actually delete"),
):
    from ..services.apifootball import LEAGUE_MAP as TOP  # protect these comps if exclude_top_leagues
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=older_than_days)

    q = (
        db.query(Fixture)
        .filter(Fixture.kickoff_utc < cutoff)
        .outerjoin(Bet, Fixture.id == Bet.fixture_id)
        .filter(Bet.id == None)  # no placed bets
    )

    if only_unsettled:
        q = q.filter((Fixture.result_settled == None) | (Fixture.result_settled == False))

    if exclude_top_leagues:
        # protect by comp name if you saved provider league names, otherwise use your keys if stored
        protected_names = [
            "Premier League","Championship","League One","League Two",
            "Scottish Premiership","Scottish Championship","Scottish League One","Scottish League Two",
            "La Liga","Bundesliga","2. Bundesliga","Serie A","Serie B","Ligue 1",
            "UEFA Champions League","UEFA Europa League","UEFA Europa Conference League",
            "World Cup - Qualification Europe"
        ]
        q = q.filter(~Fixture.comp.in_(protected_names))

    candidates = q.all()
    ids = [f.id for f in candidates]

    # optionally require zero odds rows
    if require_no_odds and ids:
        have_odds = set(
            r[0] for r in db.query(Odds.fixture_id).filter(Odds.fixture_id.in_(ids)).distinct().all()
        )
        ids = [fid for fid in ids if fid not in have_odds]
        candidates = [f for f in candidates if f.id in ids]

    if dry_run:
        return {
            "dry_run": True,
            "older_than_days": older_than_days,
            "require_no_odds": bool(require_no_odds),
            "only_unsettled": bool(only_unsettled),
            "exclude_top_leagues": bool(exclude_top_leagues),
            "would_delete": len(ids),
            "example_ids": ids[:25],
        }

    if not ids:
        return {"message": "Nothing to delete under current filters.", "deleted": 0}

    del_model_probs = db.query(ModelProb).filter(ModelProb.fixture_id.in_(ids)).delete(synchronize_session=False)
    del_odds = db.query(Odds).filter(Odds.fixture_id.in_(ids)).delete(synchronize_session=False)
    del_edges = db.query(Edge).filter(Edge.fixture_id.in_(ids)).delete(synchronize_session=False)
    del_fixtures = db.query(Fixture).filter(Fixture.id.in_(ids)).delete(synchronize_session=False)
    db.commit()
    return {
        "message": "Old fixtures cleaned.",
        "fixtures_deleted": len(ids),
        "model_probs_deleted": del_model_probs,
        "odds_deleted": del_odds,
        "edges_deleted": del_edges
    }

@router.post("/drop-create")
def admin_drop_create(recompute: int = 0, db: Session = Depends(get_db)):
    from ..db import Base, engine
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    out = {"dropped": True, "created": True}
    if recompute:
        ts = datetime.now(timezone.utc)
        ensure_baseline_probs(db, ts, source="team_form")
        compute_edges(db, ts, settings.EDGE_MIN, source="team_form")
        out["recomputed"] = True
    return out

# Alerts & Telegram helpers
@router.get("/alerts/why-not")
def admin_alerts_why_not(
    fixture_id: int = Query(...),
    market: str = Query(...),
    bookmaker: str = Query(...),
    price: float = Query(...),
    db: Session = Depends(get_db),
):
    alert_hash_input = f"{fixture_id}|{market}|{bookmaker}|{price:.4f}"
    digest = sha256(alert_hash_input.encode()).hexdigest()
    exists = db.query(Bet).filter(Bet.duplicate_alert_hash == digest).first()
    return {
        "fixture_id": fixture_id,
        "market": market,
        "bookmaker": bookmaker,
        "price": f"{price:.4f}",
        "hash": digest,
        "already_alerted": bool(exists),
        "marker_bet_id": getattr(exists, "id", None),
        "marker_bet_stake": getattr(exists, "stake", None),
        "hint": "If already_alerted is true, delete the marker or change the alert key."
    }

@router.post("/alerts/clear-markers")
def admin_clear_markers(
    fixture_id: int | None = Query(None),
    db: Session = Depends(get_db),
):
    q = db.query(Bet).filter(Bet.stake == 0.0)
    if fixture_id:
        q = q.filter(Bet.fixture_id == fixture_id)
    n = q.delete(synchronize_session=False)
    db.commit()
    return {"deleted_markers": n}

@router.post("/alerts/force-send")
def admin_force_send(
    fixture_id: int = Query(...),
    market: str = Query(...),
    bookmaker: str = Query(...),
    db: Session = Depends(get_db),
):
    e = (
        db.query(Edge)
        .filter(Edge.fixture_id == fixture_id, Edge.market == market, Edge.bookmaker == bookmaker)
        .order_by(Edge.edge.desc())
        .first()
    )
    f = db.query(Fixture).filter(Fixture.id == fixture_id).one_or_none()
    if not e or not f:
        return {"ok": False, "error": "Edge or Fixture not found"}

    alert_hash_input = f"{fixture_id}|{market}|{bookmaker}|{float(e.price):.4f}"
    alert_hash = sha256(alert_hash_input.encode()).hexdigest()

    try:
        send_telegram_alert(
            match=f"{f.home_team} v {f.away_team}",
            market=e.market,
            odds=float(e.price),
            edge=float(e.edge) * 100.0,
            kickoff=f.kickoff_utc.strftime("%a %d %b, %H:%M"),
            league=f.comp,
            bookmaker=e.bookmaker,
            model_source=e.model_source,
            link=None,
            bet_id=f"{f.id}-{e.market.replace(' ', '')}-{e.bookmaker}",
        )
        db.add(Bet(
            fixture_id=fixture_id,
            market=market,
            bookmaker=bookmaker,
            price=float(e.price),
            stake=0.0,
            placed_at=datetime.now(timezone.utc),
            duplicate_alert_hash=alert_hash,
        ))
        db.commit()
        return {"ok": True, "sent": True, "hash": alert_hash}
    except Exception as ex:
        return {"ok": False, "error": repr(ex)}

@router.post("/alerts/test")
def admin_alerts_test(
    match: str = Query(...),
    market: str = Query(...),
    bookmaker: str = Query(...),
    odds: float = Query(...),
    edge: float = Query(...),
    league: str = Query(...),
    model: str = Query(...),
    kickoff: str | None = Query(None),
):
    if not kickoff:
        kickoff = datetime.now(timezone.utc).strftime("%a %d %b, %H:%M")
    send_telegram_alert(
        match=match, market=market, odds=odds, edge=edge, kickoff=kickoff,
        league=league, bookmaker=bookmaker, model_source=model, link=None, bet_id="TEST_ONLY",
    )
    return {"ok": True, "sent": True}

# Bulk refresh for a competition (soccer-only)
@router.post("/refresh-odds-for-comp")
def admin_refresh_odds_for_comp(
    comp: str = Query(..., description="Exact Fixture.comp"),
    hours_ahead: int = Query(120, ge=1, le=14*24),
    prefer_book: str | None = Query(None),
    allow_fallback: int = Query(1),
    compute_after: int = Query(1),
    db: Session = Depends(get_db),
):
    now = datetime.now(timezone.utc)
    until = now + timedelta(hours=hours_ahead)
    fixtures = (
        db.query(Fixture)
        .filter(Fixture.comp == comp, Fixture.kickoff_utc >= now, Fixture.kickoff_utc <= until)
        .order_by(Fixture.kickoff_utc.asc())
        .all()
    )

    refreshed = []
    for f in fixtures:
        r = refresh_soccer_fixture(
            db,
            provider_fixture_id=f.provider_fixture_id,
            prefer_book=prefer_book,
            allow_fallback=bool(allow_fallback),
        )
        refreshed.append({"fixture_id": f.id, "home": f.home_team, "away": f.away_team, **r})

    if compute_after:
        ts = datetime.now(timezone.utc)
        ensure_baseline_probs(db, ts, source="team_form")
        compute_edges(db, ts, settings.EDGE_MIN, source="team_form")

    return {"ok": True, "comp": comp, "count": len(refreshed), "items": refreshed, "computed": bool(compute_after)}

# --- existing stuff here (fixtures-probe, last-http, etc.) ---

@router.post("/backfill/top-leagues-by-day")
def backfill_top_leagues_by_day(
    day: str = Query(..., description="YYYY-MM-DD"),
    db: Session = Depends(get_db),
):
    """
    For a single day, fetch fixtures from API-Football, keep only leagues in LEAGUE_MAP,
    upsert fixtures, and write FT scores where available.
    """
    d = date.fromisoformat(day)
    # one HTTP call for the whole day (populates LAST_HTTP)
    raw = apifootball._get(apifootball.API_URL, {"date": d.isoformat()})
    raw_total = len(raw)

    wanted_ids = set(apifootball.LEAGUE_MAP.values())
    filtered = [fx for fx in raw if (fx.get("league") or {}).get("id") in wanted_ids]
    filtered_total = len(filtered)

    upserted = 0
    settled = 0

    def _iso(dt: str) -> datetime:
        try:
            # API-Football returns ISO with Z; make tz-aware UTC
            return datetime.fromisoformat(dt.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)

    for fx in filtered:
        league = fx.get("league") or {}
        fix    = fx.get("fixture") or {}
        teams  = fx.get("teams")   or {}
        goals  = fx.get("goals")   or {}

        pid     = str((fix.get("id") or "")).strip()
        comp    = league.get("name") or ""
        country = league.get("country") or ""
        home    = (teams.get("home") or {}).get("name") or "Home"
        away    = (teams.get("away") or {}).get("name") or "Away"
        kickoff = _iso(fix.get("date") or "")

        if not pid:
            continue

        f = db.query(Fixture).filter(Fixture.provider_fixture_id == pid).one_or_none()
        if f:
            # update core fields in case names/times changed
            f.comp = comp
            f.country = country
            f.home_team = home
            f.away_team = away
            f.kickoff_utc = kickoff
        else:
            f = Fixture(
                provider_fixture_id=pid,
                comp=comp,
                country=country,
                home_team=home,
                away_team=away,
                kickoff_utc=kickoff,
            )
            db.add(f)
            upserted += 1

        status_short = ((fix.get("status") or {}).get("short") or "").upper()
        if status_short in {"FT", "AET", "PEN"}:
            gh = goals.get("home")
            ga = goals.get("away")
            if isinstance(gh, int) and isinstance(ga, int):
                f.full_time_home = gh
                f.full_time_away = ga
                f.result_settled = True
                settled += 1

    db.commit()
    return {
        "ok": True,
        "day": day,
        "raw_total": raw_total,
        "filtered_total": filtered_total,
        "fixtures_upserted": upserted,
        "finished_with_scores_written": settled,
        "last_http": apifootball.LAST_HTTP,
    }

@router.get("/apifootball/health")
def apifootball_health():
    # ping any known day; we only care that a request is made
    day = "2025-08-23"
    data = apifootball._get(apifootball.API_URL, {"date": day})
    return {
        "ok": True,
        "received_fixtures": len(data),
        "api_host": apifootball.API_HOST,
        "has_api_key": bool(apifootball.API_KEY),
        "last_http": apifootball.LAST_HTTP,
    }

@router.post("/backfill/odds-range")
def admin_backfill_odds_range(
    start_day: str = Query(..., description="YYYY-MM-DD"),
    ndays: int = Query(7, ge=1, le=31),
    leagues: str = Query("", description="CSV of LEAGUE_MAP keys (EPL,CHAMP,...) or empty=all comps"),
    prefer_book: str | None = Query(None),
    delay_sec: float = Query(0.35, ge=0.0, le=2.0),
    force: int = Query(0, description="1=fetch even if fixture already has odds"),
    db: Session = Depends(get_db),
):
    from ..services.apifootball import LEAGUE_MAP as TOP_KEYS

    start = datetime.fromisoformat(start_day).date()
    end = start + timedelta(days=ndays)

    # base query: fixtures in [start, end)
    q = (
        db.query(Fixture)
        .filter(Fixture.kickoff_utc >= datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc))
        .filter(Fixture.kickoff_utc <  datetime.combine(end,   datetime.min.time(), tzinfo=timezone.utc))
    )

    wanted_keys = [s.strip() for s in (leagues or "").split(",") if s.strip()]
    if wanted_keys:
        key_name_hint = {
            "EPL": "Premier League","CHAMP":"Championship","LG1":"League One","LG2":"League Two",
            "SCO_PREM":"Premiership","SCO_CHAMP":"Championship","SCO1":"League One","SCO2":"League Two",
            "LA_LIGA":"La Liga","BUNDES":"Bundesliga","BUNDES2":"2. Bundesliga",
            "SERIE_A":"Serie A","SERIE_B":"Serie B","LIGUE1":"Ligue 1",
            "UCL":"UEFA Champions League","UEL":"UEFA Europa League","UECL":"UEFA Europa Conference League",
            "WCQ_EUR":"World Cup - Qualification Europe",
        }
        name_hints = [key_name_hint[k] for k in wanted_keys if k in key_name_hint]
        if name_hints:
            cond = None
            for hint in name_hints:
                c = Fixture.comp.ilike(f"%{hint}%")
                cond = c if cond is None else (cond | c)
            if cond is not None:
                q = q.filter(cond)

    fixtures = q.order_by(Fixture.kickoff_utc.asc()).all()

    # if not forced, skip fixtures that already have some odds
    if not force and fixtures:
        have_odds_ids = set(r[0] for r in db.query(Odds.fixture_id).filter(Odds.fixture_id.in_([f.id for f in fixtures])).distinct())
        fixtures = [f for f in fixtures if f.id not in have_odds_ids]

    wrote = 0
    tried = 0
    for f in fixtures:
        tried += 1
        try:
            _ = refresh_soccer_fixture(
                db,
                provider_fixture_id=f.provider_fixture_id,
                prefer_book=prefer_book,
                allow_fallback=True,
            )
            wrote += 1
        except Exception:
            pass
        time.sleep(delay_sec)

    return {
        "ok": True,
        "window": [start.isoformat(), (end - timedelta(days=1)).isoformat()],
        "fixtures_considered": tried,
        "fetch_calls_made": wrote,
        "hint": "Re-run with force=1 to re-fetch even if odds exist; increase ndays or run multiple windows.",
    }

@router.post("/refresh-odds-for-comp-range")
def admin_refresh_odds_for_comp_range(
    league_key: str = Query(..., description="Key from your LEAGUE_MAP (disambiguates ENG vs SCO Championship)."),
    start_day: str = Query(..., description="YYYY-MM-DD (inclusive)"),
    ndays: int = Query(7, ge=1, le=62),
    prefer_book: str | None = Query("bet365"),
    allow_fallback: int = Query(1),
    odds_delay_sec: float = Query(0.25, ge=0.0, le=2.0),
    batch_limit: int = Query(75, ge=1, le=1000, description="Max fixtures to attempt in this call"),
    max_seconds: int = Query(600, ge=10, le=3600, description="Time budget for this call"),
    compute_after: int = Query(1),
    db: Session = Depends(get_db),
):
    """
    Refresh odds for fixtures already in the DB for [start_day, start_day+ndays).
    Processes up to `batch_limit` fixtures or `max_seconds`, whichever first.
    Safe to call repeatedly to chip away without hanging the request.
    """
    t0 = time.time()

    # Resolve comp/country
    comp_country = LEAGUE_KEY_FILTER.get(league_key.upper())
    if not comp_country:
        return {"ok": False, "error": f"Unknown league_key '{league_key}'", "known_keys": sorted(LEAGUE_KEY_FILTER.keys())}
    comp_name, country = comp_country

    start = datetime.fromisoformat(start_day).date()
    end   = start + timedelta(days=ndays)

    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
    end_dt   = datetime.combine(end,   datetime.min.time(), tzinfo=timezone.utc)

    # Only fixtures ALREADY in DB, matching comp (and optional country), in date range
    q = (
        db.query(Fixture)
        .filter(
            Fixture.kickoff_utc >= start_dt,
            Fixture.kickoff_utc <  end_dt,
            Fixture.comp == comp_name,
        )
        .order_by(Fixture.kickoff_utc.asc())
    )
    if country:
        q = q.filter(or_(Fixture.country == country, Fixture.country == None))  # include nulls to be permissive

    fixtures: List[Fixture] = q.limit(batch_limit).all()

    refreshed_ok = 0
    refreshed_fail = 0
    items: List[dict] = []

    for f in fixtures:
        if time.time() - t0 > max_seconds:
            break

        last = apifootball.LAST_HTTP or {}
        if last.get("status") == 429:
            try:
                reset_sec = float(last.get("ratelimit_reset") or 1.5)
                time.sleep(min(reset_sec, 10.0))
            except Exception:
                time.sleep(1.5)

        try:
            r = refresh_soccer_fixture(
                db,
                provider_fixture_id=f.provider_fixture_id,
                prefer_book=prefer_book,
                allow_fallback=bool(allow_fallback),
            )
            items.append({
                "fixture_id": f.id,
                "home": f.home_team, "away": f.away_team,
                "ko": f.kickoff_utc.isoformat(),
                **(r or {})
            })
            if r and r.get("error"):
                refreshed_fail += 1
            else:
                refreshed_ok += int(bool(r))
        except Exception as ex:
            refreshed_fail += 1
            items.append({
                "fixture_id": f.id,
                "home": f.home_team, "away": f.away_team,
                "ko": f.kickoff_utc.isoformat(),
                "error": repr(ex),
            })

        if odds_delay_sec:
            time.sleep(odds_delay_sec)

    if compute_after:
        ts = datetime.now(timezone.utc)
        ensure_baseline_probs(db, ts, source="team_form")
        compute_edges(db, ts, settings.EDGE_MIN, source="team_form")

    return {
        "ok": True,
        "start_day": start_day,
        "ndays": ndays,
        "filter": {"league_key": league_key, "comp": comp_name, "country": country},
        "attempted": len(items),
        "refreshed_ok": refreshed_ok,
        "refreshed_fail": refreshed_fail,
        "took_seconds": round(time.time() - t0, 2),
        "batch_limit": batch_limit,
        "max_seconds": max_seconds,
        "ratelimit_hint": {
            "last": apifootball.LAST_HTTP.get("status"),
            "remaining": apifootball.LAST_HTTP.get("ratelimit_remaining"),
            "reset": apifootball.LAST_HTTP.get("ratelimit_reset"),
            "error": apifootball.LAST_HTTP.get("error"),
        },
        "items": items,
        "computed": bool(compute_after),
    }

@router.post("/recompute-window")
def admin_recompute_window(
    hours_back: int = Query(24, ge=0, le=90*24),
    hours_ahead: int = Query(72, ge=0, le=90*24),
    source: str = Query("team_form"),
    db: Session = Depends(get_db),
):
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours_back)
    end = now + timedelta(hours=hours_ahead)

    ensure_baseline_probs(db, now, source=source, time_window=(start, end))
    compute_edges(db, now, settings.EDGE_MIN, source=source)

    return {"ok": True, "source": source, "window": [start.isoformat(), end.isoformat()]}

@router.post("/recompute-range")
def admin_recompute_range(
    start_day: str = Query(..., description="YYYY-MM-DD (UTC inclusive)"),
    ndays: int = Query(7, ge=1, le=90),
    comps: str | None = Query(None, description="CSV of exact Fixture.comp values to include"),
    use_closing_when_past: int = Query(1, description="1=use ClosingOdds if the window is in the past"),
    db: Session = Depends(get_db),
):
    start_date = datetime.fromisoformat(start_day).date()
    start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = start_dt + timedelta(days=ndays)
    now = datetime.now(timezone.utc)

    comp_list: List[str] = []
    if comps:
        comp_list = [c.strip() for c in comps.split(",") if c.strip()]

    fq = db.query(Fixture.id).filter(
        Fixture.kickoff_utc >= start_dt,
        Fixture.kickoff_utc <  end_dt,
    )
    if comp_list:
        fq = fq.filter(Fixture.comp.in_(comp_list))
    fixture_ids = [row.id for row in fq.all()]

    if not fixture_ids:
        return {
            "ok": True,
            "start_day": start_day,
            "ndays": ndays,
            "fixtures_in_window": 0,
            "model_probs_deleted": 0,
            "model_probs_written": 0,
            "note": "No fixtures found in window."
        }

    deleted = db.query(ModelProb).filter(
        and_(ModelProb.source == MODEL_SOURCE, ModelProb.fixture_id.in_(fixture_ids))
    ).delete(synchronize_session=False)
    db.flush()

    ensure_baseline_probs(
        db=db,
        now=now,
        time_window=(start_dt, end_dt),
        league_comp_filter=comp_list or None,
        use_closing_when_past=bool(use_closing_when_past),
        source=MODEL_SOURCE,
    )

    written = db.query(ModelProb).filter(
        and_(ModelProb.source == MODEL_SOURCE, ModelProb.fixture_id.in_(fixture_ids))
    ).count()

    db.commit()
    return {
        "ok": True,
        "start_day": start_day,
        "ndays": ndays,
        "fixtures_in_window": len(fixture_ids),
        "model_probs_deleted": int(deleted),
        "model_probs_written": int(written),
        "used_closing_odds": bool(use_closing_when_past and end_dt <= now),
    }

@router.post("/recompute-hours")
def admin_recompute_hours(
    db: Session = Depends(get_db),
    hours_back: int = Query(1440, ge=1, le=90*24),
    source: str = Query("team_form"),
    recompute_edges: int = Query(1),
):
    """
    Recompute baseline probs (and edges) for fixtures with KO in the last N hours.
    """
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=hours_back)
    ensure_baseline_probs(db, now, source=source, time_window=(since, now))
    if recompute_edges:
        compute_edges(db, now, settings.EDGE_MIN, source=source)
    probs_ct = db.query(func.count()).select_from(ModelProb).scalar()
    edges_ct = db.query(func.count()).select_from(Edge).scalar()
    return {"ok": True, "hours_back": hours_back, "model_probs_now": probs_ct, "edges_now": edges_ct}

@router.post("/recompute-from-closing")
def admin_recompute_from_closing(
    start_day: str = Query(..., description="YYYY-MM-DD"),
    ndays: int = Query(7, ge=1, le=60),
    comps: str | None = Query(None, description="CSV of Fixture.comp to restrict"),
    db: Session = Depends(get_db),
):
    start_dt = datetime.fromisoformat(start_day).date()
    end_dt = start_dt + timedelta(days=ndays)

    q = db.query(Fixture.id).filter(
        Fixture.kickoff_utc >= start_dt,
        Fixture.kickoff_utc < end_dt,
        Fixture.result_settled == True,
    )
    if comps:
        comp_list = [c.strip() for c in comps.split(",") if c.strip()]
        q = q.filter(Fixture.comp.in_(comp_list))

    fixture_ids = [fid for (fid,) in q.all()]

    deleted = db.query(ModelProb).filter(
        ModelProb.source == "consensus_v2",
        ModelProb.fixture_id.in_(fixture_ids)
    ).delete(synchronize_session=False)
    db.flush()

    written = 0
    for f_id in fixture_ids:
        rows = (
            db.query(ClosingOdds)
            .filter(ClosingOdds.fixture_id == f_id)
            .all()
        )
        for row in rows:
            try:
                p = 1.0 / float(row.price) if row.price and row.price > 1.0 else 0.0
            except Exception:
                continue
            if not (0.0 < p < 1.0):
                continue
            db.add(ModelProb(
                fixture_id=f_id,
                source="consensus_v2",
                market=row.market,
                prob=p,
                as_of=row.captured_at,
            ))
            written += 1

    db.commit()

    return {
        "ok": True,
        "start_day": start_day,
        "ndays": ndays,
        "fixtures_in_window": len(fixture_ids),
        "model_probs_deleted": deleted,
        "model_probs_written": written,
        "used_closing_odds": True,
    }

@router.post("/backfill_calibration")
def backfill_calibration(
    lookback_days: int = Query(45, ge=7, le=180),
    book: str = Query("bet365"),
    markets: str = Query("HOME_WIN,DRAW,AWAY_WIN,BTTS_Y,BTTS_N,O2.5,U2.5"),
    bins: int = Query(10, ge=5, le=20),
    min_n: int = Query(30, ge=10, le=200),
    db: Session = Depends(get_db),
):
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import func
    from ..models import Fixture, ClosingOdds, ModelProb, Calibration

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=lookback_days)
    wanted = [m.strip() for m in markets.split(",") if m.strip()]

    db.query(Calibration).filter(
        Calibration.book == book,
        Calibration.bin_count == bins,
        Calibration.market.in_(wanted),
    ).delete(synchronize_session=False)

    rows = (
        db.query(
            ModelProb.market,
            ModelProb.prob,
            Fixture.full_time_home,
            Fixture.full_time_away
        )
        .join(Fixture, Fixture.id == ModelProb.fixture_id)
        .join(ClosingOdds, (ClosingOdds.fixture_id == Fixture.id) & (ClosingOdds.market == ModelProb.market) & (ClosingOdds.bookmaker == book))
        .filter(Fixture.result_settled == True)
        .filter(Fixture.kickoff_utc >= since, Fixture.kickoff_utc <= now)
        .filter(ModelProb.market.in_(wanted))
        .all()
    )

    buckets = {(m, i): [] for m in wanted for i in range(bins)}
    def win_flag(mkt, h, a):
        total = (h or 0) + (a or 0)
        m = (mkt or "").upper()
        if m == "HOME_WIN": return (h or 0) > (a or 0)
        if m == "AWAY_WIN": return (a or 0) > (h or 0)
        if m == "DRAW":     return (h or 0) == (a or 0)
        if m == "BTTS_Y":   return (h or 0) > 0 and (a or 0) > 0
        if m == "BTTS_N":   return not ((h or 0) > 0 and (a or 0) > 0)
        if m.startswith("O") or m.startswith("U"):
            import re
            m_ = re.fullmatch(r"([OU])\s*([0-9]+(?:\.[0-9]+)?)", m)
            if not m_: return None
            side, line = m_.group(1), float(m_.group(2))
            return (total > line) if side == "O" else (total < line)
        return None

    for mkt, p, ft_h, ft_a in rows:
        try:
            prob = float(p)
        except:
            continue
        if not (0.0 <= prob <= 1.0): 
            continue
        b = min(bins - 1, int(prob * bins))
        w = win_flag(mkt, ft_h, ft_a)
        if w is None:
            continue
        buckets[(mkt, b)].append((prob, 1.0 if w else 0.0))

    written = 0
    for mkt in wanted:
        for b in range(bins):
            items = buckets[(mkt, b)]
            n = len(items)
            if n == 0: 
                continue
            avg_pred = sum(p for p, _ in items) / n
            emp = sum(w for _, w in items) / n
            if n >= min_n:
                db.add(Calibration(
                    market=mkt, book=book, bin_index=b, bin_count=bins,
                    avg_pred=avg_pred, emp_rate=emp, n=n,
                    fitted_at=now
                ))
                written += 1

    db.commit()
    return {"ok": True, "lookback_days": lookback_days, "book": book, "bins": bins, "rows_written": written}

@router.get("/closing-odds-days")
def get_closing_odds_days(
    db: Session = Depends(get_db)
):
    rows = (
        db.query(func.date(ClosingOdds.captured_at))
        .distinct()
        .order_by(func.date(ClosingOdds.captured_at))
        .all()
    )
    return [r[0].isoformat() for r in rows]

@router.get("/closing-odds-summary")
def closing_odds_summary(db: Session = Depends(get_db)):
    from sqlalchemy import func
    rows = (
        db.query(func.date(ClosingOdds.captured_at), func.count())
        .group_by(func.date(ClosingOdds.captured_at))
        .order_by(func.date(ClosingOdds.captured_at))
        .all()
    )
    return [{"date": r[0].isoformat(), "count": r[1]} for r in rows]

@router.get("/closing-odds-fixture-days")
def get_closing_odd_fixture_days(
    db: Session = Depends(get_db)
):
    from sqlalchemy import func
    rows = (
        db.query(func.date(Fixture.kickoff_utc), func.count())
        .select_from(ClosingOdds)
        .join(Fixture, Fixture.id == ClosingOdds.fixture_id)
        .group_by(func.date(Fixture.kickoff_utc))
        .order_by(func.date(Fixture.kickoff_utc))
        .all()
    )
    return [{"date": r[0].isoformat(), "count": r[1]} for r in rows]

@router.get("/debug/odds-for-date")
def odds_for_date(
    day: str = Query(..., description="YYYY-MM-DD"),
    db: Session = Depends(get_db),
):
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    rows = (
        db.query(Fixture.kickoff_utc, Odds.market, Odds.last_seen, Odds.price)
        .join(Odds, Odds.fixture_id == Fixture.id)
        .filter(Fixture.kickoff_utc >= start, Fixture.kickoff_utc < end)
        .order_by(Fixture.kickoff_utc, Odds.market, Odds.last_seen.desc())
        .all()
    )
    return [
        {
            "kickoff": r[0].isoformat(),
            "market": r[1],
            "last_seen": r[2].isoformat(),
            "price": r[3],
        }
        for r in rows
    ]

@router.post("/refresh-standings")
def refresh_standings(league: str = Query(...), db: Session = Depends(get_db)):
    if league not in SOCCER_LEAGUE_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown league: {league}")

    league_id = SOCCER_LEAGUE_MAP[league]
    url = f"https://v3.football.api-sports.io/standings?league={league_id}&season=2025"
    headers = {"x-apisports-key": settings.API_FOOTBALL_KEY}
    resp = requests.get(url, headers=headers).json()

    if not resp.get("response"):
        return {"ok": False, "league": league, "rows": 0, "reason": "empty response"}

    standings = resp["response"][0]["league"]["standings"][0]  # list of teams

    # clear old
    db.query(LeagueStanding).filter(LeagueStanding.league == league).delete()

    rows = []
    for row in standings:
        ls = LeagueStanding(
            league=league,
            season=resp["response"][0]["league"]["season"],
            team=row["team"]["name"],
            position=row["rank"],
            played=row["all"]["played"],
            win=row["all"]["win"],
            draw=row["all"]["draw"],
            lose=row["all"]["lose"],
            gf=row["all"]["goals"]["for"],
            ga=row["all"]["goals"]["against"],
            points=row["points"],
            form=row.get("form"),
        )
        rows.append(ls)

    db.add_all(rows)
    db.commit()

    return {"ok": True, "league": league, "rows": len(rows)}

@router.get("/debug/probs-today")
def debug_probs_today(db: Session = Depends(get_db)):
    today = datetime.now(timezone.utc).date()
    start = datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(today, datetime.max.time(), tzinfo=timezone.utc)

    rows = (
        db.query(Fixture.home_team, Fixture.away_team, ModelProb.market, ModelProb.prob)
        .join(ModelProb, ModelProb.fixture_id == Fixture.id)
        .filter(ModelProb.source == "team_form")
        .filter(Fixture.kickoff_utc >= start, Fixture.kickoff_utc <= end)
        .limit(50)
        .all()
    )

    return [
        {"home": r[0], "away": r[1], "market": r[2], "prob": r[3]}
        for r in rows
    ]

# --- Gridiron debug -----------------------------------------
from ..services.api_gridiron import LAST_HTTP as GRID_LAST_HTTP, _CACHE as GRID_CACHE
from ..services.api_gridiron import fetch_fixtures as grid_fetch

@router.get("/debug/gridiron-env")
def debug_gridiron_env():
    import os
    from ..services import api_gridiron as ag
    # mask the key for safety
    key = os.getenv("GRIDIRON_API_KEY", "")
    masked = (key[:4] + "…" + key[-4:]) if len(key) >= 8 else bool(key)
    return {
        "GRIDIRON_FIXTURES_URL": os.getenv("GRIDIRON_FIXTURES_URL"),
        "GRIDIRON_ODDS_URL": os.getenv("GRIDIRON_ODDS_URL"),
        "GRIDIRON_API_BASE": os.getenv("GRIDIRON_API_BASE"),
        "has_GRIDIRON_API_KEY": bool(key),
        "GRIDIRON_API_KEY_masked": masked,
        "using_rapidapi": bool(os.getenv("RAPIDAPI_KEY")),
        "headers_branch": ("RapidAPI" if os.getenv("RAPIDAPI_KEY") else "x-apisports-key"),
    }

@router.post("/debug/gridiron-probe")
def debug_gridiron_probe(day: str = Query(..., description="YYYY-MM-DD"),
                         league: str = Query("NCAA")):
    d = date.fromisoformat(day)
    # make a real call (will populate GRID_LAST_HTTP)
    data = grid_fetch(d, [league]) or []
    return {
        "count": len(data),
        "last_http": GRID_LAST_HTTP,
        "first_item_keys": list(data[0].keys()) if data else [],
    }

@router.post("/debug/gridiron-clear-cache")
def debug_gridiron_clear_cache():
    n = len(GRID_CACHE)
    GRID_CACHE.clear()
    return {"ok": True, "cleared_entries": n}

from fastapi import Body
import requests
from ..services.api_gridiron import HEADERS as GRID_HEADERS, GRIDIRON_FIXTURES_URL as GRID_GAMES_URL

@router.post("/debug/gridiron-raw")
def debug_gridiron_raw(day: str = Query(...), league: str = Query("NCAA"), season: int | None = Query(2025)):
    # map league name to id the same way as our client
    from ..services.api_gridiron import _league_id
    lid = _league_id(league)
    params = {"date": day}
    if lid: params["league"] = lid
    if season: params["season"] = season

    r = requests.get(GRID_GAMES_URL, headers=GRID_HEADERS, params=params, timeout=20)
    try:
        j = r.json()
    except Exception:
        j = {"_non_json_body": r.text[:4000]}
    return {"status": r.status_code, "params": params, "json": j}

@router.post("/debug/gridiron-compare")
def debug_gridiron_compare(day: str = Query(...)):
    variants = [
        {"label":"date_only", "params":{"date":day}},
        {"label":"date_league", "params":{"date":day, "league":2}},
        {"label":"date_season_league", "params":{"date":day, "season":2025, "league":2}},
    ]
    out = []
    for v in variants:
        r = requests.get(GRID_GAMES_URL, headers=GRID_HEADERS, params=v["params"], timeout=20)
        try: j = r.json()
        except Exception: j = {"_non_json_body": r.text[:1000]}
        out.append({
            "label": v["label"],
            "status": r.status_code,
            "params": v["params"],
            "results": j.get("results"),
            "errors": j.get("errors"),
            "paging": j.get("paging"),
            "hint_first_keys": list((j.get("response") or [{}])[0].keys()) if (j.get("response") or []) else [],
        })
    return {"url": GRID_GAMES_URL, "checks": out}

