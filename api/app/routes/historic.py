# api/app/routes/historic.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, or_, and_

from ..db import get_db
from ..models import Fixture, Odds, ClosingOdds
from ..services import apifootball  # reuse your existing API-Football wrapper

# Canonical (country, league name) pairs for your keys
# Canonical (country, league name) pairs for your keys
LEAGUE_KEY_CANON = {
    # England
    "EPL":       ("England", "Premier League"),
    "CHAMP":     ("England", "Championship"),
    "LG1":       ("England", "League One"),
    "LG2":       ("England", "League Two"),
    # Scotland
    "SCO_PREM":  ("Scotland", "Premiership"),
    "SCO_CHAMP": ("Scotland", "Championship"),
    "SCO1":      ("Scotland", "League One"),
    "SCO2":      ("Scotland", "League Two"),
    "SCO_CUP":   ("Scotland", "FA Cup"),
    # Spain / Germany / Italy / France
    "LA_LIGA":   ("Spain", "La Liga"),
    "BUNDES":    ("Germany", "Bundesliga"),
    "BUNDES2":   ("Germany", "2. Bundesliga"),
    "SERIE_A":   ("Italy", "Serie A"),
    "SERIE_B":   ("Italy", "Serie B"),
    "LIGUE1":    ("France", "Ligue 1"),
    # Aliases for provider naming quirks
    "PREMIERSHIP": ("Scotland", "Premiership"),
    "LA LIGA":     ("Spain", "La Liga"),
    "LIGUE_1":     ("France", "Ligue 1"),
    # Europe
    "UCL":       ("World", "UEFA Champions League"),
    "UEL":       ("World", "UEFA Europa League"),
    "UECL":      ("World", "UEFA Europa Conference League"),
    "WCQ_EUR":   ("World", "World Cup - Qualification Europe"),
}

router = APIRouter(prefix="/historic", tags=["historic"])
from ..services.apifootball import LEAGUE_MAP


@router.post("/results-range")
def historic_results_range(
    start_day: str = Query(..., description="YYYY-MM-DD"),
    ndays: int = Query(7, ge=1, le=60),
    leagues: str = Query("", description="CSV of league keys or names; empty = LEAGUE_MAP comps"),
    db: Session = Depends(get_db),
):
    """
    For each past fixture in [start, start+ndays), fetch the provider record and
    set full_time_home/away + result_settled=True when we can confidently read an FT score.
    NFL/NCAA are excluded automatically.
    """
    start = datetime.fromisoformat(start_day).date()
    end   = start + timedelta(days=ndays)
    now   = datetime.now(timezone.utc)

    q = (
        db.query(Fixture)
        .filter(Fixture.kickoff_utc >= datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc))
        .filter(Fixture.kickoff_utc <  datetime.combine(end,   datetime.min.time(), tzinfo=timezone.utc))
        .filter(Fixture.kickoff_utc < now)  # only games that have started
        .filter(~Fixture.comp.in_(["NFL", "NCAA"]))  # 🚫 exclude US football comps
    )

    if leagues:
        keys = [s.strip() for s in leagues.split(",") if s.strip()]
        filters = []

        # direct comp string matches (e.g. "Premier League")
        filters.append(Fixture.comp.in_(keys))

        # map league keys to canonical (country, comp)
        pairs = [LEAGUE_KEY_CANON[k] for k in keys if k in LEAGUE_KEY_CANON]
        if pairs:
            filters.append(or_(*[and_(Fixture.country == c, Fixture.comp == n) for (c, n) in pairs]))

        if filters:
            q = q.filter(or_(*filters))
    else:
        # valid = provider codes, canonical names, and alias keys
        valid_comps = set(LEAGUE_MAP.keys()) \
            | set(LEAGUE_KEY_CANON.keys()) \
            | {canon_name for (_, canon_name) in LEAGUE_KEY_CANON.values()}
        q = q.filter(Fixture.comp.in_(valid_comps))

    fixtures = q.all()
    # --- helpers --------------------------------------------------------------
    def _norm_int(x):
        try:
            return int(x)
        except Exception:
            return None

    def _first(d, *path):
        cur = d
        for p in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(p)
        return cur

    def _flatten_to_item(raw):
        if isinstance(raw, dict) and isinstance(raw.get("response"), list) and raw["response"]:
            return raw["response"][0]
        if isinstance(raw, list) and raw:
            return raw[0]
        if isinstance(raw, dict):
            return raw
        return {}

    def _fetch_provider(provider_id: int) -> dict:
        try:
            raw = apifootball.fetch_fixture_by_id(int(provider_id))
            if raw:
                return raw
        except Exception:
            pass
        try:
            alt = apifootball._get(apifootball.API_URL, {"id": int(provider_id)})
            return {"response": alt} if isinstance(alt, list) else alt
        except Exception:
            return {}

    updated = 0
    examined = 0

    for f in fixtures:
        if getattr(f, "result_settled", False):
            continue

        raw = _fetch_provider(int(f.provider_fixture_id))
        if not raw:
            continue

        item = _flatten_to_item(raw)
        if not item:
            continue

        status_short = str(_first(item, "fixture", "status", "short") or "").upper()

        ko = f.kickoff_utc
        if ko.tzinfo is None:
            ko = ko.replace(tzinfo=timezone.utc)

        finished_flag = status_short in {"FT", "AET", "PEN"}
        long_overdue  = (now - ko) > timedelta(hours=3)

        goals_home = _norm_int(_first(item, "goals", "home"))
        goals_away = _norm_int(_first(item, "goals", "away"))
        if goals_home is None or goals_away is None:
            ft = _first(item, "score", "fulltime") or {}
            if isinstance(ft, dict):
                goals_home = goals_home if goals_home is not None else _norm_int(ft.get("home"))
                goals_away = goals_away if goals_away is not None else _norm_int(ft.get("away"))

        examined += 1
        if (goals_home is not None and goals_away is not None) and (finished_flag or long_overdue):
            f.full_time_home = goals_home
            f.full_time_away = goals_away
            f.result_settled = True
            updated += 1

    db.commit()
    return {
        "ok": True,
        "fixtures_checked": len(fixtures),
        "fixtures_examined": examined,
        "results_updated": updated
    }


@router.get("/inspect-provider")
def historic_inspect_provider(provider_id: int = Query(...)):
    """
    Debug helper: show the raw provider shape for a single API-Football fixture id.
    """
    try:
        raw = apifootball.fetch_fixture_by_id(int(provider_id))
    except Exception:
        raw = None
    if not raw:
        try:
            alt = apifootball._get(apifootball.API_URL, {"id": int(provider_id)})
            raw = {"response": alt} if isinstance(alt, list) else alt
        except Exception:
            raw = {}

    # shallow summary to make it readable
    item = None
    if isinstance(raw, dict) and isinstance(raw.get("response"), list) and raw["response"]:
        item = raw["response"][0]
    elif isinstance(raw, list) and raw:
        item = raw[0]
    elif isinstance(raw, dict):
        item = raw

    summary = {}
    if isinstance(item, dict):
        summary = {
            "status_short": (item.get("fixture") or {}).get("status", {}).get("short"),
            "goals": item.get("goals"),
            "score_fulltime": (item.get("score") or {}).get("fulltime"),
            "teams": item.get("teams"),
            "league": item.get("league"),
        }

    return {"raw_present": bool(raw), "summary": summary}


@router.post("/closing-odds")
def historic_closing_odds(
    hours_back: int = Query(72, ge=1, le=24*30),
    prefer_book: str | None = Query("bet365"),
    markets: str | None = Query(None, description="CSV of markets, default HOME/DRAW/AWAY, BTTS, O/U 1.5/2.5"),
    db: Session = Depends(get_db),
):
    """
    For finished fixtures within the past hours_back, pick a closing price
    per market/bookmaker from existing Odds rows.
    Rule: prefer odds in [KO-30m, KO], else latest <= KO.
    """
    import re

    def _norm_book(s: str | None) -> str:
        return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=hours_back)

    wanted_markets = (
        [s.strip() for s in markets.split(",") if s.strip()] if markets
        else ["HOME_WIN","DRAW","AWAY_WIN","BTTS_Y","BTTS_N","O2.5","U2.5"]
    )

    fixtures = (
        db.query(Fixture)
        .filter(Fixture.kickoff_utc >= since, Fixture.kickoff_utc <= now)
        .filter(Fixture.result_settled == True)
        .all()
    )

    written = 0
    window = timedelta(minutes=30)
    prefer_norm = _norm_book(prefer_book) if prefer_book else None

    for f in fixtures:
        ko = f.kickoff_utc
        if ko.tzinfo is None:
            ko = ko.replace(tzinfo=timezone.utc)

        for mkt in wanted_markets:
            q = db.query(Odds).filter(
                Odds.fixture_id == f.id,
                Odds.market == mkt,
                Odds.last_seen <= ko + timedelta(hours=2)
            )

            if prefer_norm:
                db_norm = func.replace(func.lower(Odds.bookmaker), " ", "")
                db_norm = func.replace(db_norm, "-", "")
                db_norm = func.replace(db_norm, "_", "")
                db_norm = func.replace(db_norm, ".", "")
                q = q.filter(db_norm == prefer_norm)

            # Find the best closing odds: prefer [KO-30m to KO], otherwise latest before KO+2h
            # Find the best closing odds: prefer [KO-30m to KO+2h], fallback to any latest
            # Prefer odds in [KO-30m to KO+2h], fallback to latest seen odds if none found
           # Extend loose window much further (e.g. 3 days)
            loose_cutoff = ko + timedelta(days=3)

            tight_window = (
                q.filter(Odds.last_seen >= (ko - window), Odds.last_seen <= loose_cutoff)
                .order_by(Odds.last_seen.desc())
                .first()
            )

            loose_window = (
                q.filter(Odds.last_seen <= loose_cutoff)
                .order_by(Odds.last_seen.desc())
                .first()
            )
            if not tight_window:
                print(f"[DEBUG] No tight odds — falling back to latest for fixture {f.id} / market {mkt}")

            pick = tight_window or q.order_by(Odds.last_seen.desc()).first()

            # fallback to ANY bookmaker if prefer_book odds not found
            if not pick and prefer_norm:
                print(f"[DEBUG] No preferred bookmaker odds — trying any bookmaker for fixture {f.id} / market {mkt}")
                fallback_q = (
                    db.query(Odds)
                    .filter(
                        Odds.fixture_id == f.id,
                        Odds.market == mkt,
                        Odds.last_seen <= loose_cutoff  # ✅ match extended fallback window
                    )
                    .order_by(Odds.last_seen.desc())
                    .first()
                )

                if fallback_q:
                    pick = fallback_q
                    book_for_row = fallback_q.bookmaker  # override to fallback bookmaker
                    print(f"[DEBUG] Fallback used for fixture {f.id} / market {mkt} — picked {fallback_q.bookmaker}")
                else:
                    print(f"[DEBUG] No closing odds (even fallback) for fixture {f.id} / market {mkt} — KO={ko}")
                    continue  # Still nothing, skip
            book_for_row = pick.bookmaker

            existing = (
                db.query(ClosingOdds)
                .filter(
                    ClosingOdds.fixture_id == f.id,
                    ClosingOdds.market == mkt,
                    ClosingOdds.bookmaker == book_for_row,
                )
                .one_or_none()
            )
            if existing:
                existing.price = float(pick.price)
                existing.captured_at = pick.last_seen
            else:
                db.add(ClosingOdds(
                    fixture_id=f.id,
                    market=mkt,
                    bookmaker=book_for_row,
                    price=float(pick.price),
                    captured_at=pick.last_seen,
                    source="closing_rule_30m",
                ))
            written += 1
            print(f"[DEBUG] Closing odds for fixture {f.id}, market={mkt}, price={pick.price}, book={pick.bookmaker}, seen={pick.last_seen}")

    db.commit()
    return {"ok": True, "fixtures": len(fixtures), "closing_rows_written": written}


@router.get("/finished-missing-results")
def historic_finished_missing_results(
    hours_back: int = Query(96, ge=1, le=24*30),
    db: Session = Depends(get_db),
):
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=hours_back)
    rows = (
        db.query(Fixture)
        .filter(Fixture.kickoff_utc >= since, Fixture.kickoff_utc <= now)
        .filter((Fixture.result_settled == None) | (Fixture.result_settled == False))
        .order_by(Fixture.kickoff_utc.desc())
        .all()
    )
    return {
        "count": len(rows),
        "items": [
            {
                "fixture_id": f.id,
                "provider_id": f.provider_fixture_id,
                "comp": f.comp,
                "home": f.home_team,
                "away": f.away_team,
                "ko": (f.kickoff_utc.replace(tzinfo=timezone.utc) if f.kickoff_utc.tzinfo is None else f.kickoff_utc).isoformat(),
            }
            for f in rows
        ],
    }

from ..services.apifootball import LEAGUE_MAP  # make sure this is imported

@router.post("/backfill-top-leagues")
def backfill_top_leagues(
    days_back: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
):
    """
    Backfill fixtures + results only for the curated LEAGUE_MAP set (top comps).
    """
    now = datetime.now(timezone.utc).date()
    start = now - timedelta(days=days_back)

    upserted = 0
    seen = 0

    for key, league_id in LEAGUE_MAP.items():
        for d in (start + timedelta(days=i) for i in range(days_back)):
            try:
                raw = apifootball.fetch_fixtures_by_date(league_id=league_id, date=d.isoformat())
            except Exception:
                continue
            if not raw:
                continue

            for item in raw.get("response", []):
                seen += 1
                f = apifootball.upsert_fixture(db, item)  # use your existing helper
                if f:
                    upserted += 1

    db.commit()
    return {
        "ok": True,
        "days_back": days_back,
        "leagues": list(LEAGUE_MAP.keys()),
        "provider_fixtures_seen": seen,
        "fixtures_upserted": upserted,
    }

from fastapi.responses import StreamingResponse
import csv
from io import StringIO

@router.get("/export-csv")
def historic_export_csv(
    hours_back: int = Query(96, ge=1, le=24*30),
    markets: str | None = Query(None, description="CSV markets to include; default = common 1X2/BTTS/OU"),
    prefer_book: str | None = Query("bet365"),
    db: Session = Depends(get_db),
):
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=hours_back)

    wanted_markets = (
        [s.strip() for s in markets.split(",") if s.strip()] if markets
        else ["HOME_WIN","DRAW","AWAY_WIN","BTTS_Y","BTTS_N","O2.5","U2.5"]
    )

    # join fixtures with captured closing odds and export alongside FT scores
    q = (
        db.query(
            Fixture.id,
            Fixture.provider_fixture_id,
            Fixture.comp,
            Fixture.home_team,
            Fixture.away_team,
            Fixture.kickoff_utc,
            Fixture.full_time_home,
            Fixture.full_time_away,
            ClosingOdds.market,
            ClosingOdds.bookmaker,
            ClosingOdds.price,
            ClosingOdds.captured_at,
        )
        .join(ClosingOdds, ClosingOdds.fixture_id == Fixture.id)
        .filter(Fixture.kickoff_utc >= since, Fixture.kickoff_utc <= now)
        .filter(Fixture.result_settled == True)
    )
    if prefer_book:
        q = q.filter(ClosingOdds.bookmaker == prefer_book)
    if wanted_markets:
        q = q.filter(ClosingOdds.market.in_(wanted_markets))

    rows = q.order_by(Fixture.kickoff_utc.desc()).all()

    # stream as CSV
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "fixture_id","provider_id","comp","home","away","kickoff_utc",
        "ft_home","ft_away","market","bookmaker","closing_price","captured_at"
    ])
    for r in rows:
        writer.writerow([
            r.id, r.provider_fixture_id, r.comp, r.home_team, r.away_team,
            (r.kickoff_utc.replace(tzinfo=timezone.utc) if r.kickoff_utc.tzinfo is None else r.kickoff_utc).isoformat(),
            r.full_time_home, r.full_time_away,
            r.market, r.bookmaker, float(r.price), r.captured_at.isoformat()
        ])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="closing_odds_export.csv"'}
    )

@router.get("/export-results")
def historic_export_results(
    hours_back: int | None = Query(40*24, ge=1, le=365*24, description="Look-back window in hours; ignore if start_day is provided"),
    start_day: str | None = Query(None, description="YYYY-MM-DD (inclusive); if set, uses date range mode"),
    ndays: int = Query(1, ge=1, le=365, description="Days from start_day (date range mode)"),
    all_time: int = Query(0, description="Set to 1 to export ALL settled results without a time filter"),
    leagues: str | None = Query(None, description="Optional CSV of exact Fixture.comp strings"),
    league_keys: str | None = Query(None, description="Optional CSV of LEAGUE_MAP keys, e.g. EPL,CHAMP,LA_LIGA,..."),
    db: Session = Depends(get_db),
):
    from fastapi.responses import StreamingResponse
    import csv
    from io import StringIO

    q = (
        db.query(
            Fixture.id,
            Fixture.provider_fixture_id,
            Fixture.comp,
            Fixture.country,
            Fixture.home_team,
            Fixture.away_team,
            Fixture.kickoff_utc,
            Fixture.full_time_home,
            Fixture.full_time_away,
        )
        .filter(Fixture.result_settled == True)
        .order_by(Fixture.kickoff_utc.desc())
    )

    # ---- time window ----
    if not all_time:
        if start_day:
            start = datetime.fromisoformat(start_day).date()
            start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
            end_dt = start_dt + timedelta(days=ndays)
            q = q.filter(Fixture.kickoff_utc >= start_dt, Fixture.kickoff_utc < end_dt)
        else:
            now = datetime.now(timezone.utc)
            since = now - timedelta(hours=hours_back or 21*24)
            q = q.filter(Fixture.kickoff_utc >= since, Fixture.kickoff_utc <= now)

    # ---- name-based filter (exact Fixture.comp) ----
    filters = []
    if leagues:
        wanted_names = [s.strip() for s in leagues.split(",") if s.strip()]
        if wanted_names:
            filters.append(Fixture.comp.in_(wanted_names))

    # ---- key-based filter (country + comp) ----
    if league_keys:
        keys = [s.strip() for s in league_keys.split(",") if s.strip()]
        pairs = [LEAGUE_KEY_CANON[k] for k in keys if k in LEAGUE_KEY_CANON]
        if pairs:
            filters.append(or_(*[and_(Fixture.country == c, Fixture.comp == n) for (c, n) in pairs]))

    if filters:
        q = q.filter(or_(*filters))

    rows = q.all()

    # ---- CSV ----
    buf = StringIO()
    w = csv.writer(buf)
    w.writerow(["fixture_id","provider_id","comp","country","home","away","kickoff_utc","ft_home","ft_away"])
    for r in rows:
        ko = r.kickoff_utc if r.kickoff_utc.tzinfo else r.kickoff_utc.replace(tzinfo=timezone.utc)
        w.writerow([r.id, r.provider_fixture_id, r.comp, r.country or "",
                    r.home_team, r.away_team, ko.isoformat(), r.full_time_home, r.full_time_away])

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="results_only_export.csv"',
            "X-Row-Count": str(len(rows)),
        }
    )

@router.get("/results-stats")
def historic_results_stats(
    days_back: int = Query(60, ge=1, le=365),
    db: Session = Depends(get_db),
):
    """
    Quick histogram of settled fixtures per day (UTC).
    """
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days_back)
    rows = (
        db.query(Fixture.kickoff_utc, Fixture.result_settled)
        .filter(Fixture.kickoff_utc >= since, Fixture.kickoff_utc <= now)
        .all()
    )
    by_day = {}
    settled = 0
    total = 0
    for (ko, settled_flag) in rows:
        total += 1
        day = ko.date().isoformat()
        d = by_day.setdefault(day, {"total": 0, "settled": 0})
        d["total"] += 1
        if settled_flag:
            d["settled"] += 1
            settled += 1
    return {"days_back": days_back, "total_in_window": total, "settled_in_window": settled, "by_day": by_day}

@router.get("/list-settled-comps")
def historic_list_settled_comps(
    days_back: int = Query(90, ge=1, le=365),
    db: Session = Depends(get_db),
):
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days_back)
    rows = (
        db.query(Fixture.comp, func.count(Fixture.id))
        .filter(Fixture.result_settled == True)
        .filter(Fixture.kickoff_utc >= since, Fixture.kickoff_utc <= now)
        .group_by(Fixture.comp)
        .order_by(func.count(Fixture.id).desc())
        .all()
    )
    return [{"comp": c or "(null)", "count": n} for (c, n) in rows]

@router.get("/list-settled-comps-detailed")
def historic_list_settled_comps_detailed(
    days_back: int = Query(90, ge=1, le=365),
    db: Session = Depends(get_db),
):
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days_back)
    rows = (
        db.query(Fixture.comp, Fixture.country, func.count(Fixture.id))
        .filter(Fixture.kickoff_utc >= since, Fixture.kickoff_utc <= now)
        .filter(Fixture.result_settled == True)
        .group_by(Fixture.comp, Fixture.country)
        .order_by(func.count(Fixture.id).desc())
        .all()
    )
    return [{"comp": c or "(null)", "country": k or "(null)", "count": n} for (c, k, n) in rows]

@router.post("/backfill-countries")
def historic_backfill_countries(
    days_back: int = Query(120, ge=1, le=365),
    only_nulls: int = Query(1, description="1=only where country is NULL/empty, 0=overwrite"),
    db: Session = Depends(get_db),
):
    """
    Fill Fixture.country using provider data; if provider lacks it, infer from Fixture.comp
    for common competitions (EPL, La Liga, Bundesliga, Serie A, Ligue 1, Eredivisie, etc.).
    """
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days_back)

    q = db.query(Fixture).filter(Fixture.kickoff_utc >= since, Fixture.kickoff_utc <= now)
    if only_nulls:
        q = q.filter((Fixture.country == None) | (Fixture.country == ""))  # noqa: E711

    rows = q.all()

    # Fallback map for popular comps (extend as needed)
    COMP_TO_COUNTRY = {
        # England
        "Premier League": "England",
        "Championship": "England",
        "League One": "England",
        "League Two": "England",
        "FA Cup": "England",
        "League Cup": "England",
        # Spain
        "La Liga": "Spain",
        "Primera División": "Spain",
        "Segunda División": "Spain",
        "Copa del Rey": "Spain",
        # Germany
        "Bundesliga": "Germany",
        "2. Bundesliga": "Germany",
        "DFB Pokal": "Germany",
        # Italy
        "Serie A": "Italy",
        "Serie B": "Italy",
        "Coppa Italia": "Italy",
        "Coppa Italia Serie C": "Italy",
        # France
        "Ligue 1": "France",
        "Ligue 2": "France",
        "Coupe de France": "France",
        # Netherlands / Portugal / Scotland
        "Eredivisie": "Netherlands",
        "Eerste Divisie": "Netherlands",
        "Primeira Liga": "Portugal",
        "Segunda Liga": "Portugal",
        "Scottish Premiership": "Scotland",
        "Scottish Championship": "Scotland",
        # Int’l
        "World Cup - Qualification Europe": "World",
        # US
        "Major League Soccer": "USA",
        "USL Championship": "USA",
    }

    def _first(d, *path):
        cur = d
        for p in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(p)
        return cur

    def _flatten_one(raw):
        if isinstance(raw, dict) and isinstance(raw.get("response"), list) and raw["response"]:
            return raw["response"][0]
        if isinstance(raw, list) and raw:
            return raw[0]
        if isinstance(raw, dict):
            return raw
        return {}

    def _fetch(provider_id: int) -> dict:
        try:
            raw = apifootball.fetch_fixture_by_id(provider_id)
            if raw: return raw
        except Exception:
            pass
        try:
            alt = apifootball._get(apifootball.API_URL, {"id": provider_id})
            return {"response": alt} if isinstance(alt, list) else alt
        except Exception:
            return {}

    updated = 0
    examined = 0

    for f in rows:
        examined += 1

        # 1) Try provider league.country
        league_country = None
        raw = _fetch(int(f.provider_fixture_id)) if f.provider_fixture_id else {}
        item = _flatten_one(raw)
        if item:
            league_country = _first(item, "league", "country") or _first(item, "country", "name")

        # 2) Fallback: infer from comp name
        if not league_country:
            comp = (f.comp or "").strip()
            league_country = COMP_TO_COUNTRY.get(comp)

        # 3) Light heuristics (optional): map some common variants
        if not league_country and "Premier League" in (f.comp or "") and "Scot" in (f.comp or ""):
            league_country = "Scotland"

        # 4) Write if we have something
        if league_country and (not f.country or not only_nulls):
            f.country = str(league_country)
            updated += 1

    db.commit()
    return {"ok": True, "considered": len(rows), "examined": examined, "updated": updated}