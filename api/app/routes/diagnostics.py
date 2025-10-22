from __future__ import annotations

from datetime import datetime, timedelta, timezone, date
from typing import Dict, List, Optional, Any
from collections import defaultdict, Counter
import json

from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from ..db import get_db
from ..models import Fixture, Edge, ModelProb, Odds
from ..services import apifootball
from ..services import api_gridiron
from ..services import api_nba  # NBA diagnostics at bottom
from ..services.apifootball import fetch_odds_for_fixture, LAST_HTTP as SOCCER_LAST_HTTP

router = APIRouter(prefix="/admin", tags=["diagnostics"])


@router.get("/debug")
def admin_debug(db: Session = Depends(get_db)):
    edges_cnt = db.query(Edge).count()
    probs_cnt = db.query(ModelProb).count()
    latest_edge = db.query(Edge.created_at).order_by(Edge.created_at.desc()).first()
    return {
        "edges_count": edges_cnt,
        "model_probs_count": probs_cnt,
        "latest_edge": latest_edge[0].isoformat() if latest_edge else None,
    }


# --- Football provider probes -------------------------------------------------

@router.get("/fixtures-probe")
def admin_fixtures_probe(
    day: str = Query(..., description="YYYY-MM-DD"),
    leagues: str = Query("", description="CSV league keys or empty for all returned"),
):
    d = datetime.fromisoformat(day).date()
    league_keys = [s.strip() for s in leagues.split(",") if s.strip()] if leagues else []

    # Single call, no pagination
    raw = apifootball._get(apifootball.API_URL, {"date": d.isoformat()}) or []
    total_raw = len(raw)

    by_league: Dict[str, int] = defaultdict(int)
    sample_ids: Dict[str, list] = defaultdict(list)
    for fx in raw:
        lg = fx.get("league", {}) or {}
        lid = lg.get("id")
        name = lg.get("name") or str(lid)
        key = f"{name} ({lid})"
        by_league[key] += 1
        if len(sample_ids[key]) < 3:
            sample_ids[key].append((fx.get("fixture") or {}).get("id"))

    if league_keys:
        wanted_ids = {apifootball.LEAGUE_MAP[k] for k in league_keys if k in apifootball.LEAGUE_MAP}
        filtered = [fx for fx in raw if (fx.get("league") or {}).get("id") in wanted_ids]
    else:
        filtered = raw

    return {
        "day": day,
        "league_keys_used": league_keys,
        "raw_total": total_raw,
        "raw_by_league": by_league,
        "raw_sample_fixture_ids": sample_ids,
        "filtered_total": len(filtered),
        "last_http": apifootball.LAST_HTTP,
    }


@router.get("/fixtures-probe-detailed")
def fixtures_probe_detailed(
    day: str = Query(..., description="YYYY-MM-DD"),
):
    d = datetime.fromisoformat(day).date()

    # Curated set (same as backfill)
    TOP_KEYS = [
        "EPL", "CHAMP", "LG1", "LG2",
        "SCO_PREM", "SCO_CHAMP", "SCO1", "SCO2", "SCO_CUP",
        "LA_LIGA", "BUNDES", "BUNDES2", "SERIE_A", "SERIE_B", "LIGUE1",
        "UCL", "UEL", "UECL", "WCQ_EUR"
    ]
    TOP_IDS = {int(apifootball.LEAGUE_MAP[k]) for k in TOP_KEYS if k in apifootball.LEAGUE_MAP}

    # Page through the whole day once
    raw = apifootball._get_all_pages(apifootball.API_URL, {"date": d.isoformat()}) or []
    raw_total = len(raw)

    # Keep only curated leagues (by numeric id)
    filtered = [fx for fx in raw if (fx.get("league") or {}).get("id") in TOP_IDS]
    filtered_total = len(filtered)

    # Summaries
    def league_key(fx):
        lg = fx.get("league") or {}
        return (lg.get("id"), lg.get("name"), lg.get("country"))

    cnt = Counter(league_key(fx) for fx in filtered)
    kept_league_counts = [
        {"id": k[0], "name": k[1], "country": k[2], "count": v}
        for k, v in cnt.most_common(30)
    ]

    # Sample a few fixtures from the filtered set
    sample = []
    for fx in filtered[:10]:
        lg = fx.get("league") or {}
        fixture = fx.get("fixture") or {}
        teams = fx.get("teams") or {}
        sample.append({
            "league_id": lg.get("id"),
            "league_name": lg.get("name"),
            "league_country": lg.get("country"),
            "fixture_id": fixture.get("id"),
            "home": (teams.get("home") or {}).get("name"),
            "away": (teams.get("away") or {}).get("name"),
        })

    return {
        "day": day,
        "raw_total": raw_total,
        "filtered_total": filtered_total,
        "kept_league_counts": kept_league_counts,
        "sample_filtered": sample,
        "last_http": apifootball.LAST_HTTP,
    }


@router.get("/last-http")
def admin_last_http():
    return apifootball.LAST_HTTP


# --- Edges --------------------------------------------------------------------

@router.get("/debug-edges")
def admin_debug_edges(
    db: Session = Depends(get_db),
    min_edge: float = Query(-1.0, ge=-1.0, le=1.0),
    market: str | None = Query(None),
    prefer_book: str | None = Query(None),
    leagues: str | None = Query(None),
    hours_ahead: int = Query(168, ge=1, le=240),
    limit: int = Query(200, ge=1, le=2000),
):
    now = datetime.now(timezone.utc)
    until = now + timedelta(hours=hours_ahead)

    q = (
        db.query(Edge, Fixture)
        .join(Fixture, Edge.fixture_id == Fixture.id)
        .filter(Fixture.kickoff_utc >= now, Fixture.kickoff_utc <= until)
        .filter(Edge.edge >= min_edge)
        .order_by(Edge.edge.desc(), Fixture.kickoff_utc.asc())
    )

    if market:
        q = q.filter(Edge.market == market)

    if leagues:
        wanted = [s.strip() for s in leagues.split(",") if s.strip()]
        if wanted:
            q = q.filter(Fixture.comp.in_(wanted))

    if prefer_book:
        import re
        db_norm = func.replace(func.lower(Edge.bookmaker), " ", "")
        db_norm = func.replace(db_norm, "-", "")
        db_norm = func.replace(db_norm, "_", "")
        db_norm = func.replace(db_norm, ".", "")
        q = q.filter(db_norm == re.sub(r"[^a-z0-9]+", "", prefer_book.lower()))

    rows = q.limit(limit).all()

    def fair_price_from_prob(p: float) -> float | None:
        try:
            return (1.0 / p) if (p and p > 0.0) else None
        except Exception:
            return None

    out = []
    for e, f in rows:
        p = float(e.prob)
        price = float(e.price)
        fair = fair_price_from_prob(p)
        out.append({
            "fixture_id": f.id,
            "comp": f.comp,
            "home_team": f.home_team,
            "away_team": f.away_team,
            "kickoff_utc": f.kickoff_utc.isoformat(),
            "market": e.market,
            "bookmaker": e.bookmaker,
            "price": round(price, 4),
            "prob": round(p, 6),
            "fair_price": round(fair, 4) if fair else None,
            "edge": round(float(e.edge), 6),
            "edge_pct": round(float(e.edge) * 100.0, 3),
            "model_source": e.model_source,
            "created_at": e.created_at.isoformat() if e.created_at else None,
        })
    return {"count": len(out), "items": out}


# --- Gridiron probes ----------------------------------------------------------

@router.get("/gridiron-last-http")
def admin_gridiron_last_http():
    return api_gridiron.LAST_HTTP


@router.get("/gridiron-probe")
def admin_gridiron_probe(day: str = Query(..., description="YYYY-MM-DD")):
    d = datetime.fromisoformat(day).date()
    data = api_gridiron.fetch_fixtures(d, ["NFL", "CFB"])
    return {
        "day": day,
        "count": len(data),
        "last_http": api_gridiron.LAST_HTTP,
        "sample_ids": [(g.get("id") or (g.get("game") or {}).get("id")) for g in data[:10]],
    }


# --- Inspect a fixture (soccer odds shown if present) -------------------------

@router.get("/inspect/{fixture_id}", response_class=HTMLResponse)
def admin_inspect_fixture(fixture_id: int, db: Session = Depends(get_db)):
    """Show provider raw odds (soccer if available) + parsed rows for a fixture."""
    f = db.query(Fixture).filter(Fixture.id == fixture_id).one_or_none()
    if not f:
        raise HTTPException(status_code=404, detail="Fixture not found")

    # Raw provider odds (API-Football). NFL/CFB will naturally be empty here.
    try:
        raw = fetch_odds_for_fixture(int(f.provider_fixture_id))
    except Exception:
        raw = []
    pretty = json.dumps(raw, indent=2)

    # What we currently have parsed in our DB
    current = (
        db.query(Odds)
        .filter(Odds.fixture_id == fixture_id)
        .order_by(Odds.market.asc(), Odds.last_seen.desc())
        .all()
    )
    rows = [
        {
            "market": o.market,
            "bookmaker": o.bookmaker,
            "price": float(o.price),
            "last_seen": o.last_seen.isoformat(),
        }
        for o in current
    ]
    current_json = json.dumps(rows, indent=2)

    html = f"""
    <html><head><meta charset="utf-8"><title>Inspect Fixture {fixture_id}</title>
    <style>
      body{{font-family:ui-monospace,Menlo,Consolas,monospace;padding:16px}}
      pre{{background:#fafafa;border:1px solid #eee;padding:12px;overflow:auto}}
      .muted{{opacity:.7}}
    </style>
    </head><body>
      <h2>Fixture: {f.home_team} vs {f.away_team} — {f.kickoff_utc} <span class='muted'>({f.comp})</span></h2>
      <p><a href="/fixtures/{fixture_id}">Back to fixture page</a></p>

      <h3>Raw odds (API-Football; empty is normal for NFL/CFB)</h3>
      <pre>{pretty}</pre>

      <h3>Current parsed rows in DB</h3>
      <pre>{current_json}</pre>

      <h3>Provider diagnostics (soccer)</h3>
      <pre>{json.dumps(SOCCER_LAST_HTTP, indent=2)}</pre>
    </body></html>
    """
    return HTMLResponse(html)


# --- Provider introspection (soccer) ------------------------------------------

@router.get("/provider-introspect")
def provider_introspect():
    import inspect, os
    return {
        "module_file": os.path.abspath(inspect.getfile(apifootball)),
        "has_API_URL": hasattr(apifootball, "API_URL"),
        "API_URL": getattr(apifootball, "API_URL", None),
        "has_LAST_HTTP": hasattr(apifootball, "LAST_HTTP"),
        "LAST_HTTP": getattr(apifootball, "LAST_HTTP", None),
    }


@router.get("/provider-ping")
def provider_ping(date: str = Query(..., description="YYYY-MM-DD")):
    try:
        data = apifootball.fetch_fixtures_by_date(date)
        cnt = len(data) if isinstance(data, list) else 0
        return {"ok": True, "count": cnt, "last_http": apifootball.LAST_HTTP}
    except Exception as e:
        return {"ok": False, "error": repr(e), "last_http": getattr(apifootball, "LAST_HTTP", None)}


# --- Odds coverage & counts ---------------------------------------------------

@router.get("/odds-count")
def odds_count(db: Session = Depends(get_db)):
    total = db.query(Odds).count()
    by_market = (
        db.query(Odds.market, func.count(Odds.id))
        .group_by(Odds.market)
        .order_by(func.count(Odds.id).desc())
        .all()
    )
    return {
        "total_odds_rows": total,
        "by_market": {m: n for m, n in by_market},
    }


@router.get("/odds-coverage")
def odds_coverage(
    hours_back: int = Query(30 * 24, ge=1, le=365 * 24, description="Look-back window (hours)"),
    leagues: Optional[str] = Query(None, description="Optional CSV of exact Fixture.comp strings"),
    db: Session = Depends(get_db),
):
    """
    Coverage report over a time window, optionally filtered by competitions.
    """
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=hours_back)

    # precompute wanted leagues for reuse
    wanted: List[str] = []
    if leagues:
        wanted = [s.strip() for s in leagues.split(",") if s.strip()]

    # Base fixture set in time window
    fq = db.query(Fixture).filter(Fixture.kickoff_utc >= since, Fixture.kickoff_utc <= now)
    if wanted:
        fq = fq.filter(Fixture.comp.in_(wanted))

    fixtures: List[Fixture] = fq.all()
    fixture_ids = [f.id for f in fixtures]
    total_fixtures = len(fixtures)

    if not fixture_ids:
        return {
            "hours_back": hours_back,
            "total_fixtures": 0,
            "fixtures_with_odds": 0,
            "coverage_pct": 0.0,
            "rows_stats": {},
            "top_fixtures_by_odds": [],
            "missing_fixtures_sample": [],
            "by_market": {},
        }

    # Odds counts per fixture
    cnt_rows = (
        db.query(Odds.fixture_id, func.count(Odds.id).label("rows"))
        .filter(Odds.fixture_id.in_(fixture_ids))
        .group_by(Odds.fixture_id)
        .all()
    )
    rows_by_fix: Dict[int, int] = {fid: int(n) for (fid, n) in cnt_rows}

    fixtures_with_odds = len(rows_by_fix)
    coverage_pct = (fixtures_with_odds / total_fixtures * 100.0) if total_fixtures else 0.0

    # Build distribution
    counts = sorted(rows_by_fix.values())

    def _percentile(arr: List[int], p: float) -> Optional[float]:
        if not arr:
            return None
        k = (len(arr) - 1) * p
        f = int(k)
        c = min(f + 1, len(arr) - 1)
        if f == c:
            return float(arr[f])
        return arr[f] + (arr[c] - arr[f]) * (k - f)

    rows_stats = {
        "avg": (sum(counts) / len(counts)) if counts else 0.0,
        "median": _percentile(counts, 0.5),
        "p10": _percentile(counts, 0.10),
        "p90": _percentile(counts, 0.90),
        "max": max(counts) if counts else 0,
    }

    # Top fixtures by odds volume
    top_ids = sorted(rows_by_fix.items(), key=lambda kv: kv[1], reverse=True)[:15]
    fixture_map = {f.id: f for f in fixtures}
    top_fixtures = [
        {
            "fixture_id": fid,
            "rows": n,
            "comp": fixture_map[fid].comp,
            "home": fixture_map[fid].home_team,
            "away": fixture_map[fid].away_team,
            "kickoff_utc": (fixture_map[fid].kickoff_utc).isoformat(),
        }
        for fid, n in top_ids if fid in fixture_map
    ]

    # Missing odds sample (fixtures with 0 rows)
    missing = [
        {
            "fixture_id": f.id,
            "comp": f.comp,
            "home": f.home_team,
            "away": f.away_team,
            "kickoff_utc": f.kickoff_utc.isoformat(),
        }
        for f in fixtures if f.id not in rows_by_fix
    ][:25]

    # By-market breakdown within same window (and optional leagues filter)
    mq = (
        db.query(Odds.market, func.count(Odds.id))
        .join(Fixture, Fixture.id == Odds.fixture_id)
        .filter(Fixture.kickoff_utc >= since, Fixture.kickoff_utc <= now)
        .group_by(Odds.market)
    )
    if wanted:
        mq = mq.filter(Fixture.comp.in_(wanted))
    by_market = {m or "(null)": int(n) for (m, n) in mq.all()}

    return {
        "hours_back": hours_back,
        "total_fixtures": total_fixtures,
        "fixtures_with_odds": fixtures_with_odds,
        "coverage_pct": round(coverage_pct, 2),
        "rows_stats": rows_stats,
        "top_fixtures_by_odds": top_fixtures,
        "missing_fixtures_sample": missing,
        "by_market": by_market,
    }


# --- Fixtures by comp listing -------------------------------------------------

@router.get("/fixtures-by-comp")
def diagnostics_fixtures_by_comp(
    start_day: str = Query(..., description="YYYY-MM-DD (inclusive)"),
    ndays: int = Query(7, ge=1, le=90),
    comp: str | None = Query(None, description="Exact Fixture.comp match, e.g. 'World Cup - Qualification Europe'"),
    comp_like: str | None = Query(None, description="Case-insensitive substring match, e.g. 'Qualification Europe'"),
    country: str | None = Query(None, description="Optional exact country filter, e.g. 'World' or 'England'"),
    limit: int = Query(500, ge=1, le=5000),
    db: Session = Depends(get_db),
):
    """
    List fixtures in your DB for a given competition (exact or fuzzy) and date range.
    Useful to confirm what's actually present/tagged before running odds refresh.
    """
    # date window
    start = datetime.fromisoformat(start_day).date()
    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = start_dt + timedelta(days=ndays)

    q = (
        db.query(
            Fixture.id,
            Fixture.provider_fixture_id,
            Fixture.comp,
            Fixture.country,
            Fixture.home_team,
            Fixture.away_team,
            Fixture.kickoff_utc,
            Fixture.result_settled,
        )
        .filter(Fixture.kickoff_utc >= start_dt, Fixture.kickoff_utc < end_dt)
        .order_by(Fixture.kickoff_utc.asc(), Fixture.id.asc())
    )

    # optional filters
    if comp:
        q = q.filter(Fixture.comp == comp)
    if comp_like:
        q = q.filter(func.lower(Fixture.comp).like(f"%{comp_like.lower()}%"))
    if country:
        q = q.filter(Fixture.country == country)

    rows = q.limit(limit).all()

    # quick per-day counts
    per_day: Dict[str, int] = defaultdict(int)
    for r in rows:
        # ensure tz-aware for safe .date(); convert to UTC if naive
        ko = r.kickoff_utc if r.kickoff_utc.tzinfo else r.kickoff_utc.replace(tzinfo=timezone.utc)
        per_day[ko.date().isoformat()] += 1

    items = [
        {
            "fixture_id": r.id,
            "provider_id": r.provider_fixture_id,
            "comp": r.comp,
            "country": r.country,
            "home": r.home_team,
            "away": r.away_team,
            "kickoff_utc": (r.kickoff_utc if r.kickoff_utc.tzinfo else r.kickoff_utc.replace(tzinfo=timezone.utc)).isoformat(),
            "settled": bool(r.result_settled),
        }
        for r in rows
    ]

    return {
        "start_day": start_day,
        "ndays": ndays,
        "filters": {"comp": comp, "comp_like": comp_like, "country": country},
        "count": len(items),
        "per_day": dict(sorted(per_day.items())),
        "items": items,
    }


# --- Simple performance rollup (very rough) -----------------------------------

@router.get("/performance")
def diagnostics_performance(
    db: Session = Depends(get_db),
    hours_back: int = Query(720, ge=1, le=60 * 24),
    comps: str = Query("", description="CSV of Fixture.comp values to include"),
):
    """
    Join Fixtures + Odds + Results to measure odds coverage and performance.
    Filtered by comps (Fixture.comp) if provided.
    """
    since = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    q = db.query(Fixture).filter(Fixture.kickoff_utc >= since)

    comp_list: List[str] = []
    if comps:
        comp_list = [c.strip() for c in comps.split(",") if c.strip()]
        q = q.filter(Fixture.comp.in_(comp_list))

    fixtures = q.all()

    total = len(fixtures)
    settled = [f for f in fixtures if f.result_settled]
    settled_total = len(settled)

    # join odds
    rows = (
        db.query(Odds)
        .join(Fixture, Fixture.id == Odds.fixture_id)
        .filter(Fixture.kickoff_utc >= since)
        .all()
    )

    if comp_list:
        rows = [r for r in rows if r.fixture and r.fixture.comp in comp_list]

    by_market = defaultdict(int)
    by_market_hit = defaultdict(int)
    roi_by_market = defaultdict(float)

    for r in rows:
        by_market[r.market] += 1
        fx = r.fixture
        if fx and fx.result_settled:
            # simplistic: mark as win if market result matched (extend per-market later)
            if r.market == "HOME_WIN" and fx.full_time_home > fx.full_time_away:
                by_market_hit[r.market] += 1
                roi_by_market[r.market] += float(r.price) - 1.0
            if r.market == "AWAY_WIN" and fx.full_time_away > fx.full_time_home:
                by_market_hit[r.market] += 1
                roi_by_market[r.market] += float(r.price) - 1.0
            if r.market == "DRAW" and fx.full_time_home == fx.full_time_away:
                by_market_hit[r.market] += 1
                roi_by_market[r.market] += float(r.price) - 1.0
            # TODO: extend for O/U, BTTS once result schema is confirmed

    return {
        "hours_back": hours_back,
        "comps": comp_list,
        "fixtures_total": total,
        "fixtures_settled": settled_total,
        "odds_rows": len(rows),
        "by_market": dict(by_market),
        "hit_rates": {m: f"{by_market_hit[m]}/{by_market[m]}" for m in by_market},
        "roi": {m: roi_by_market[m] for m in roi_by_market},
    }


@router.get("/debug/calibrations")
def list_calibrations(db: Session = Depends(get_db)):
    from ..models import Calibration
    rows = db.query(Calibration).all()
    return [{"market": c.market, "book": c.book, "scope": c.scope, "alpha": c.alpha, "beta": c.beta} for c in rows]


# --- NBA diagnostics ----------------------------------------------------------

@router.get("/nba-last-http")
def admin_nba_last_http():
    return api_nba.LAST_HTTP


@router.get("/nba-probe")
def admin_nba_probe(day: str = Query(..., description="YYYY-MM-DD")):
    from ..services.api_nba import fetch_fixtures

    d = date.fromisoformat(day)
    rows = fetch_fixtures(d) or []

    sample: List[Dict[str, Any]] = []
    for r in rows[:10]:
        if not isinstance(r, dict):
            continue

        # league can be str or dict
        lg = r.get("league")
        if isinstance(lg, dict):
            league = lg.get("name") or lg.get("key") or lg.get("id")
        else:
            league = lg

        # date can be dict with "start" or plain string
        dv = r.get("date")
        if isinstance(dv, dict):
            start = dv.get("start")
        else:
            start = dv

        # teams block can vary
        tm = r.get("teams") or {}
        if isinstance(tm, dict):
            visitors = (tm.get("visitors") or tm.get("away") or {}) or {}
            home = (tm.get("home") or tm.get("host") or {}) or {}
        else:
            visitors, home = {}, {}

        def _nm(t: Any) -> str:
            if not isinstance(t, dict):
                return ""
            return (t.get("name") or t.get("nickname") or t.get("code") or "").strip()

        sample.append({
            "id": r.get("id"),
            "league": str(league) if league is not None else None,
            "home": _nm(home),
            "away": _nm(visitors),
            "start": start,
        })

    return {"day": day, "count": len(rows), "sample": sample}