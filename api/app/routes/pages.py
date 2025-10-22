from __future__ import annotations

from datetime import datetime, timedelta, timezone
from collections import defaultdict
import re

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..db import get_db
from ..models import Fixture, Edge, Odds, Prediction, ModelProb
from ..web import templates
from .explain import confidence_from_prob

router = APIRouter(tags=["pages"])

@router.get("/", response_class=HTMLResponse)
def home():
    # server-rendered shortlist page
    return templates.TemplateResponse("shortlist.html", {"request": {}})

@router.get("/_books", response_class=HTMLResponse)
def list_books(db: Session = Depends(get_db)):
    rows = db.query(func.distinct(Odds.bookmaker)).order_by(Odds.bookmaker.asc()).all()
    opts = ['<option value="">All books</option>']
    for (name,) in rows:
        if not name:
            continue
        opts.append(f'<option value="{name}">{name}</option>')
    return HTMLResponse("".join(opts) if len(opts) > 1 else '<option value="">(none)</option>')

@router.get("/_shortlist", response_class=HTMLResponse)
def shortlist_partial(
    db: Session = Depends(get_db),
    min_edge: float = Query(0.02, ge=-1.0, le=1.0),
    market: str | None = None,
    model_source: str | None = Query(None),
    mode: str = Query("fixture"),
    prefer_book: str | None = Query(None),
    leagues: str | None = Query(None),
    limit: int = Query(100, ge=5, le=500),
    include_negative: int = Query(0),
    sort_by: str = Query("edge"),   # ✅ new param
):
    now = datetime.now(timezone.utc)
    threshold = min(min_edge, -0.10) if include_negative else max(min_edge, 0.0)

    base = (
        db.query(Edge, Fixture)
        .join(Fixture, Edge.fixture_id == Fixture.id)
        .filter(Fixture.kickoff_utc >= now, Edge.edge >= threshold)
    )

    if leagues:
        wanted = [s.strip() for s in leagues.split(",") if s.strip()]
        if wanted:
            base = base.filter(Fixture.comp.in_(wanted))

    if market:
        base = base.filter(Edge.market == market)

    if prefer_book:
        norm = re.sub(r"[^a-z0-9]+", "", prefer_book.lower())
        db_norm = func.replace(func.lower(Edge.bookmaker), " ", "")
        db_norm = func.replace(db_norm, "-", "")
        db_norm = func.replace(db_norm, "_", "")
        db_norm = func.replace(db_norm, ".", "")
        base = base.filter(db_norm == norm)

    if model_source:
        base = base.filter(Edge.model_source == model_source)

    # ✅ Sorting
    if sort_by == "kickoff":
        base = base.order_by(Fixture.kickoff_utc.asc())
    elif sort_by == "comp":
        base = base.order_by(Fixture.comp.asc(), Fixture.kickoff_utc.asc())
    elif sort_by == "prob":
        base = base.order_by(Edge.prob.desc())
    else:  # default: edge
        base = base.order_by(Edge.edge.desc(), Fixture.kickoff_utc.asc())

    rows = base.all()

    grouped: defaultdict[tuple, list[tuple[Edge, Fixture]]] = defaultdict(list)
    if mode == "market":
        for e, f in rows:
            grouped[(f.id, e.market)].append((e, f))
    else:
        for e, f in rows:
            grouped[(f.id,)].append((e, f))

    picked: list[tuple[Edge, Fixture]] = []
    for items in grouped.values():
        best = max(items, key=lambda t: float(t[0].edge))
        picked.append(best)

    picked = picked[:limit]

    if not picked:
        return HTMLResponse("<tr><td colspan='12'>No edges yet — adjust filters or check 'Include negative'.</td></tr>")

    html_rows: list[str] = []
    for e, f in picked:
        # ✅ Always use latest ModelProb
        prob_row = (
            db.query(ModelProb.prob)
            .filter(ModelProb.fixture_id == f.id, ModelProb.market == e.market)
            .order_by(ModelProb.as_of.desc())
            .first()
        )
        prob_val = float(prob_row[0]) if prob_row else float(e.prob)

        # Confidence tier
        conf = confidence_from_prob(prob_val)
        if conf == "High":
            conf_badge = "<span style='color:green;font-weight:bold'>🟢 High</span>"
        elif conf == "Medium":
            conf_badge = "<span style='color:orange;font-weight:bold'>🟡 Medium</span>"
        else:
            conf_badge = "<span style='color:red;font-weight:bold'>🔴 Low</span>"

        html_rows.append(
            f"<tr>"
            f"<td>{f.comp}</td>"
            f"<td><a href='/fixtures/{f.id}'>{f.home_team} vs {f.away_team}</a></td>"
            f"<td>{f.kickoff_utc:%Y-%m-%d %H:%M}</td>"
            f"<td><span class='pill'>{e.market}</span></td>"
            f"<td>{e.bookmaker}</td>"
            f"<td>{float(e.price):.2f}</td>"
            f"<td>{prob_val*100:.1f}% {conf_badge}</td>"
            f"<td><b>{float(e.edge)*100:.1f}%</b></td>"
            f"<td class='muted'>{e.model_source}</td>"

            # "Why?" button
            f"<td><button type='button' onclick=\"showExplain({f.id}, '{e.market}')\">Why?</button></td>"

            f"<td>"
            f"<form hx-post='/bets/new' hx-target='#status' hx-swap='innerHTML' "
            f"style='display:flex;gap:6px;align-items:center'>"
            f"<input type='hidden' name='fixture_id' value='{f.id}' />"
            f"<input type='hidden' name='market' value='{e.market}' />"
            f"<input type='hidden' name='bookmaker' value='{e.bookmaker}' />"
            f"<input type='hidden' name='price' value='{float(e.price):.2f}' />"
            f"<input type='number' name='stake' step='0.01' min='0' "
            f"placeholder='Stake' style='width:90px' required />"
            f"<button type='submit'>Add</button>"
            f"</form>"
            f"</td>"

            f"<td>"
            f"<button onclick=\""
            f"  const el = document.getElementById('form-{f.id}');"
            f"  el.classList.toggle('hidden');"
            f"  if (!el.dataset.loaded) {{"
            f"    htmx.ajax('GET', '/form/fixture?fixture_id={f.id}', '#form-{f.id}');"
            f"    el.dataset.loaded = 'true';"
            f"  }}"
            f"\">Form</button>"
            f"</td>"
            f"</tr>"
            f"<tr><td colspan='12'>"
            f"<div id='form-{f.id}' class='hidden' "
            f"style='transition:all 0.3s ease;padding:4px;'></div>"
            f"</td></tr>"
        )

    html_rows.append("""
    <style>
    .hidden { display: none; }
    </style>
    """)

    return HTMLResponse("".join(html_rows))

@router.get("/fixtures", response_class=HTMLResponse)
def fixtures_index(
    db: Session = Depends(get_db),
    hours_ahead: int = Query(72, ge=1, le=168),
    comp: str | None = None,
):
    now = datetime.now(timezone.utc)
    until = now + timedelta(hours=hours_ahead)
    q = (
        db.query(Fixture)
        .filter(Fixture.kickoff_utc >= now, Fixture.kickoff_utc <= until)
        .order_by(Fixture.kickoff_utc.asc())
    )
    if comp:
        q = q.filter(Fixture.comp == comp)
    fixtures = q.all()
    return templates.TemplateResponse(
        "fixtures.html",
        {"request": {}, "fixtures": fixtures, "hours_ahead": hours_ahead, "comp": comp or ""},
    )

@router.get("/fixtures/{fixture_id}", response_class=HTMLResponse)
def fixture_detail(
    fixture_id: int,
    db: Session = Depends(get_db),
):
    f = db.query(Fixture).filter(Fixture.id == fixture_id).one_or_none()
    if not f:
        return HTMLResponse("Fixture not found", status_code=404)

    from sqlalchemy import and_

    sub = (
        db.query(Edge.market, func.max(Edge.edge).label("max_edge"))
        .filter(Edge.fixture_id == fixture_id)
        .group_by(Edge.market)
        .subquery()
    )
    best = (
        db.query(Edge)
        .join(sub, and_(Edge.market == sub.c.market, Edge.edge == sub.c.max_edge))
        .filter(Edge.fixture_id == fixture_id)
        .order_by(Edge.edge.desc())
        .all()
    )

    odds = (
        db.query(Odds)
        .filter(Odds.fixture_id == fixture_id)
        .order_by(Odds.market.asc(), Odds.last_seen.desc())
        .all()
    )

    return templates.TemplateResponse(
        "fixture.html",
        {"request": {}, "fixture": f, "best": best, "odds": odds},
    )

@router.get("/fixtures/{fixture_id}/_best", response_class=HTMLResponse)
def fixture_best_partial(fixture_id: int, db: Session = Depends(get_db)):
    from sqlalchemy import and_
    sub = (
        db.query(Edge.market, func.max(Edge.edge).label("max_edge"))
        .filter(Edge.fixture_id == fixture_id)
        .group_by(Edge.market)
        .subquery()
    )
    best = (
        db.query(Edge)
        .join(sub, and_(Edge.market == sub.c.market, Edge.edge == sub.c.max_edge))
        .filter(Edge.fixture_id == fixture_id)
        .order_by(Edge.edge.desc())
        .all()
    )
    if not best:
        return HTMLResponse("<tr><td colspan='5'>No edges yet for this fixture.</td></tr>")

    rows = [
        f"<tr><td><span class='pill'>{e.market}</span></td>"
        f"<td>{e.bookmaker}</td>"
        f"<td>{float(e.price):.2f}</td>"
        f"<td>{float(e.prob)*100:.1f}%</td>"
        f"<td><b>{float(e.edge)*100:.1f}%</b></td></tr>"
        for e in best
    ]
    return HTMLResponse("".join(rows))

@router.get("/fixtures/{fixture_id}/_odds", response_class=HTMLResponse)
def fixture_odds_partial(fixture_id: int, db: Session = Depends(get_db)):
    odds = (
        db.query(Odds)
        .filter(Odds.fixture_id == fixture_id)
        .order_by(Odds.market.asc(), Odds.bookmaker.asc(), Odds.last_seen.desc())
        .all()
    )
    if not odds:
        return HTMLResponse("<tr><td colspan='4'>No odds stored yet.</td></tr>")

    rows = [
        f"<tr><td>{o.market}</td><td>{o.bookmaker}</td>"
        f"<td>{float(o.price):.2f}</td>"
        f"<td class='muted'>{o.last_seen:%Y-%m-%d %H:%M}</td></tr>"
        for o in odds
    ]
    return HTMLResponse("".join(rows))