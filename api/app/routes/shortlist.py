# api/app/routes/shortlist.py
from __future__ import annotations
import re
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import List, Optional, Set, Dict, Tuple

from fastapi import APIRouter, Depends, Query, HTTPException, Body
from sqlalchemy.orm import Session
from sqlalchemy import or_

from ..db import get_db
from ..models import Edge, Fixture, Bet
from ..edge import ensure_baseline_probs, compute_edges
from ..settings import settings
from ..telegram_alert import send_telegram_alert  # legacy "today" send
from ..services.telegram import send_alert_message  # manual single send

router = APIRouter(prefix="", tags=["shortlist"])

# ----------------------- helpers -----------------------

def _norm_book_name(name: str | None) -> str:
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]+", "", name.lower())

# --- helpers (put near the top of shortlist.py, after _norm_book_name) ---
def _same_bookmaker(a: Optional[str], b: Optional[str]) -> bool:
    """Portable equality: lowercase and strip spaces/dashes/underscores/dots."""
    def norm(x: Optional[str]) -> str:
        s = (x or "").lower()
        for ch in (" ", "-", "_", "."):
            s = s.replace(ch, "")
        return s
    return norm(a) == norm(b)

def _safe_float(x):
    try:
        return float(x) if x is not None else None
    except Exception:
        return None

def _alert_digest(fixture_id: int, market: str, bookmaker: str, price: float) -> str:
    alert_key = f"{fixture_id}|{market}|{bookmaker}|{price:.4f}"
    return sha256(alert_key.encode()).hexdigest()

def _implied_prob(price: Optional[float]) -> Optional[float]:
    try:
        p = float(price)
        return 1.0 / p if p > 0 else None
    except Exception:
        return None

def _market_key(m: Optional[str]) -> str:
    return (m or "").strip().upper()

# ----------------------- shortlist (raw) -----------------------

@router.get("/shortlist/today")
def shortlist_today(
    db: Session = Depends(get_db),
    hours_ahead: int = Query(96, ge=1, le=14*24),
    min_edge: float = Query(0.00, ge=-1.0, le=1.0),
    send_alerts: int = Query(0, description="(deprecated) 1=send Telegram alerts immediately; default 0"),
    prefer_book: Optional[str] = Query(None),
    leagues: Optional[str] = Query(None),
    exclude_started: bool = Query(True, description="Hide fixtures already kicked off unless unsettled"),
    grace_mins: int = Query(10, ge=0, le=120, description="In-play grace for 'exclude_started'"),
):
    """
    Returns shortlist edges for the next N hours.
    Includes alert helpers (already_sent, alert_hash, alert_payload).
    """
    now = datetime.now(timezone.utc)
    until = now + timedelta(hours=hours_ahead)

    q = (
        db.query(Edge, Fixture)
        .join(Fixture, Edge.fixture_id == Fixture.id)
        .filter(
            or_(
                Fixture.kickoff_utc >= now,       # upcoming
                Fixture.result_settled == False   # in-progress / unsettled
            )
        )
        .filter(Fixture.kickoff_utc <= until)
        .filter(Edge.edge >= min_edge)
        .order_by(Edge.edge.desc(), Edge.created_at.desc())
    )

    rows = q.all()

    # Python-side filters (portable)
    if leagues:
        wanted = {s.strip() for s in leagues.split(",") if s.strip()}
        rows = [(e, f) for (e, f) in rows if (f.comp or "") in wanted]

    if prefer_book:
        rows = [(e, f) for (e, f) in rows if _same_bookmaker(e.bookmaker, prefer_book)]

    if exclude_started:
        cutoff = now - timedelta(minutes=grace_mins)
        rows = [(e, f) for (e, f) in rows if ((f.kickoff_utc or now) >= cutoff) or (not f.result_settled)]

    out: list[dict] = []
    for e, f in rows:
        price = _safe_float(e.price)
        if price is None:
            # skip edges with no usable price (prevents 500 + not actionable)
            continue

        prob_val = _safe_float(e.prob)
        edge_val = _safe_float(e.edge)

        digest = _alert_digest(f.id, e.market, e.bookmaker, price)
        dup = db.query(Bet).filter(Bet.duplicate_alert_hash == digest).first()

        kickoff_iso = f.kickoff_utc.isoformat() if f.kickoff_utc else None

        row = {
            "fixture_id": f.id,
            "comp": f.comp,
            "home_team": f.home_team,
            "away_team": f.away_team,
            "kickoff_utc": kickoff_iso,
            "market": e.market,
            "bookmaker": e.bookmaker,
            "price": price,
            "prob": prob_val,
            "edge": edge_val,
            "model_source": e.model_source,
            # UI helpers
            "already_sent": bool(dup),
            "alert_hash": digest,
            "alert_payload": {
                "match": f"{f.home_team} v {f.away_team}",
                "market": e.market,
                "odds": price,
                "edge": (edge_val * 100.0) if edge_val is not None else None,
                "kickoff": f.kickoff_utc.strftime("%a %d %b, %H:%M") if f.kickoff_utc else None,
                "league": f.comp,
                "bookmaker": e.bookmaker,
                "model_source": e.model_source,
                "link": None,
                "bet_id": f"{f.id}-{(e.market or '').replace(' ', '')}-{e.bookmaker}",
            },
        }
        out.append(row)

    # Deprecated: blast everything (still safe-casted)
    if send_alerts:
        for e, f in rows:
            edge_val = _safe_float(e.edge)
            if (edge_val or 0.0) < 0.0:
                continue
            price = _safe_float(e.price)
            if price is None:
                continue
            digest = _alert_digest(f.id, e.market, e.bookmaker, price)
            exists = db.query(Bet).filter(Bet.duplicate_alert_hash == digest).first()
            if exists:
                continue
            try:
                send_telegram_alert(
                    match=f"{f.home_team} v {f.away_team}",
                    market=e.market,
                    odds=price,
                    edge=(edge_val * 100.0) if edge_val is not None else None,
                    kickoff=f.kickoff_utc.strftime("%a %d %b, %H:%M") if f.kickoff_utc else None,
                    league=f.comp,
                    bookmaker=e.bookmaker,
                    model_source=e.model_source,
                    link=None,
                    bet_id=f"{f.id}-{(e.market or '').replace(' ', '')}-{e.bookmaker}",
                )
            except Exception as te:
                print(f"[Telegram] send error: {te!r}")
            db.add(Bet(
                fixture_id=f.id,
                market=e.market,
                bookmaker=e.bookmaker,
                price=price,
                stake=0.0,
                placed_at=datetime.now(timezone.utc),
                duplicate_alert_hash=digest,
            ))
            db.commit()

    return out

@router.post("/shortlist/send-batch")
def send_shortlist_batch(
    payload: dict = Body(..., example={
        "items": [
            {"fixture_id": 1234, "market": "Over 2.5", "bookmaker": "Bet365", "price": 1.91}
        ],
        "dry_run": False
    }),
    db: Session = Depends(get_db),
):
    items = payload.get("items") or []
    dry_run = bool(payload.get("dry_run", False))
    if not isinstance(items, list) or not items:
        raise HTTPException(status_code=400, detail="No items to send")

    results = []
    for it in items:
        try:
            fid = int(it["fixture_id"])
            market = str(it["market"])
            bookmaker = str(it["bookmaker"])
            price = float(it["price"])
        except Exception:
            results.append({"ok": False, "reason": "bad item fields", "item": it})
            continue

        f = db.query(Fixture).filter(Fixture.id == fid).first()
        if not f:
            results.append({"ok": False, "reason": "fixture not found", "item": it})
            continue

        digest = _alert_digest(fid, market, bookmaker, price)
        dup = db.query(Bet).filter(Bet.duplicate_alert_hash == digest).first()
        if dup:
            results.append({"ok": False, "reason": "duplicate", "item": it, "duplicate": True})
            continue

        payload_obj = it.get("alert_payload") or {
            "match": f"{f.home_team} v {f.away_team}",
            "market": market,
            "odds": price,
            "edge": None,
            "kickoff": f.kickoff_utc.strftime("%a %d %b, %H:%M") if f.kickoff_utc else None,
            "league": f.comp,
            "bookmaker": bookmaker,
            "model_source": None,
            "link": None,
            "bet_id": f"{fid}-{market.replace(' ', '')}-{bookmaker}",
        }

        if dry_run:
            results.append({"ok": True, "sent": False, "dry_run": True, "item": it})
            continue

        try:
            send_alert_message(payload_obj)
            db.add(Bet(
                fixture_id=fid,
                market=market,
                bookmaker=bookmaker,
                price=price,
                stake=0.0,
                placed_at=datetime.now(timezone.utc),
                duplicate_alert_hash=digest,
            ))
            db.commit()
            results.append({"ok": True, "sent": True, "item": it})
        except Exception as e:
            results.append({"ok": False, "reason": f"send error: {e}", "item": it})

    return {"ok": True, "count": len(results), "results": results}

@router.post("/compute")
def compute_now(db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc)
    source = "team_form"
    ensure_baseline_probs(db, now, source=source)
    compute_edges(db, now, settings.EDGE_MIN, source=source)
    return {"ok": True, "source": source}

@router.post("/telegram/send-alert")
def send_alert(alert: dict):
    return send_alert_message(alert)

# ====================== BEST EDGES & FEATURED PICKS ======================

@router.get("/shortlist/best")
def shortlist_best(
    day: str = Query(datetime.now(timezone.utc).date().isoformat(), description="YYYY-MM-DD (UTC)"),
    hours_ahead: Optional[int] = Query(None, ge=1, le=14*24, description="If set, ignore 'day' and use now..now+hours"),
    sport: Optional[str] = Query(None, description="Filter by Fixture.sport"),
    model: Optional[str] = Query("team_form", description="Edge.model_source"),
    min_edge: float = Query(0.03, ge=-1.0, le=1.0, description="e.g. 0.04 = 4%"),
    markets: Optional[str] = Query(None, description="Comma list e.g. 'HOME_WIN,AWAY_WIN,O2.5,BTTS_Y'"),
    bookmaker: Optional[str] = Query(None, description="Bookmaker name (normalized)"),
    per_fixture: bool = Query(True, description="Only the top edge per fixture"),
    prefer_distinct_leagues: bool = Query(False, description="Avoid multiple from same league"),
    exclude_started: bool = Query(True),
    grace_mins: int = Query(10, ge=0, le=120),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    now = datetime.now(timezone.utc)
    if hours_ahead:
        start, end = now, now + timedelta(hours=hours_ahead)
    else:
        day_obj = datetime.fromisoformat(day).date()
        start = datetime.combine(day_obj, datetime.min.time(), tzinfo=timezone.utc)
        end = start + timedelta(days=1)

    q = (
        db.query(Edge, Fixture)
        .join(Fixture, Fixture.id == Edge.fixture_id)
        .filter(Fixture.kickoff_utc >= start, Fixture.kickoff_utc < end)
        .filter(Edge.edge >= min_edge)
        .order_by(Edge.edge.desc(), Edge.created_at.desc())
    )
    rows = q.all()

    # Python-side filters
    if sport:
        rows = [(e, f) for (e, f) in rows if (f.sport or "football") == sport]
    if model:
        rows = [(e, f) for (e, f) in rows if (e.model_source or "") == model]
    if markets:
        allowed = {_market_key(m) for m in markets.split(",") if m.strip()}
        rows = [(e, f) for (e, f) in rows if _market_key(e.market) in allowed]
    if bookmaker:
        rows = [(e, f) for (e, f) in rows if _same_bookmaker(e.bookmaker, bookmaker)]
    if exclude_started:
        cutoff = now - timedelta(minutes=grace_mins)
        rows = [(e, f) for (e, f) in rows if (f.kickoff_utc or now) >= cutoff or (not f.result_settled)]

    # Deduplicate per fixture + optional distinct leagues
    payload = []
    seen_fx: Set[int] = set()
    seen_leagues: Set[str] = set()

    for e, f in rows:
        if per_fixture and f.id in seen_fx:
            continue
        if prefer_distinct_leagues and f.comp in seen_leagues:
            continue
        price = _safe_float(e.price)
        item = {
            "fixture_id": f.id,
            "kickoff_utc": f.kickoff_utc.isoformat() if f.kickoff_utc else None,
            "league": f.comp,
            "home": f.home_team,
            "away": f.away_team,
            "market": e.market,
            "bookmaker": e.bookmaker,
            "price": price,
            "model_prob": _safe_float(e.prob),
            "implied_prob": _implied_prob(price),
            "edge": _safe_float(e.edge),
            "model_source": e.model_source,
            "label": f"{f.home_team} vs {f.away_team} · {e.market} @{e.bookmaker} {price}",
            # Telegram/picks helpers
            "telegram_text": f"🔥 {f.comp} — {f.home_team} v {f.away_team}\n{e.market} @ {e.bookmaker} {price} (edge {(_safe_float(e.edge) or 0.0)*100:.1f}%)\nKO: {f.kickoff_utc.strftime('%a %d %b, %H:%M') if f.kickoff_utc else 'TBC'}",
            "pick_payload": {
                "day": start.date().isoformat(),
                "fixture_id": f.id,
                "sport": (f.sport or "football"),
                "title": f"{f.home_team} v {f.away_team} — {e.market}",
                "blurb": f"{e.bookmaker} {price} | model {(_safe_float(e.prob) or 0.0)*100:.1f}% | fair { (1.0/(_safe_float(e.prob) or 1.0)):.2f } | edge {(_safe_float(e.edge) or 0.0)*100:.1f}%",
                "is_public": False,
            },
        }
        payload.append(item)
        seen_fx.add(f.id)
        seen_leagues.add(f.comp)
        if len(payload) >= limit:
            break

    payload.sort(key=lambda r: (-(r["edge"] or 0.0), r["kickoff_utc"] or ""))

    return {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "sport": sport,
        "model": model,
        "min_edge": min_edge,
        "markets": [m.strip().upper() for m in markets.split(",")] if markets else None,
        "per_fixture": per_fixture,
        "prefer_distinct_leagues": prefer_distinct_leagues,
        "count": len(payload),
        "rows": payload[:limit],
    }

@router.get("/shortlist/featured")
def shortlist_featured(
    top_n: int = Query(5, ge=1, le=20),
    **kwargs,
):
    """
    Wrapper over /shortlist/best returning the top-N edges across fixtures,
    with Telegram text and /admin/picks payloads.
    Accepts all query params of /shortlist/best (pass-through via **kwargs).
    """
    # Reuse shortlist_best
    res = shortlist_best(**kwargs)
    rows = res.get("rows", [])[:top_n]
    return {
        "meta": {"top_n": top_n, **{k: v for k, v in res.items() if k != "rows"}},
        "featured": rows,
    }

# ====================== ACCA SUGGEST ======================

@router.get("/shortlist/acca/suggest")
def shortlist_acca_suggest(
    day: str = Query(datetime.now(timezone.utc).date().isoformat(), description="YYYY-MM-DD (UTC)"),
    hours_ahead: Optional[int] = Query(None, ge=1, le=14*24),
    sport: Optional[str] = Query(None),
    model: Optional[str] = Query("team_form"),
    min_edge: float = Query(0.03, ge=-1.0, le=1.0),
    markets: Optional[str] = Query("HOME_WIN,AWAY_WIN,O2.5,BTTS_Y", description="Market whitelist"),
    legs: int = Query(4, ge=2, le=12, description="Number of legs"),
    prefer_distinct_leagues: bool = Query(False, description="Try to avoid multiple legs from same league"),
    bankroll: Optional[float] = Query(None, ge=0.0, description="If provided, returns Kelly stake (full Kelly)"),
    kelly_fraction: float = Query(0.5, ge=0.0, le=1.0, description="Fractional Kelly to apply"),
    target_odds: float = Query(5.0, ge=1.1, description="For admin_acca_auto_hint only"),
    bookmaker_hint: Optional[str] = Query("bet365", description="For admin_acca_auto_hint only"),
    db: Session = Depends(get_db),
):
    # Reuse shortlist_best
    params = {
        "day": day,
        "hours_ahead": hours_ahead,
        "sport": sport,
        "model": model,
        "min_edge": min_edge,
        "markets": markets,
        "per_fixture": True,
        "limit": 500,
    }
    result = shortlist_best(**params, db=db)
    rows: List[Dict] = result["rows"]

    if not rows:
        return {"ok": True, "legs": [], "reason": "no rows", "window": result["window"]}

    chosen: List[Dict] = []
    seen_fixtures: Set[int] = set()
    seen_leagues: Set[str] = set()

    for r in rows:
        if r["fixture_id"] in seen_fixtures:
            continue
        if prefer_distinct_leagues and r["league"] in seen_leagues:
            continue
        chosen.append(r)
        seen_fixtures.add(r["fixture_id"])
        seen_leagues.add(r["league"])
        if len(chosen) >= legs:
            break

    combined_price = 1.0
    combined_prob = 1.0
    for leg in chosen:
        p = leg.get("price")
        mp = leg.get("model_prob")
        if p:
            combined_price *= float(p)
        if mp is not None:
            combined_prob *= float(mp)

    kelly_full = None
    kelly_stake = None
    acca_edge = None
    if combined_prob is not None and combined_price:
        q = 1.0 - combined_prob
        b = combined_price - 1.0
        acca_edge = combined_prob * (combined_price) - 1.0
        try:
            kelly_full = ((b * combined_prob) - q) / b if b > 0 else None
        except Exception:
            kelly_full = None
        if bankroll is not None and kelly_full is not None:
            kelly_stake = max(0.0, kelly_full * kelly_fraction * bankroll)

    # Handy hint to POST to /admin/accas/auto
    admin_acca_auto_hint = {
        "day": day,
        "bookmaker": bookmaker_hint,
        "min_edge": max(0.01, min_edge),
        "target_odds": target_odds,
        "legs_min": legs,
        "legs_max": legs,
        "diversify_markets": True,
        "avoid_same_fixture": True,
        "is_public": False,
        "title": f"Auto ACCA ({legs} legs)",
        # keep the default price band wide; the /admin/accas/auto route will refine
        "min_price": 1.3,
        "max_price": 3.0,
        "target_tolerance_pct": 0.50,
        "search_pool_size": 24,
    }

    return {
        "ok": True,
        "legs": chosen,
        "combined_price": round(combined_price, 4) if chosen else None,
        "combined_prob": round(combined_prob, 6) if chosen else None,
        "acca_edge": round(acca_edge, 4) if acca_edge is not None else None,
        "kelly_full": round(kelly_full, 4) if kelly_full is not None else None,
        "kelly_fraction": kelly_fraction,
        "kelly_stake": round(kelly_stake, 2) if kelly_stake is not None else None,
        "window": result["window"],
        "model": model,
        "markets": markets.split(",") if markets else None,
        "min_edge": min_edge,
        "prefer_distinct_leagues": prefer_distinct_leagues,
        "admin_acca_auto_hint": admin_acca_auto_hint,
    }