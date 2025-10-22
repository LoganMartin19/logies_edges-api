# api/app/routes/shortlist.py
from __future__ import annotations
import re
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, HTTPException, Body
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from ..db import get_db
from ..models import Edge, Fixture, Bet, ModelProb
from ..schemas import EdgeOut
from ..edge import ensure_baseline_probs, compute_edges
from ..settings import settings
from ..telegram_alert import send_telegram_alert  # legacy "today" send
from ..services.telegram import send_alert_message  # manual single send

router = APIRouter(prefix="", tags=["shortlist"])

def _norm_book_name(name: str | None) -> str:
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]+", "", name.lower())

def _alert_digest(fixture_id: int, market: str, bookmaker: str, price: float) -> str:
    alert_key = f"{fixture_id}|{market}|{bookmaker}|{price:.4f}"
    return sha256(alert_key.encode()).hexdigest()

@router.get("/shortlist/today", response_model=List[EdgeOut])
def shortlist_today(
    db: Session = Depends(get_db),
    hours_ahead: int = Query(96, ge=1, le=14*24),
    min_edge: float = Query(0.00, ge=-1.0, le=1.0),
    send_alerts: int = Query(0, description="(deprecated) 1=send Telegram alerts immediately; default 0"),
    prefer_book: Optional[str] = Query(None),
    leagues: Optional[str] = Query(None),
):
    """
    Returns shortlist edges for the next N hours.
    - Does NOT send alerts by default.
    - Each row includes a ready-to-send `alert_payload` and `already_sent` bool
      so the frontend can let you cherry-pick and send later.
    """
    now = datetime.now(timezone.utc)
    until = now + timedelta(hours=hours_ahead)

    q = (
        db.query(Edge, Fixture)
        .join(Fixture, Edge.fixture_id == Fixture.id)
        .filter(
            or_(
                Fixture.kickoff_utc >= now,            # upcoming
                Fixture.result_settled == False         # in-progress
            )
        )
        .filter(Fixture.kickoff_utc <= until)
        .filter(Edge.edge >= min_edge)
        .order_by(Edge.edge.desc())
    )

    if leagues:
        wanted = [s.strip() for s in leagues.split(",") if s.strip()]
        if wanted:
            q = q.filter(Fixture.comp.in_(wanted))

    if prefer_book:
        norm = _norm_book_name(prefer_book)
        db_norm = func.replace(func.lower(Edge.bookmaker), " ", "")
        db_norm = func.replace(db_norm, "-", "")
        db_norm = func.replace(db_norm, "_", "")
        db_norm = func.replace(db_norm, ".", "")
        q = q.filter(db_norm == norm)

    rows = q.all()

    out: list[dict] = []
    for e, f in rows:
        price = float(e.price)
        digest = _alert_digest(f.id, e.market, e.bookmaker, price)
        dup = db.query(Bet).filter(Bet.duplicate_alert_hash == digest).first()

        # Standard EdgeOut fields
        row = EdgeOut(
            fixture_id=f.id,
            comp=f.comp,
            home_team=f.home_team,
            away_team=f.away_team,
            kickoff_utc=f.kickoff_utc,
            market=e.market,
            bookmaker=e.bookmaker,
            price=price,
            prob=float(e.prob),
            edge=float(e.edge),
            model_source=e.model_source,
        ).dict()

        # Extra UI helpers (NOT part of EdgeOut schema; included in the JSON)
        row["already_sent"] = bool(dup)
        row["alert_hash"] = digest
        row["alert_payload"] = {
            "match": f"{f.home_team} v {f.away_team}",
            "market": e.market,
            "odds": price,
            "edge": float(e.edge) * 100.0,
            "kickoff": f.kickoff_utc.strftime("%a %d %b, %H:%M"),
            "league": f.comp,
            "bookmaker": e.bookmaker,
            "model_source": e.model_source,
            "link": None,
            "bet_id": f"{f.id}-{e.market.replace(' ', '')}-{e.bookmaker}",
        }
        out.append(row)

    # Deprecated: legacy path that sent everything immediately
    if send_alerts:
        for e, f in rows:
            if float(e.edge) < 0.0:
                continue
            price = float(e.price)
            digest = _alert_digest(f.id, e.market, e.bookmaker, price)
            exists = db.query(Bet).filter(Bet.duplicate_alert_hash == digest).first()
            if exists:
                continue
            try:
                send_telegram_alert(
                    match=f"{f.home_team} v {f.away_team}",
                    market=e.market,
                    odds=price,
                    edge=float(e.edge) * 100.0,
                    kickoff=f.kickoff_utc.strftime("%a %d %b, %H:%M"),
                    league=f.comp,
                    bookmaker=e.bookmaker,
                    model_source=e.model_source,
                    link=None,
                    bet_id=f"{f.id}-{e.market.replace(' ', '')}-{e.bookmaker}",
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
            {
                "fixture_id": 1234,
                "market": "Over 2.5",
                "bookmaker": "Bet365",
                "price": 1.91,
                # You can optionally include alert_payload to avoid
                # the server re-assembling strings:
                # "alert_payload": {...}
            }
        ],
        "dry_run": False
    }),
    db: Session = Depends(get_db),
):
    """
    Accepts a selection of shortlist rows and sends Telegram alerts for those only.
    Duplicate protection via hash. Returns a per-item status report.
    """
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

        # fetch fixture for text fields if alert_payload not supplied
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
            "edge": None,  # optional
            "kickoff": f.kickoff_utc.strftime("%a %d %b, %H:%M"),
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
            # store duplicate guard
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
    source = "team_form"  # keep your current model source
    ensure_baseline_probs(db, now, source=source)
    compute_edges(db, now, settings.EDGE_MIN, source=source)
    return {"ok": True, "source": source}

@router.post("/telegram/send-alert")
def send_alert(alert: dict):
    return send_alert_message(alert)