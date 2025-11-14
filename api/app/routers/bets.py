from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Form, HTTPException, Body
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from typing import Optional, List
from datetime import datetime, timezone

from ..db import get_db
from ..models import Bet, Fixture, UserBet, User
from ..auth_firebase import get_current_user

router = APIRouter(tags=["bets"])
templates = Jinja2Templates(directory="api/app/templates")


# -------------------- helpers --------------------

def _p2f(x):
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def _serialize_bet(b: Bet) -> dict:
    """JSON-friendly row for the frontend (Bet-only; comp/teams may be None)."""
    return {
        "id": b.id,
        "fixture_id": b.fixture_id,
        "market": b.market,
        "bookmaker": b.bookmaker,
        "price": _p2f(b.price),
        "stake": _p2f(b.stake),
        "ret": _p2f(b.ret),
        "pnl": _p2f(b.pnl),
        "result": b.result or None,  # PENDING|WON|LOST|VOID -> None means pending in FE
        "placed_at": b.placed_at.isoformat() if getattr(b, "placed_at", None) else None,
        # These are not columns on Bet in your schema; keep as optional for compatibility
        "comp": getattr(b, "comp", None),
        "teams": getattr(b, "teams", None),
        "notes": getattr(b, "notes", None),
    }


def render_bet_row(bet: Bet, fx: Fixture) -> str:
    price = _p2f(bet.price) or 0.0
    stake = _p2f(bet.stake) or 0.0
    ret   = _p2f(bet.ret)
    pnl   = _p2f(bet.pnl)

    ret_txt = "" if ret is None else f"{ret:.2f}"
    pnl_txt = "" if pnl is None else f"{pnl:.2f}"

    def opt(val):
        sel = "selected" if (bet.result or "PENDING") == val else ""
        return f"<option value='{val}' {sel}>{val}</option>"

    return (
        f"<tr id='bet-{bet.id}'>"
        f"<td>{fx.comp}</td>"
        f"<td><a href='/fixtures/{fx.id}'>{fx.home_team} vs {fx.away_team}</a></td>"
        f"<td>{bet.market}</td>"
        f"<td>{bet.bookmaker}</td>"
        f"<td>{price:.2f}</td>"
        f"<td>"
          f"<form hx-post='/bets/{bet.id}/edit' hx-target='#bet-{bet.id}' hx-swap='outerHTML' style='display:flex;gap:6px'>"
            f"<input type='number' name='stake' step='0.01' min='0' value='{stake:.2f}' style='width:90px'/>"
            f"<button type='submit'>Save</button>"
          f"</form>"
        f"</td>"
        f"<td>{ret_txt}</td>"
        f"<td>{pnl_txt}</td>"
        f"<td>"
          f"<form hx-post='/bets/{bet.id}/settle' hx-target='#bet-{bet.id}' hx-swap='outerHTML' style='display:flex;gap:6px'>"
            f"<select name='result'>"
              f"{opt('PENDING')}{opt('WON')}{opt('LOST')}{opt('VOID')}"
            f"</select>"
            f"<button type='submit'>Update</button>"
          f"</form>"
        f"</td>"
        f"<td>"
          f"<form hx-post='/bets/{bet.id}/delete' hx-target='#bet-{bet.id}' hx-swap='outerHTML'>"
            f"<button type='submit' onclick=\"return confirm('Delete bet?')\">🗑</button>"
          f"</form>"
        f"</td>"
        f"</tr>"
    )


# -------------------- HTML (existing) --------------------

@router.get("/bets", response_class=HTMLResponse)
def bets_page(request: Request, db: Session = Depends(get_db)):
    rows = (
        db.query(Bet, Fixture)
        .join(Fixture, Fixture.id == Bet.fixture_id)
        .order_by(Bet.placed_at.desc())
        .all()
    )
    return templates.TemplateResponse("bets.html", {"request": request, "rows": rows})


@router.post("/bets/new")
def create_bet(
    request: Request,
    fixture_id: int = Form(...),
    market: str = Form(...),
    bookmaker: str = Form(...),
    price: float = Form(...),
    stake: float = Form(...),
    db: Session = Depends(get_db),
):
    bet = Bet(
        fixture_id=fixture_id,
        market=market,
        bookmaker=bookmaker,
        price=price,
        stake=stake,
        placed_at=datetime.now(timezone.utc),
    )
    db.add(bet)
    db.commit()

    if request.headers.get("HX-Request") == "true":
        return HTMLResponse(f"✅ Added bet: {market} @ {price:.2f} (stake {stake:.2f})", status_code=201)
    return RedirectResponse(url="/bets", status_code=303)


@router.post("/bets/{bet_id}/edit", response_class=HTMLResponse)
def edit_bet(
    bet_id: int,
    stake: float = Form(...),
    db: Session = Depends(get_db),
):
    bet = db.query(Bet).filter(Bet.id == bet_id).one_or_none()
    if not bet:
        return HTMLResponse("<tr><td colspan='10'>Bet not found</td></tr>", status_code=404)

    bet.stake = stake
    # recompute if settled
    if bet.result == "WON":
        bet.ret = (float(bet.price) * float(bet.stake))
        bet.pnl = bet.ret - float(bet.stake)
    elif bet.result == "LOST":
        bet.ret = 0.0
        bet.pnl = -float(bet.stake)
    elif bet.result == "VOID":
        bet.ret = float(bet.stake)
        bet.pnl = 0.0
    else:
        bet.ret = None
        bet.pnl = None

    db.commit()
    fx = db.query(Fixture).filter(Fixture.id == bet.fixture_id).one()
    return HTMLResponse(render_bet_row(bet, fx))


@router.post("/bets/{bet_id}/settle", response_class=HTMLResponse)
def settle_bet(
    bet_id: int,
    result: str = Form(...),  # PENDING/WON/LOST/VOID
    db: Session = Depends(get_db),
):
    bet = db.query(Bet).filter(Bet.id == bet_id).one_or_none()
    if not bet:
        return HTMLResponse("<tr><td colspan='10'>Bet not found</td></tr>", status_code=404)

    r = (result or "PENDING").upper()
    stake = float(bet.stake or 0)
    price = float(bet.price or 0)

    if r == "WON":
        ret = stake * price
        pnl = ret - stake
    elif r == "LOST":
        ret = 0.0
        pnl = -stake
    elif r == "VOID":
        ret = stake
        pnl = 0.0
    else:  # PENDING
        ret = None
        pnl = None

    bet.result = r
    bet.ret = ret
    bet.pnl = pnl
    db.commit()

    fx = db.query(Fixture).filter(Fixture.id == bet.fixture_id).one()
    return HTMLResponse(render_bet_row(bet, fx))


@router.post("/bets/{bet_id}/delete", response_class=HTMLResponse)
def delete_bet(
    bet_id: int,
    db: Session = Depends(get_db),
):
    bet = db.query(Bet).filter(Bet.id == bet_id).one_or_none()
    if not bet:
        return HTMLResponse("", status_code=204)
    db.delete(bet)
    db.commit()
    return HTMLResponse("")


# -------------------- JSON (for React - global Bet) --------------------

class BetIn(BaseModel):
    fixture_id: Optional[int] = None
    market: str
    bookmaker: str
    price: float
    stake: float
    # `comp` & `teams` can be passed by the FE, but we DO NOT store them on Bet.
    comp: Optional[str] = None
    teams: Optional[str] = None
    notes: Optional[str] = None


class BetPatch(BaseModel):
    # allow partial updates
    stake: Optional[float] = None
    result: Optional[str] = None   # "won"|"lost"|"push"|"pending" or "PENDING"/"WON"/"LOST"/"VOID"
    notes: Optional[str] = None


def _bet_to_json(bet: Bet, fx: Fixture | None) -> dict:
    """Shape used by the React BetTracker table (legacy/global)."""
    teams = f"{fx.home_team} vs {fx.away_team}" if fx else getattr(bet, "teams", None)
    comp  = fx.comp if fx else getattr(bet, "comp", None)
    return {
        "id": bet.id,
        "fixture_id": bet.fixture_id,
        "teams": teams,
        "comp": comp,
        "market": bet.market,
        "bookmaker": bet.bookmaker,
        "price": float(bet.price or 0),
        "stake": float(bet.stake or 0),
        "result": bet.result or None,
        "notes": getattr(bet, "notes", None),
        "ret": None if bet.ret is None else float(bet.ret),
        "pnl": None if bet.pnl is None else float(bet.pnl),
        "placed_at": bet.placed_at.isoformat() if getattr(bet, "placed_at", None) else None,
    }


@router.get("/bets.json")
def list_bets_json(db: Session = Depends(get_db)) -> List[dict]:
    """
    Return Bet rows joined to Fixture so the FE gets `teams` and `comp`
    without needing those columns on Bet.
    """
    rows = (
        db.query(Bet, Fixture)
        .outerjoin(Fixture, Fixture.id == Bet.fixture_id)
        .order_by(Bet.placed_at.desc())
        .all()
    )
    out = []
    for b, f in rows:
        out.append(_bet_to_json(b, f))
    return out


@router.post("/bets.json")
def create_bet_json(payload: BetIn, db: Session = Depends(get_db)) -> dict:
    """
    Ignore unknown Bet fields (`comp`, `teams`) to avoid SQLA TypeError.
    """
    b = Bet(
        fixture_id=payload.fixture_id,
        market=payload.market,
        bookmaker=payload.bookmaker,
        price=float(payload.price),
        stake=float(payload.stake),
        placed_at=datetime.now(timezone.utc),
        # notes is common; include if your model has it
        **({"notes": payload.notes} if hasattr(Bet, "notes") and payload.notes is not None else {}),
    )
    db.add(b)
    db.commit()
    db.refresh(b)

    fx = db.query(Fixture).filter(Fixture.id == b.fixture_id).one_or_none()
    return _bet_to_json(b, fx)


@router.patch("/bets/{bet_id}.json")
def patch_bet_json(bet_id: int, payload: BetPatch, db: Session = Depends(get_db)) -> dict:
    b = db.query(Bet).filter(Bet.id == bet_id).one_or_none()
    if not b:
        raise HTTPException(status_code=404, detail="Bet not found")

    if payload.stake is not None:
        b.stake = float(payload.stake)

    if payload.notes is not None and hasattr(Bet, "notes"):
        b.notes = payload.notes

    if payload.result is not None:
        r = payload.result.upper()
        # accept both push/void wordings
        if r not in ("WON", "LOST", "PENDING", "VOID", "PUSH"):
            raise HTTPException(status_code=400, detail="Invalid result")
        if r == "PUSH":
            r = "VOID"
        stake = float(b.stake or 0)
        price = float(b.price or 0)
        if r == "WON":
            b.ret = stake * price
            b.pnl = b.ret - stake
        elif r == "LOST":
            b.ret = 0.0
            b.pnl = -stake
        elif r == "VOID":
            b.ret = stake
            b.pnl = 0.0
        else:  # PENDING
            b.ret = None
            b.pnl = None
        b.result = r

    db.commit()
    db.refresh(b)
    fx = db.query(Fixture).filter(Fixture.id == b.fixture_id).one_or_none()
    return _bet_to_json(b, fx)


@router.delete("/bets/{bet_id}.json")
def delete_bet_json(bet_id: int, db: Session = Depends(get_db)) -> dict:
    b = db.query(Bet).filter(Bet.id == bet_id).one_or_none()
    if not b:
        raise HTTPException(status_code=404, detail="Bet not found")
    db.delete(b)
    db.commit()
    return {"ok": True}


# --- JSON helpers + cleanup endpoints (global Bet) --------------------------

def _bet_to_json_legacy(bet: Bet, fx: Fixture) -> dict:
    return {
        "id": bet.id,
        "fixture_id": bet.fixture_id,
        "match": f"{fx.home_team} vs {fx.away_team}",
        "comp": fx.comp,
        "kickoff_utc": fx.kickoff_utc,
        "market": bet.market,
        "bookmaker": bet.bookmaker,
        "price": float(bet.price or 0),
        "stake": float(bet.stake or 0),
        "result": bet.result or "PENDING",
        "ret": None if bet.ret is None else float(bet.ret),
        "pnl": None if bet.pnl is None else float(bet.pnl),
    }


@router.get("/bets/json")
def bets_json(db: Session = Depends(get_db)):
    rows = (
        db.query(Bet, Fixture)
        .join(Fixture, Fixture.id == Bet.fixture_id)
        .order_by(Bet.placed_at.desc())
        .all()
    )
    return [_bet_to_json_legacy(b, f) for (b, f) in rows]


@router.post("/bets/cleanup")
def bets_cleanup(
    action: str = Body(..., embed=True, description="delete_zero_pending | delete_pending | delete_all"),
    db: Session = Depends(get_db),
):
    """
    delete_zero_pending : delete PENDING bets with stake == 0
    delete_pending      : delete ALL PENDING bets (any stake)
    delete_all          : delete *all* bets (nuclear)
    """
    if action not in {"delete_zero_pending", "delete_pending", "delete_all"}:
        return {"ok": False, "deleted": 0, "detail": "unknown action"}

    q = db.query(Bet)
    if action == "delete_zero_pending":
        q = q.filter(or_(Bet.result == None, Bet.result == "PENDING"), Bet.stake == 0)  # noqa: E711
    elif action == "delete_pending":
        q = q.filter(or_(Bet.result == None, Bet.result == "PENDING"))  # noqa: E711
    else:  # delete_all
        pass

    to_delete = q.count()
    q.delete(synchronize_session=False)
    db.commit()
    return {"ok": True, "deleted": to_delete}


@router.post("/bets/bulk-delete")
def bets_bulk_delete(
    ids: list[int] = Body(..., embed=True),
    db: Session = Depends(get_db),
):
    if not ids:
        return {"ok": True, "deleted": 0}
    deleted = db.query(Bet).filter(Bet.id.in_(ids)).delete(synchronize_session=False)
    db.commit()
    return {"ok": True, "deleted": deleted}


# ===================== USER-BETS (per-account JSON API) =====================

class UserBetIn(BaseModel):
    fixture_id: Optional[int] = None
    market: str
    bookmaker: Optional[str] = None
    price: float
    stake: float
    notes: Optional[str] = None   # if you add notes column on UserBet


class UserBetPatch(BaseModel):
    stake: Optional[float] = None
    result: Optional[str] = None   # "won"|"lost"|"void"|"pending"
    notes: Optional[str] = None


def _get_or_create_user_from_claims(db: Session, claims: dict) -> User:
    """
    Map Firebase claims -> local User row.
    Adjust field names to match your User model.
    """
    uid = (claims.get("uid") or "").strip()
    email = (claims.get("email") or "").lower()

    if not uid and not email:
        raise HTTPException(400, "Missing Firebase UID/email")

    q = db.query(User)
    if hasattr(User, "firebase_uid"):
        q = q.filter(User.firebase_uid == uid)
    elif hasattr(User, "uid"):
        q = q.filter(User.uid == uid)
    elif hasattr(User, "email") and email:
        q = q.filter(User.email == email)

    user = q.first()
    if user:
        return user

    # Create a light user row if missing (adjust fields to your schema)
    kwargs = {}
    if hasattr(User, "firebase_uid"):
        kwargs["firebase_uid"] = uid
    if hasattr(User, "email") and email:
        kwargs["email"] = email

    user = User(**kwargs)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _userbet_to_json(b: UserBet, fx: Fixture | None) -> dict:
    teams = None
    comp = None
    if fx:
        teams = f"{fx.home_team} vs {fx.away_team}"
        comp = fx.comp

    return {
        "id": b.id,
        "fixture_id": b.fixture_id,
        "teams": teams,
        "comp": comp,
        "market": b.market,
        "bookmaker": b.bookmaker,
        "price": float(b.price or 0),
        "stake": float(b.stake or 0),
        "result": b.result or None,   # WON/LOST/VOID/None
        "ret": None if b.ret is None else float(b.ret),
        "pnl": None if b.pnl is None else float(b.pnl),
        "notes": getattr(b, "notes", None),
        "placed_at": b.placed_at.isoformat() if b.placed_at else None,
    }


@router.get("/api/user-bets")
def list_user_bets(
    db: Session = Depends(get_db),
    user_claims=Depends(get_current_user),
):
    u = _get_or_create_user_from_claims(db, user_claims)

    rows = (
        db.query(UserBet, Fixture)
        .outerjoin(Fixture, Fixture.id == UserBet.fixture_id)
        .filter(UserBet.user_id == u.id)
        .order_by(UserBet.placed_at.desc())
        .all()
    )

    return [_userbet_to_json(b, f) for (b, f) in rows]


@router.post("/api/user-bets")
def create_user_bet(
    payload: UserBetIn,
    db: Session = Depends(get_db),
    user_claims=Depends(get_current_user),
):
    u = _get_or_create_user_from_claims(db, user_claims)

    b = UserBet(
        user_id=u.id,
        fixture_id=payload.fixture_id,
        market=payload.market,
        bookmaker=payload.bookmaker,
        price=float(payload.price),
        stake=float(payload.stake),
        placed_at=datetime.now(timezone.utc),
    )
    if hasattr(UserBet, "notes") and payload.notes is not None:
        b.notes = payload.notes

    db.add(b)
    db.commit()
    db.refresh(b)

    fx = db.query(Fixture).filter(Fixture.id == b.fixture_id).one_or_none()
    return _userbet_to_json(b, fx)


@router.patch("/api/user-bets/{bet_id}")
def patch_user_bet(
    bet_id: int,
    payload: UserBetPatch,
    db: Session = Depends(get_db),
    user_claims=Depends(get_current_user),
):
    u = _get_or_create_user_from_claims(db, user_claims)

    b: UserBet | None = (
        db.query(UserBet)
        .filter(UserBet.id == bet_id, UserBet.user_id == u.id)
        .one_or_none()
    )
    if not b:
        raise HTTPException(status_code=404, detail="Bet not found")

    if payload.stake is not None:
        b.stake = float(payload.stake)

    if payload.notes is not None and hasattr(UserBet, "notes"):
        b.notes = payload.notes

    if payload.result is not None:
        r = payload.result.upper()
        if r not in ("WON", "LOST", "VOID", "PENDING"):
            raise HTTPException(status_code=400, detail="Invalid result")

        stake = float(b.stake or 0)
        price = float(b.price or 0)

        if r == "WON":
            b.ret = stake * price
            b.pnl = b.ret - stake
        elif r == "LOST":
            b.ret = 0.0
            b.pnl = -stake
        elif r == "VOID":
            b.ret = stake
            b.pnl = 0.0
        else:  # PENDING
            b.ret = None
            b.pnl = None

        # store None for pending, actual codes for settled
        b.result = None if r == "PENDING" else r

    db.commit()
    db.refresh(b)
    fx = db.query(Fixture).filter(Fixture.id == b.fixture_id).one_or_none()
    return _userbet_to_json(b, fx)


@router.delete("/api/user-bets/{bet_id}")
def delete_user_bet(
    bet_id: int,
    db: Session = Depends(get_db),
    user_claims=Depends(get_current_user),
):
    u = _get_or_create_user_from_claims(db, user_claims)

    b: UserBet | None = (
        db.query(UserBet)
        .filter(UserBet.id == bet_id, UserBet.user_id == u.id)
        .one_or_none()
    )
    if not b:
        raise HTTPException(status_code=404, detail="Bet not found")

    db.delete(b)
    db.commit()
    return {"ok": True, "deleted": bet_id}


@router.post("/api/user-bets/cleanup")
def cleanup_user_bets(
    action: str = Body(..., embed=True),
    db: Session = Depends(get_db),
    user_claims=Depends(get_current_user),
):
    """
    delete_zero_pending : delete PENDING bets with stake == 0
    delete_pending      : delete ALL PENDING bets
    delete_all          : delete *all* bets for this user
    """
    if action not in {"delete_zero_pending", "delete_pending", "delete_all"}:
        raise HTTPException(status_code=400, detail="unknown action")

    u = _get_or_create_user_from_claims(db, user_claims)

    q = db.query(UserBet).filter(UserBet.user_id == u.id)

    if action == "delete_zero_pending":
        q = q.filter(
            or_(UserBet.result == None, UserBet.result == "PENDING"),  # noqa: E711
            UserBet.stake == 0,
        )
    elif action == "delete_pending":
        q = q.filter(or_(UserBet.result == None, UserBet.result == "PENDING"))  # noqa: E711

    deleted = q.delete(synchronize_session=False)
    db.commit()
    return {"ok": True, "deleted": deleted}