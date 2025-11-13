# api/app/routers/tipsters.py
from __future__ import annotations

from datetime import date, datetime, timezone as _tz
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import (
    Tipster,
    TipsterPick,
    Fixture,
    AccaTicket,
    AccaLeg,
    TipsterFollow,
)
from ..services.tipster_perf import compute_tipster_rolling_stats, model_edge_for_pick
from ..auth_firebase import get_current_user
from ..services.firebase import get_current_user as get_user_from_header

router = APIRouter(prefix="/api/tipsters", tags=["tipsters"])

# ---------- Schemas ----------

class TipsterIn(BaseModel):
  name: str
  username: str
  bio: str | None = None
  avatar_url: str | None = None
  sport_focus: str | None = None
  social_links: dict | None = None


class TipsterOut(BaseModel):
  id: int
  name: str
  username: str
  bio: str | None
  avatar_url: str | None
  sport_focus: str | None
  roi_30d: float
  winrate_30d: float
  profit_30d: float
  picks_30d: int
  social_links: dict | None = None   # public socials (no email)
  follower_count: int = 0
  is_following: bool = False
  is_owner: bool = False


class PickIn(BaseModel):
  fixture_id: int
  market: str
  bookmaker: str | None = None
  price: float
  stake: float = 1.0


class PickOut(BaseModel):
  id: int
  fixture_id: int
  market: str
  bookmaker: str | None
  price: float
  stake: float
  created_at: datetime
  result: str | None
  profit: float
  model_edge: float | None = None
  # fixture context for UI
  fixture_label: str | None = None
  fixture_path: str | None = None
  sport: str | None = None
  home_name: str | None = None
  away_name: str | None = None
  # deletion helpers
  kickoff_utc: str | None = None
  can_delete: bool = False


# ----- ACCA schemas (tipster-created tickets) -----

class AccaLegIn(BaseModel):
  fixture_id: int
  market: str
  bookmaker: Optional[str] = None
  price: float
  note: Optional[str] = None


class AccaIn(BaseModel):
  day: date = Field(default_factory=lambda: date.today())
  sport: str = "football"
  title: Optional[str] = None
  note: Optional[str] = None
  stake_units: float = 1.0
  is_public: bool = False
  legs: List[AccaLegIn]


class AccaLegOut(BaseModel):
  id: int
  fixture_id: int | None
  home_name: Optional[str] = None
  away_name: Optional[str] = None
  market: str
  bookmaker: Optional[str] = None
  price: float
  note: Optional[str] = None
  result: Optional[str] = None


class AccaOut(BaseModel):
  id: int
  source: str
  tipster_username: Optional[str] = None
  day: date
  sport: str
  title: Optional[str]
  note: Optional[str]
  stake_units: float
  is_public: bool
  combined_price: Optional[float] = None
  est_edge: Optional[float] = None
  result: Optional[str] = None
  profit: Optional[float] = None
  settled_at: Optional[datetime] = None
  created_at: datetime
  # deletion helpers
  earliest_kickoff_utc: Optional[str] = None
  can_delete: bool = False
  legs: List[AccaLegOut]


# ---------- helpers ----------

def _public_social_links(c: Tipster) -> dict | None:
  """
  Return social_links without private fields like email.
  Also normalises 'x' -> 'twitter' for the frontend.
  """
  try:
    sl = dict(c.social_links or {})
    sl.pop("email", None)
    # normalise X/Twitter
    if "x" in sl and "twitter" not in sl:
      sl["twitter"] = sl["x"]
    return sl or None
  except Exception:
    return None


def _follower_count(db: Session, tipster_id: int) -> int:
  return (
    db.query(TipsterFollow)
    .filter(TipsterFollow.tipster_id == tipster_id)
    .count()
  )


def _is_user_following(db: Session, tipster_id: int, email: str) -> bool:
  if not email:
    return False
  return (
    db.query(TipsterFollow)
    .filter(
      TipsterFollow.tipster_id == tipster_id,
      TipsterFollow.user_email == email.lower(),
    )
    .first()
    is not None
  )


def _to_tipster_out(c: Tipster) -> dict:
  return {
    "id": c.id,
    "name": c.name,
    "username": c.username,
    "bio": c.bio,
    "avatar_url": c.avatar_url,
    "sport_focus": c.sport_focus,
    "roi_30d": c.roi_30d or 0.0,
    "winrate_30d": c.winrate_30d or 0.0,
    "profit_30d": c.profit_30d or 0.0,
    "picks_30d": c.picks_30d or 0,
    "social_links": _public_social_links(c),
  }


def _email_of_tipster(c: Tipster) -> str | None:
  try:
    return (c.social_links or {}).get("email")
  except Exception:
    return None


def _require_owner(username: str, db: Session, user_claims: dict) -> Tipster:
  c = db.query(Tipster).filter(Tipster.username == username).first()
  if not c:
    raise HTTPException(404, "tipster not found")
  email = (user_claims.get("email") or "").lower()
  if (_email_of_tipster(c) or "").lower() != email:
    raise HTTPException(403, "not your profile")
  return c


def _fixture_info(db: Session, fixture_id: int) -> dict:
  f: Fixture | None = db.query(Fixture).get(fixture_id)
  if not f:
    return {
      "fixture_label": None,
      "fixture_path": None,
      "sport": None,
      "home_name": None,
      "away_name": None,
    }

  home = f.home_team or ""
  away = f.away_team or ""
  label = f"{home} vs {away}" if (home or away) else "Fixture"
  sport = (f.sport or "").lower()

  path = None
  if sport in ("football", "soccer"):
    path = f"/fixture/{f.id}"
  elif sport == "nhl":
    path = f"/nhl/game/{f.id}"
  elif sport in ("cfb", "ncaaf"):
    path = f"/cfb/fixture/{f.id}"
  elif sport in ("nba", "basketball"):
    path = f"/basketball/game/{f.id}"
  elif sport in ("nfl", "american_football"):
    path = None  # no dedicated page yet

  return {
    "fixture_label": label,
    "fixture_path": path,
    "sport": f.sport,
    "home_name": home or None,
    "away_name": away or None,
  }


def _pick_kickoff_iso(db: Session, fixture_id: int) -> str | None:
  fx = db.query(Fixture).get(fixture_id)
  if not fx or not fx.kickoff_utc:
    return None
  dt = fx.kickoff_utc
  if dt.tzinfo is None:
    dt = dt.replace(tzinfo=_tz.utc)
  return dt.isoformat()


def _pick_can_delete(db: Session, p: TipsterPick) -> bool:
  # no deletes after settlement or after kickoff
  if p.result:
    return False
  fx = db.query(Fixture).get(p.fixture_id)
  if not fx or not fx.kickoff_utc:
    return True  # if no KO recorded, allow (change to False if you prefer)
  now = datetime.now(_tz.utc)
  ko = fx.kickoff_utc if fx.kickoff_utc.tzinfo else fx.kickoff_utc.replace(tzinfo=_tz.utc)
  return now < ko


def _acca_can_delete(db: Session, t: AccaTicket) -> bool:
  if t.result:
    return False
  now = datetime.now(_tz.utc)
  for lg in t.legs or []:
    if not lg.fixture_id:
      continue
    fx = db.query(Fixture).get(lg.fixture_id)
    if not fx or not fx.kickoff_utc:
      continue
    ko = fx.kickoff_utc if fx.kickoff_utc.tzinfo else fx.kickoff_utc.replace(tzinfo=_tz.utc)
    if now >= ko:
      return False
  return True


def _acca_earliest_ko_iso(db: Session, t: AccaTicket) -> Optional[str]:
  kos: list[str] = []
  for lg in t.legs or []:
    if not lg.fixture_id:
      continue
    fx = db.query(Fixture).get(lg.fixture_id)
    if fx and fx.kickoff_utc:
      dt = fx.kickoff_utc
      if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_tz.utc)
      kos.append(dt.isoformat())
  return sorted(kos)[0] if kos else None


def _settle_profit(result: str, stake: float, price: float) -> float:
  if result == "WIN":
    return stake * (price - 1.0)
  if result == "LOSE":
    return -stake
  return 0.0  # PUSH/VOID/None


def _with_live_metrics(db: Session, c: Tipster) -> dict:
  out = _to_tipster_out(c)
  live = compute_tipster_rolling_stats(db, c.id, days=30)
  # live = {"picks","profit","roi","winrate"}
  out["roi_30d"] = float(live.get("roi") or 0.0)
  out["winrate_30d"] = float(live.get("winrate") or 0.0)
  out["profit_30d"] = float(live.get("profit") or 0.0)
  out["picks_30d"] = int(live.get("picks") or 0)
  return out


# ---------- routes: tipster profile ----------

@router.post("", response_model=TipsterOut)
def create_tipster(
  payload: TipsterIn,
  db: Session = Depends(get_db),
  user=Depends(get_current_user),
):
  email = (user.get("email") or "").lower()
  if not email:
    raise HTTPException(400, "Email missing in Firebase token")

  existing = db.query(Tipster).filter(Tipster.username == payload.username).first()
  if existing:
    if (_email_of_tipster(existing) or "").lower() == email:
      existing.name = payload.name
      existing.bio = payload.bio
      existing.avatar_url = payload.avatar_url
      existing.sport_focus = payload.sport_focus
      existing.social_links = (payload.social_links or {}) | {"email": email}
      db.commit()
      db.refresh(existing)
      out = _with_live_metrics(db, existing)
      out["is_owner"] = True
      # follower info (owner never "follows" themselves)
      out["follower_count"] = _follower_count(db, existing.id)
      out["is_following"] = False
      return out
    raise HTTPException(400, "username already exists")

  c = Tipster(**payload.model_dump())
  c.social_links = (payload.social_links or {}) | {"email": email}
  db.add(c)
  db.commit()
  db.refresh(c)
  out = _with_live_metrics(db, c)
  out["is_owner"] = True
  out["follower_count"] = 0
  out["is_following"] = False
  return out


@router.get("/me", response_model=TipsterOut | None)
def get_my_tipster(
  db: Session = Depends(get_db),
  user=Depends(get_current_user),
):
  email = (user.get("email") or "").lower()
  if not email:
    return None

  rows = db.query(Tipster).all()
  for c in rows:
    tip_email = ((_email_of_tipster(c) or "")).lower()
    if tip_email == email:
      out = _with_live_metrics(db, c)
      out["is_owner"] = True
      out["follower_count"] = _follower_count(db, c.id)
      out["is_following"] = False
      return out
  return None


@router.get("", response_model=list[TipsterOut])
def list_tipsters(db: Session = Depends(get_db)):
  rows = db.query(Tipster).all()
  enriched = []
  for c in rows:
    base = _with_live_metrics(db, c)
    base["follower_count"] = _follower_count(db, c.id)
    base["is_following"] = False  # leaderboard doesn't know viewer; keep false
    base["is_owner"] = False
    enriched.append(base)

  enriched.sort(key=lambda x: x["profit_30d"], reverse=True)
  return enriched


@router.get("/{username}", response_model=TipsterOut)
def get_tipster(username: str, request: Request, db: Session = Depends(get_db)):
  c = db.query(Tipster).filter(Tipster.username == username).first()
  if not c:
    raise HTTPException(404, "tipster not found")

  viewer = get_user_from_header(request.headers.get("Authorization"))
  viewer_email = (viewer or {}).get("email", "").lower()
  tipster_email = ((_email_of_tipster(c) or "")).lower()

  out = _with_live_metrics(db, c)
  out["is_owner"] = bool(viewer_email and viewer_email == tipster_email)
  out["follower_count"] = _follower_count(db, c.id)
  out["is_following"] = _is_user_following(db, c.id, viewer_email)
  return out


# ---------- routes: follow / unfollow ----------

@router.post("/{username}/follow")
def follow_tipster(
  username: str,
  db: Session = Depends(get_db),
  user=Depends(get_current_user),
):
  email = (user.get("email") or "").lower()
  if not email:
    raise HTTPException(400, "missing email")

  c = db.query(Tipster).filter(Tipster.username == username).first()
  if not c:
    raise HTTPException(404, "tipster not found")

  # prevent following yourself
  tip_email = (_email_of_tipster(c) or "").lower()
  if email == tip_email:
    raise HTTPException(400, "cannot follow yourself")

  exists = (
    db.query(TipsterFollow)
    .filter(
      TipsterFollow.tipster_id == c.id,
      TipsterFollow.user_email == email,
    )
    .first()
  )
  if exists:
    return {"ok": True, "already": True}

  db.add(TipsterFollow(tipster_id=c.id, user_email=email))
  db.commit()
  return {"ok": True, "followed": username}


@router.post("/{username}/unfollow")
def unfollow_tipster(
  username: str,
  db: Session = Depends(get_db),
  user=Depends(get_current_user),
):
  email = (user.get("email") or "").lower()
  if not email:
    raise HTTPException(400, "missing email")

  c = db.query(Tipster).filter(Tipster.username == username).first()
  if not c:
    raise HTTPException(404, "tipster not found")

  row = (
    db.query(TipsterFollow)
    .filter(
      TipsterFollow.tipster_id == c.id,
      TipsterFollow.user_email == email,
    )
    .first()
  )
  if not row:
    return {"ok": True, "was_following": False}

  db.delete(row)
  db.commit()
  return {"ok": True, "unfollowed": username}


# ---------- routes: picks ----------

class SettleIn(BaseModel):
  result: str  # WIN | LOSE | PUSH | VOID


@router.post("/{username}/picks", response_model=PickOut)
def create_pick(
  username: str,
  payload: PickIn,
  db: Session = Depends(get_db),
  user=Depends(get_current_user),
):
  c = _require_owner(username, db, user)
  p = TipsterPick(tipster_id=c.id, **payload.model_dump())
  db.add(p)
  db.commit()
  db.refresh(p)

  extra = _fixture_info(db, p.fixture_id)
  return {
    "id": p.id,
    "fixture_id": p.fixture_id,
    "market": p.market,
    "bookmaker": p.bookmaker,
    "price": p.price,
    "stake": p.stake,
    "created_at": p.created_at,
    "result": p.result,
    "profit": p.profit,
    "model_edge": model_edge_for_pick(db, p.fixture_id, p.market, p.price),
    "kickoff_utc": _pick_kickoff_iso(db, p.fixture_id),
    "can_delete": _pick_can_delete(db, p),
    **extra,
  }


@router.get("/{username}/picks", response_model=list[PickOut])
def list_picks(username: str, db: Session = Depends(get_db)):
  c = db.query(Tipster).filter(Tipster.username == username).first()
  if not c:
    raise HTTPException(404, "tipster not found")

  rows = (
    db.query(TipsterPick)
    .filter(TipsterPick.tipster_id == c.id)
    .order_by(TipsterPick.created_at.desc())
    .all()
  )

  out: list[dict] = []
  for p in rows:
    extra = _fixture_info(db, p.fixture_id)
    out.append(
      {
        "id": p.id,
        "fixture_id": p.fixture_id,
        "market": p.market,
        "bookmaker": p.bookmaker,
        "price": p.price,
        "stake": p.stake,
        "created_at": p.created_at,
        "result": p.result,
        "profit": p.profit,
        "model_edge": model_edge_for_pick(
          db, p.fixture_id, p.market, p.price
        ),
        "kickoff_utc": _pick_kickoff_iso(db, p.fixture_id),
        "can_delete": _pick_can_delete(db, p),
        **extra,
      }
    )
  return out


@router.post("/picks/{pick_id}/settle", response_model=PickOut)
def settle_pick(
  pick_id: int,
  body: SettleIn,
  db: Session = Depends(get_db),
  user=Depends(get_current_user),
):
  result = (body.result or "").upper()
  if result not in ("WIN", "LOSE", "PUSH", "VOID"):
    raise HTTPException(400, "result must be WIN, LOSE, PUSH or VOID")

  p = db.query(TipsterPick).get(pick_id)
  if not p:
    raise HTTPException(404, "pick not found")

  tip = db.query(Tipster).get(p.tipster_id)
  if not tip:
    raise HTTPException(404, "tipster not found for pick")
  email_claim = (user.get("email") or "").lower()
  tip_email = ((tip.social_links or {}).get("email") or "").lower()
  if not email_claim or email_claim != tip_email:
    raise HTTPException(403, "not your pick")

  p.result = result
  if result == "WIN":
    p.profit = _settle_profit("WIN", p.stake or 0.0, p.price or 0.0)
  elif result == "LOSE":
    p.profit = _settle_profit("LOSE", p.stake or 0.0, p.price or 0.0)
  else:  # PUSH or VOID
    p.profit = 0.0
  db.commit()
  db.refresh(p)

  extra = _fixture_info(db, p.fixture_id)
  return {
    "id": p.id,
    "fixture_id": p.fixture_id,
    "market": p.market,
    "bookmaker": p.bookmaker,
    "price": p.price,
    "stake": p.stake,
    "created_at": p.created_at,
    "result": p.result,
    "profit": p.profit,
    "model_edge": model_edge_for_pick(db, p.fixture_id, p.market, p.price),
    "kickoff_utc": _pick_kickoff_iso(db, p.fixture_id),
    "can_delete": _pick_can_delete(db, p),
    **extra,
  }


@router.delete("/picks/{pick_id}")
def delete_pick(
  pick_id: int,
  db: Session = Depends(get_db),
  user=Depends(get_current_user),
):
  p = db.query(TipsterPick).get(pick_id)
  if not p:
    raise HTTPException(404, "pick not found")

  tip = db.query(Tipster).get(p.tipster_id)
  if not tip:
    raise HTTPException(404, "tipster not found for pick")

  email_claim = (user.get("email") or "").lower()
  tip_email = ((tip.social_links or {}).get("email") or "").lower()
  if email_claim != tip_email:
    raise HTTPException(403, "not your pick")

  if not _pick_can_delete(db, p):
    raise HTTPException(
      403, "cannot delete after kickoff or once settled"
    )

  db.delete(p)
  db.commit()
  return {"ok": True, "deleted": pick_id}


# ---------- routes: ACCAs (tipster-created) ----------

class AccaSettleIn(BaseModel):
  result: str  # "WON" | "LOST" | "VOID"
  profit: Optional[float] = None  # manual override


def _to_acca_out(db: Session, t: AccaTicket) -> dict:
  tip_user = None
  if t.tipster_id:
    tip = db.query(Tipster).get(t.tipster_id)
    tip_user = tip.username if tip else None

  legs: list[dict] = []
  for lg in t.legs:
    legs.append(
      {
        "id": lg.id,
        "fixture_id": lg.fixture_id,
        "home_name": lg.home_name,
        "away_name": lg.away_name,
        "market": lg.market,
        "bookmaker": lg.bookmaker,
        "price": lg.price,
        "note": lg.note,
        "result": lg.result,
      }
    )

  return {
    "id": t.id,
    "source": t.source,
    "tipster_username": tip_user,
    "day": t.day,
    "sport": t.sport,
    "title": t.title,
    "note": t.note,
    "stake_units": t.stake_units,
    "is_public": t.is_public,
    "combined_price": t.combined_price,
    "est_edge": t.est_edge,
    "result": t.result,
    "profit": t.profit,
    "settled_at": t.settled_at,
    "created_at": t.created_at,
    "earliest_kickoff_utc": _acca_earliest_ko_iso(db, t),
    "can_delete": _acca_can_delete(db, t),
    "legs": legs,
  }


@router.post("/{username}/accas", response_model=AccaOut)
def create_tipster_acca(
  username: str,
  payload: AccaIn,
  db: Session = Depends(get_db),
  user=Depends(get_current_user),
):
  tip = _require_owner(username, db, user)

  if not payload.legs or len(payload.legs) < 2:
    raise HTTPException(400, "Acca needs at least 2 legs")

  t = AccaTicket(
    source="tipster",
    tipster_id=tip.id,
    day=payload.day,
    sport=payload.sport,
    title=payload.title,
    note=payload.note,
    stake_units=payload.stake_units,
    is_public=payload.is_public,
  )
  db.add(t)
  db.flush()  # get t.id

  combined = 1.0
  for leg in payload.legs:
    fx = db.query(Fixture).get(leg.fixture_id)
    home_name = fx.home_team if fx else None
    away_name = fx.away_team if fx else None

    price = float(leg.price or 0.0)
    combined *= price if price > 0 else 1.0

    db.add(
      AccaLeg(
        ticket_id=t.id,
        fixture_id=leg.fixture_id,
        home_name=home_name,
        away_name=away_name,
        market=leg.market,
        bookmaker=leg.bookmaker,
        price=price,
        note=leg.note,
      )
    )

  t.combined_price = round(combined, 4)
  db.commit()
  db.refresh(t)
  return _to_acca_out(db, t)


@router.get("/{username}/accas", response_model=list[AccaOut])
def list_tipster_accas(username: str, db: Session = Depends(get_db)):
  tip = db.query(Tipster).filter(Tipster.username == username).first()
  if not tip:
    raise HTTPException(404, "tipster not found")

  tickets = (
    db.query(AccaTicket)
    .filter(AccaTicket.source == "tipster", AccaTicket.tipster_id == tip.id)
    .order_by(AccaTicket.created_at.desc())
    .all()
  )
  return [_to_acca_out(db, t) for t in tickets]


@router.post("/accas/{ticket_id}/settle", response_model=AccaOut)
def settle_tipster_acca(
  ticket_id: int,
  body: AccaSettleIn,
  db: Session = Depends(get_db),
  user=Depends(get_current_user),
):
  t = db.query(AccaTicket).get(ticket_id)
  if not t:
    raise HTTPException(404, "acca not found")
  if t.source != "tipster":
    raise HTTPException(400, "only tipster accas can be settled here")

  tip = db.query(Tipster).get(t.tipster_id) if t.tipster_id else None
  if not tip:
    raise HTTPException(404, "tipster not found for acca")

  email_claim = (user.get("email") or "").lower()
  tip_email = ((tip.social_links or {}).get("email") or "").lower()
  if not email_claim or email_claim != tip_email:
    raise HTTPException(403, "not your acca")

  result = (body.result or "").upper()
  if result not in ("WON", "LOST", "VOID"):
    raise HTTPException(400, "result must be WON, LOST, or VOID")

  t.result = result
  if body.profit is not None:
    t.profit = body.profit
  else:
    if result == "WON" and t.combined_price:
      t.profit = (t.stake_units or 1.0) * (float(t.combined_price) - 1.0)
    elif result == "LOST":
      t.profit = -(t.stake_units or 1.0)
    else:
      t.profit = 0.0
  t.settled_at = datetime.utcnow()

  db.commit()
  db.refresh(t)
  return _to_acca_out(db, t)


@router.delete("/accas/{ticket_id}")
def delete_tipster_acca(
  ticket_id: int,
  db: Session = Depends(get_db),
  user=Depends(get_current_user),
):
  t = db.query(AccaTicket).get(ticket_id)
  if not t:
    raise HTTPException(404, "acca not found")
  if t.source != "tipster":
    raise HTTPException(400, "only tipster accas can be deleted here")

  tip = db.query(Tipster).get(t.tipster_id) if t.tipster_id else None
  if not tip:
    raise HTTPException(404, "tipster not found for acca")

  email_claim = (user.get("email") or "").lower()
  tip_email = ((tip.social_links or {}).get("email") or "").lower()
  if email_claim != tip_email:
    raise HTTPException(403, "not your acca")

  if not _acca_can_delete(db, t):
    raise HTTPException(
      403, "cannot delete after any leg has kicked off or once settled"
    )

  db.delete(t)
  db.commit()
  return {"ok": True, "deleted": ticket_id}