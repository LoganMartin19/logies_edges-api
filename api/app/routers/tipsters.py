# api/app/routers/tipsters.py
from __future__ import annotations

from datetime import date, datetime, timezone as _tz
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException
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
    User,
    TipsterSubscription,  # ✅ for subscriber gating
    TipsterApplication,   # ✅ NEW: applications table
)
from ..services.tipster_perf import compute_tipster_rolling_stats, model_edge_for_pick
from ..auth_firebase import get_current_user, optional_user
from ..services import stripe_connect
from ..services.push import send_new_pick_push

router = APIRouter(prefix="/api/tipsters", tags=["tipsters"])

# ---------- Schemas ----------


class TipsterIn(BaseModel):
    name: str
    username: str
    bio: str | None = None
    avatar_url: str | None = None
    sport_focus: str | None = None
    social_links: dict | None = None

    # 💸 pricing & capacity (optional)
    default_price_cents: int | None = None  # e.g. 1500 = £15
    currency: str | None = None            # e.g. "GBP"
    subscriber_limit: int | None = None    # e.g. 50, or None = unlimited
    is_open_for_new_subs: bool | None = None


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
    social_links: dict | None = None  # public socials (no email)
    follower_count: int = 0
    is_following: bool = False
    is_owner: bool = False

    # ⭐ NEW: verification flag
    is_verified: bool = False

    # ⭐ NEW: subscription & pricing metadata
    is_subscribed: bool = False
    subscriber_count: int = 0
    default_price_cents: int | None = None
    currency: str | None = None
    subscriber_limit: int | None = None
    is_open_for_new_subs: bool = True


class TipsterPricingIn(BaseModel):
    """
    Owner-only pricing settings for a tipster.
    Stored in GBP for now; Stripe can later map onto these.
    """
    monthly_price_gbp: float | None = None
    subscriber_limit: int | None = None
    is_open_for_new_subs: bool = True


class PickIn(BaseModel):
    fixture_id: int
    market: str
    bookmaker: str | None = None
    price: float
    stake: float = 1.0
    # ⭐️ allow tipster to mark as premium-only (global site premium)
    is_premium_only: bool = False
    # ⭐️ NEW: allow tipster to mark as subscriber-only (per-tipster subs)
    is_subscriber_only: bool = False


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
    # gating flags
    is_premium_only: bool = False
    is_subscriber_only: bool = False
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

    # NEW
    is_premium_only: bool = False
    is_subscriber_only: bool = False

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

    # NEW: gating flags
    is_premium_only: bool = False
    is_subscriber_only: bool = False

    earliest_kickoff_utc: Optional[str] = None
    can_delete: bool = False
    legs: List[AccaLegOut]


# ---------- tipster applications (public) ----------


class TipsterApplicationIn(BaseModel):
    """
    Payload from TipsterApply.jsx when a user submits an application.
    Email / uid come from the Firebase token; no need in the body.
    """
    name: str
    username: str
    bio: str | None = None
    sport_focus: str = "Football"
    avatar_url: str | None = None
    social_links: dict | None = None  # {"twitter": "...", "instagram": "..."}


class TipsterApplicationOut(BaseModel):
    id: int
    name: str
    username: str
    bio: str | None
    sport_focus: str
    avatar_url: str | None
    social_links: dict | None
    status: str
    admin_note: str | None
    created_at: datetime
    reviewed_at: datetime | None

    class Config:
        from_attributes = True


class AdminDecisionIn(BaseModel):
    admin_note: str | None = None


# ---------- helpers ----------

def _get_or_create_user_by_claims(db: Session, claims: dict) -> User:
    uid = claims.get("uid")
    email = (claims.get("email") or "").lower()

    if not uid:
        raise HTTPException(400, "Firebase uid missing in token")

    user = db.query(User).filter(User.firebase_uid == uid).first()
    if user:
        # keep email in sync
        if email and user.email != email:
            user.email = email
            db.commit()
            db.refresh(user)
        return user

    # create new user row
    user = User(firebase_uid=uid, email=email)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _require_admin_user(db: Session, claims: dict) -> User:
    """
    Ensure the current Firebase user maps to a User row with is_admin = True.
    """
    claims = claims or {}
    uid = claims.get("uid")
    email = (claims.get("email") or "").lower()

    if not uid and not email:
        raise HTTPException(403, "Admin only")

    q = db.query(User)
    user_row: User | None = None
    if uid:
        user_row = q.filter(User.firebase_uid == uid).first()
    if not user_row and email:
        user_row = q.filter(User.email == email).first()
    if not user_row:
        # fall back to creating if somehow missing
        user_row = _get_or_create_user_by_claims(db, claims)

    if not getattr(user_row, "is_admin", False):
        raise HTTPException(403, "Admin only")

    return user_row


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

        # ⭐ NEW: verification flag
        "is_verified": bool(getattr(c, "is_verified", False)),

        # ⭐ NEW: basic pricing flags copied from model
        "default_price_cents": c.default_price_cents,
        "currency": c.currency,
        "subscriber_limit": c.subscriber_limit,
        "is_open_for_new_subs": bool(c.is_open_for_new_subs),
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


# ---- follower / subscriber helpers ----


def _follower_count(db: Session, tipster_id: int) -> int:
    return (
        db.query(TipsterFollow)
        .filter(TipsterFollow.tipster_id == tipster_id)
        .count()
    )


def _subscriber_count(db: Session, tipster_id: int) -> int:
    return (
        db.query(TipsterSubscription)
        .filter(
            TipsterSubscription.tipster_id == tipster_id,
            TipsterSubscription.status == "active",
        )
        .count()
    )


def _is_user_following(db: Session, tipster_id: int, email: str) -> bool:
    if not email:
        return False
    return (
        db.query(TipsterFollow)
        .filter(
            TipsterFollow.tipster_id == tipster_id,
            TipsterFollow.follower_email == email.lower(),
        )
        .first()
        is not None
    )


def _with_live_metrics(db: Session, c: Tipster) -> dict:
    out = _to_tipster_out(c)
    live = compute_tipster_rolling_stats(db, c.id, days=30)
    # live = {"picks","profit","roi","winrate"}
    out["roi_30d"] = float(live.get("roi") or 0.0)
    out["winrate_30d"] = float(live.get("winrate") or 0.0)
    out["profit_30d"] = float(live.get("profit") or 0.0)
    out["picks_30d"] = int(live.get("picks") or 0)
    # followers + subscribers (for leaderboard + detail)
    out["follower_count"] = _follower_count(db, c.id)
    out["subscriber_count"] = _subscriber_count(db, c.id)
    return out


def _viewer_premium_status(
    db: Session,
    viewer_claims: dict | None,
) -> tuple[bool, str, int | None]:
    """
    Helper: given Firebase claims (or None), return:
      (is_premium, email_lower, user_id_or_None)
    """
    viewer_claims = viewer_claims or {}
    email = (viewer_claims.get("email") or "").lower()
    uid = viewer_claims.get("uid")

    user_row: User | None = None
    if uid:
        user_row = db.query(User).filter(User.firebase_uid == uid).first()
    if not user_row and email:
        user_row = db.query(User).filter(User.email == email).first()

    is_premium = bool(user_row.is_premium) if user_row else False
    user_id = user_row.id if user_row else None
    return is_premium, email, user_id


def _viewer_is_subscribed_to_tipster(
    db: Session, user_id: int | None, tipster_id: int
) -> bool:
    """
    True if the viewer has an active subscription to this tipster.
    """
    if not user_id:
        return False
    row = (
        db.query(TipsterSubscription)
        .filter(
            TipsterSubscription.user_id == user_id,
            TipsterSubscription.tipster_id == tipster_id,
            TipsterSubscription.status == "active",
        )
        .first()
    )
    return row is not None


# ---------- routes: tipster applications (public + admin) ----------


@router.post("/apply", response_model=TipsterApplicationOut, status_code=201)
def apply_tipster(
    payload: TipsterApplicationIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Logged-in users submit a tipster application.
    Creates a TipsterApplication row with status='pending'.
    """
    uid = user.get("uid")
    email = (user.get("email") or "").lower()

    if not uid or not email:
        raise HTTPException(400, "Firebase uid/email missing in token")

    # ensure we have a User row – re-use your helper
    _get_or_create_user_by_claims(db, user)

    cleaned_username = (payload.username or "").strip().lower()
    if not cleaned_username:
        raise HTTPException(400, "username is required")

    # 1️⃣ prevent duplicate pending apps for same uid
    existing_pending = (
        db.query(TipsterApplication)
        .filter(
            TipsterApplication.firebase_uid == uid,
            TipsterApplication.status == "pending",
        )
        .first()
    )
    if existing_pending:
        raise HTTPException(
            400,
            "You already have a pending tipster application.",
        )

    # 2️⃣ username not already taken by an existing tipster
    user_clash = (
        db.query(Tipster)
        .filter(Tipster.username == cleaned_username)
        .first()
    )
    if user_clash:
        raise HTTPException(400, "That username is already taken by a tipster.")

    # 3️⃣ username not pending on another application
    app_clash = (
        db.query(TipsterApplication)
        .filter(
            TipsterApplication.username == cleaned_username,
            TipsterApplication.status == "pending",
        )
        .first()
    )
    if app_clash:
        raise HTTPException(
            400, "That username is already pending on another application."
        )

    app = TipsterApplication(
        firebase_uid=uid,
        email=email,
        name=(payload.name or "").strip(),
        username=cleaned_username,
        sport_focus=(payload.sport_focus or "Football").strip(),
        avatar_url=(payload.avatar_url or "").strip() or None,
        bio=(payload.bio or "").strip() or None,
        social_links=payload.social_links or {},
        status="pending",
    )
    db.add(app)
    db.commit()
    db.refresh(app)
    return app


@router.get("/applications/me", response_model=list[TipsterApplicationOut])
def my_tipster_applications(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Viewer's own applications (history).
    """
    uid = user.get("uid")
    if not uid:
        raise HTTPException(400, "Firebase uid missing in token")

    rows = (
        db.query(TipsterApplication)
        .filter(TipsterApplication.firebase_uid == uid)
        .order_by(TipsterApplication.created_at.desc())
        .all()
    )
    return rows


@router.get("/applications", response_model=list[TipsterApplicationOut])
def list_tipster_applications(
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Admin: list all tipster applications, optionally filtered by status.
    """
    _require_admin_user(db, user)

    q = db.query(TipsterApplication)
    if status in ("pending", "approved", "rejected"):
        q = q.filter(TipsterApplication.status == status)

    rows = q.order_by(TipsterApplication.created_at.desc()).all()
    return rows


@router.post("/applications/{app_id}/approve", response_model=TipsterApplicationOut)
def approve_tipster_application(
    app_id: int,
    body: AdminDecisionIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Admin: approve an application and create a verified Tipster profile.
    """
    _require_admin_user(db, user)

    app = db.query(TipsterApplication).get(app_id)
    if not app:
        raise HTTPException(404, "application not found")
    if app.status != "pending":
        raise HTTPException(400, "Only pending applications can be approved")

    cleaned_username = (app.username or "").strip().lower()

    # ensure no tipster already exists with this username
    username_clash = (
        db.query(Tipster).filter(Tipster.username == cleaned_username).first()
    )
    if username_clash:
        raise HTTPException(
            400,
            "A tipster already exists with this username – cannot auto-approve.",
        )

    # ensure we have owner User row
    owner_claims = {"uid": app.firebase_uid, "email": app.email}
    owner_user = _get_or_create_user_by_claims(db, owner_claims)

    # create the tipster profile
    tip = Tipster(
        name=app.name,
        username=cleaned_username,
        bio=app.bio,
        avatar_url=app.avatar_url,
        sport_focus=app.sport_focus,
        social_links=(app.social_links or {}) | {"email": app.email},
        owner_user_id=owner_user.id,
        currency="GBP",
        is_open_for_new_subs=True,
    )

    # mark as verified if column exists
    if hasattr(tip, "is_verified"):
        tip.is_verified = True  # type: ignore[attr-defined]

    db.add(tip)

    # update application status
    app.status = "approved"
    app.admin_note = body.admin_note
    app.reviewed_at = datetime.utcnow()

    db.commit()
    db.refresh(app)
    return app


@router.post("/applications/{app_id}/reject", response_model=TipsterApplicationOut)
def reject_tipster_application(
    app_id: int,
    body: AdminDecisionIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Admin: reject an application (no tipster created).
    """
    _require_admin_user(db, user)

    app = db.query(TipsterApplication).get(app_id)
    if not app:
        raise HTTPException(404, "application not found")
    if app.status != "pending":
        raise HTTPException(400, "Only pending applications can be rejected")

    app.status = "rejected"
    app.admin_note = body.admin_note
    app.reviewed_at = datetime.utcnow()

    db.commit()
    db.refresh(app)
    return app


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

    # make sure we have a User row
    owner_user = _get_or_create_user_by_claims(db, user)

    existing = db.query(Tipster).filter(Tipster.username == payload.username).first()
    if existing:
        # --- UPDATE EXISTING PROFILE (owner only) ---
        if (_email_of_tipster(existing) or "").lower() == email:
            existing.name = payload.name
            existing.bio = payload.bio
            existing.avatar_url = payload.avatar_url
            existing.sport_focus = payload.sport_focus
            existing.social_links = (payload.social_links or {}) | {"email": email}

            # 💸 allow owner to update pricing + capacity
            if payload.default_price_cents is not None:
                existing.default_price_cents = payload.default_price_cents
            if payload.currency is not None:
                existing.currency = payload.currency
            if payload.subscriber_limit is not None:
                existing.subscriber_limit = payload.subscriber_limit
            if payload.is_open_for_new_subs is not None:
                existing.is_open_for_new_subs = payload.is_open_for_new_subs

            # ensure owner_user_id is populated
            if not existing.owner_user_id:
                existing.owner_user_id = owner_user.id

            db.commit()
            db.refresh(existing)
            out = _with_live_metrics(db, existing)
            out["is_owner"] = True
            out["is_following"] = False
            return out
        raise HTTPException(400, "username already exists")

    # --- CREATE NEW TIPSTER ---
    c = Tipster(**payload.model_dump())
    c.social_links = (payload.social_links or {}) | {"email": email}
    c.owner_user_id = owner_user.id  # 👈 tie to User

    # sensible defaults if not provided
    if c.currency is None:
        c.currency = "GBP"
    if c.is_open_for_new_subs is None:
        c.is_open_for_new_subs = True

    db.add(c)
    db.commit()
    db.refresh(c)
    out = _with_live_metrics(db, c)
    out["is_owner"] = True
    out["is_following"] = False
    return out


@router.post("/{username}/pricing", response_model=TipsterOut)
def update_tipster_pricing(
    username: str,
    payload: TipsterPricingIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Owner-only endpoint to set subscription pricing + caps.
    Currently stores monthly price in GBP cents on the Tipster row.
    """
    tip = _require_owner(username, db, user)

    # price in GBP → cents
    if payload.monthly_price_gbp is not None:
        cents = int(round(payload.monthly_price_gbp * 100))
        tip.default_price_cents = cents
        tip.currency = "GBP"
    else:
        tip.default_price_cents = None

    tip.subscriber_limit = payload.subscriber_limit
    tip.is_open_for_new_subs = payload.is_open_for_new_subs

    db.commit()
    db.refresh(tip)

    out = _with_live_metrics(db, tip)
    email = (user.get("email") or "").lower()
    tip_email = (_email_of_tipster(tip) or "").lower()
    out["is_owner"] = email == tip_email
    out["is_following"] = False
    out["is_subscribed"] = False
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
            out["is_following"] = False  # you don't "follow" yourself
            # owner view: is_subscribed not really relevant; leave default False
            return out
    return None


@router.get("", response_model=list[TipsterOut])
def list_tipsters(db: Session = Depends(get_db)):
    rows = db.query(Tipster).all()
    enriched = [_with_live_metrics(db, c) for c in rows]
    enriched.sort(key=lambda x: x["profit_30d"], reverse=True)
    # leaderboard view – we don't care about is_following/is_owner/is_subscribed here
    return [
        {
            **r,
            "is_owner": False,
            "is_following": False,
            "is_subscribed": False,
        }
        for r in enriched
    ]


@router.get("/{username}", response_model=TipsterOut)
def get_tipster(
    username: str,
    db: Session = Depends(get_db),
    viewer=Depends(optional_user),
):
    """
    Public tipster profile:
      - works for guests (viewer = None)
      - if logged in, marks is_owner + is_following + is_subscribed
    """
    c = db.query(Tipster).filter(Tipster.username == username).first()
    if not c:
        raise HTTPException(404, "tipster not found")

    viewer_is_premium, viewer_email, viewer_user_id = _viewer_premium_status(db, viewer)
    tipster_email = ((_email_of_tipster(c) or "")).lower()

    out = _with_live_metrics(db, c)
    out["is_owner"] = bool(viewer_email and viewer_email == tipster_email)
    out["is_following"] = _is_user_following(db, c.id, viewer_email)
    out["is_subscribed"] = _viewer_is_subscribed_to_tipster(db, viewer_user_id, c.id)
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
        raise HTTPException(400, "Email missing in Firebase token")

    c = db.query(Tipster).filter(Tipster.username == username).first()
    if not c:
        raise HTTPException(404, "tipster not found")

    tip_email = ((_email_of_tipster(c) or "")).lower()
    if tip_email == email:
        raise HTTPException(400, "You cannot follow yourself")

    existing = (
        db.query(TipsterFollow)
        .filter(
            TipsterFollow.tipster_id == c.id,
            TipsterFollow.follower_email == email,
        )
        .first()
    )
    if existing:
        # already following; return current count
        return {
            "ok": True,
            "status": "already_following",
            "follower_count": _follower_count(db, c.id),
        }

    db.add(
        TipsterFollow(
            tipster_id=c.id,
            follower_email=email,
            created_at=datetime.utcnow(),
        )
    )
    db.commit()

    return {
        "ok": True,
        "status": "following",
        "follower_count": _follower_count(db, c.id),
    }


@router.delete("/{username}/follow")
def unfollow_tipster(
    username: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    email = (user.get("email") or "").lower()
    if not email:
        raise HTTPException(400, "Email missing in Firebase token")

    c = db.query(Tipster).filter(Tipster.username == username).first()
    if not c:
        raise HTTPException(404, "tipster not found")

    row = (
        db.query(TipsterFollow)
        .filter(
            TipsterFollow.tipster_id == c.id,
            TipsterFollow.follower_email == email,
        )
        .first()
    )
    if row:
        db.delete(row)
        db.commit()

    return {
        "ok": True,
        "status": "not_following",
        "follower_count": _follower_count(db, c.id),
    }


# ---------- routes: following feed & list ----------


@router.get("/following/list")
def list_following(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    email = (user.get("email") or "").lower()
    if not email:
        raise HTTPException(400, "Email missing in Firebase token")

    # all relationships
    rows = (
        db.query(TipsterFollow)
        .filter(TipsterFollow.follower_email == email)
        .all()
    )
    tipster_ids = [r.tipster_id for r in rows]

    if not tipster_ids:
        return []

    tipsters = (
        db.query(Tipster)
        .filter(Tipster.id.in_(tipster_ids))
        .all()
    )

    out: list[dict] = []
    for t in tipsters:
        row = _with_live_metrics(db, t)
        row["is_owner"] = False
        row["is_following"] = True
        # list view: is_subscribed not shown; default False
        out.append(row)

    return out


@router.get("/following/feed")
def following_feed(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    # viewer is logged in here
    viewer_is_premium, viewer_email, viewer_user_id = _viewer_premium_status(db, user)

    if not viewer_email:
        raise HTTPException(400, "Email missing in Firebase token")

    # find whom the user follows
    rows = (
        db.query(TipsterFollow)
        .filter(TipsterFollow.follower_email == viewer_email)
        .all()
    )
    tipster_ids = [r.tipster_id for r in rows]

    if not tipster_ids:
        return []

    # get latest picks from followed tipsters
    picks = (
        db.query(TipsterPick)
        .filter(TipsterPick.tipster_id.in_(tipster_ids))
        .order_by(TipsterPick.created_at.desc())
        .limit(200)
        .all()
    )

    out = []
    for p in picks:
        extra = _fixture_info(db, p.fixture_id)
        tipster = db.query(Tipster).get(p.tipster_id)

        viewer_is_subscriber = _viewer_is_subscribed_to_tipster(
            db, viewer_user_id, p.tipster_id
        )

        locked = (
            (p.is_premium_only and not viewer_is_premium)
            or (getattr(p, "is_subscriber_only", False) and not viewer_is_subscriber)
        )

        if locked:
            # 🔒 Non-premium / non-subscriber viewer → send locked stub row
            out.append(
                {
                    "id": p.id,
                    "tipster_username": tipster.username if tipster else None,
                    "tipster_name": tipster.name if tipster else None,
                    "created_at": p.created_at,
                    "market": None,
                    "fixture_id": p.fixture_id,
                    "bookmaker": None,
                    "price": None,
                    "stake": None,
                    "result": None,
                    "profit": 0.0,
                    "model_edge": None,
                    "is_premium_only": bool(p.is_premium_only),
                    "is_subscriber_only": bool(
                        getattr(p, "is_subscriber_only", False)
                    ),
                    **extra,
                }
            )
            continue

        # normal visible pick
        out.append(
            {
                "id": p.id,
                "tipster_username": tipster.username if tipster else None,
                "tipster_name": tipster.name if tipster else None,
                "created_at": p.created_at,
                "market": p.market,
                "fixture_id": p.fixture_id,
                "bookmaker": p.bookmaker,
                "price": p.price,
                "stake": p.stake,
                "result": p.result,
                "profit": p.profit,
                "model_edge": model_edge_for_pick(
                    db, p.fixture_id, p.market, p.price
                ),
                "is_premium_only": bool(p.is_premium_only),
                "is_subscriber_only": bool(
                    getattr(p, "is_subscriber_only", False)
                ),
                **extra,
            }
        )

    return out


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
    p = TipsterPick(
        tipster_id=c.id,
        fixture_id=payload.fixture_id,
        market=payload.market,
        bookmaker=payload.bookmaker,
        price=payload.price,
        stake=payload.stake,
        is_premium_only=payload.is_premium_only,
        is_subscriber_only=getattr(payload, "is_subscriber_only", False),
    )
    db.add(p)
    db.commit()
    db.refresh(p)

    # Fixture context
    extra = _fixture_info(db, p.fixture_id)
    fx = db.query(Fixture).get(p.fixture_id)

    # 🔔 Fire-and-forget push notifications (followers / subscribers)
    try:
        send_new_pick_push(db, c, p, fx)
    except Exception as e:
        # Don't break pick creation if push fails
        print("[push] error sending pick push:", e)

    # 🔒 sanitize required fields for PickOut
    market = p.market or "Unknown"
    price = float(p.price) if p.price is not None else 0.0
    stake = float(p.stake) if p.stake is not None else 0.0
    profit = float(p.profit) if p.profit is not None else 0.0

    return {
        "id": p.id,
        "fixture_id": p.fixture_id,
        "market": market,
        "bookmaker": p.bookmaker or None,
        "price": price,
        "stake": stake,
        "created_at": p.created_at,
        "result": p.result,
        "profit": profit,
        "model_edge": model_edge_for_pick(db, p.fixture_id, p.market, p.price),
        "kickoff_utc": _pick_kickoff_iso(db, p.fixture_id),
        "can_delete": _pick_can_delete(db, p),
        "is_premium_only": bool(p.is_premium_only),
        "is_subscriber_only": bool(getattr(p, "is_subscriber_only", False)),
        **extra,
    }


@router.get("/{username}/picks", response_model=list[PickOut])
def list_picks(
    username: str,
    db: Session = Depends(get_db),
    viewer=Depends(optional_user),
):
    tip = db.query(Tipster).filter(Tipster.username == username).first()
    if not tip:
        raise HTTPException(404, "tipster not found")

    viewer_is_premium, viewer_email, viewer_user_id = _viewer_premium_status(db, viewer)
    tip_email = ((_email_of_tipster(tip) or "")).lower()
    is_owner = bool(viewer_email and viewer_email == tip_email)

    viewer_is_subscriber = _viewer_is_subscribed_to_tipster(
        db, viewer_user_id, tip.id
    )

    rows = (
        db.query(TipsterPick)
        .filter(TipsterPick.tipster_id == tip.id)
        .order_by(TipsterPick.created_at.desc())
        .all()
    )

    out: list[dict] = []
    for p in rows:
        extra = _fixture_info(db, p.fixture_id)
        is_sub_only = bool(getattr(p, "is_subscriber_only", False))

        locked = (
            (p.is_premium_only and not (viewer_is_premium or is_owner))
            or (is_sub_only and not (viewer_is_subscriber or is_owner))
        )

        if locked:
            # 🔒 Non-premium, non-owner, non-subscriber → locked row
            out.append(
                {
                    "id": p.id,
                    "fixture_id": p.fixture_id,
                    "market": "Premium pick",
                    "bookmaker": None,
                    "price": 0.0,
                    "stake": 0.0,
                    "created_at": p.created_at,
                    "result": None,
                    "profit": 0.0,
                    "model_edge": None,
                    "kickoff_utc": _pick_kickoff_iso(db, p.fixture_id),
                    "can_delete": False,
                    "is_premium_only": bool(p.is_premium_only),
                    "is_subscriber_only": is_sub_only,
                    **extra,
                }
            )
            continue

        # normal visible pick, sanitized
        market = p.market or "Unknown"
        price = float(p.price) if p.price is not None else 0.0
        stake = float(p.stake) if p.stake is not None else 0.0
        profit = float(p.profit) if p.profit is not None else 0.0

        out.append(
            {
                "id": p.id,
                "fixture_id": p.fixture_id,
                "market": market,
                "bookmaker": p.bookmaker or None,
                "price": price,
                "stake": stake,
                "created_at": p.created_at,
                "result": p.result,
                "profit": profit,
                "model_edge": model_edge_for_pick(
                    db, p.fixture_id, p.market, p.price
                ),
                "kickoff_utc": _pick_kickoff_iso(db, p.fixture_id),
                "can_delete": _pick_can_delete(db, p),
                "is_premium_only": bool(p.is_premium_only),
                "is_subscriber_only": is_sub_only,
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

    # 🔒 sanitize again for PickOut
    market = p.market or "Unknown"
    price = float(p.price) if p.price is not None else 0.0
    stake = float(p.stake) if p.stake is not None else 0.0
    profit = float(p.profit) if p.profit is not None else 0.0

    return {
        "id": p.id,
        "fixture_id": p.fixture_id,
        "market": market,
        "bookmaker": p.bookmaker or None,
        "price": price,
        "stake": stake,
        "created_at": p.created_at,
        "result": p.result,
        "profit": profit,
        "model_edge": model_edge_for_pick(db, p.fixture_id, p.market, p.price),
        "kickoff_utc": _pick_kickoff_iso(db, p.fixture_id),
        "can_delete": _pick_can_delete(db, p),
        "is_premium_only": bool(p.is_premium_only),
        "is_subscriber_only": bool(getattr(p, "is_subscriber_only", False)),
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

        # NEW
        "is_premium_only": bool(getattr(t, "is_premium_only", False)),
        "is_subscriber_only": bool(getattr(t, "is_subscriber_only", False)),

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

        # NEW
        is_premium_only=payload.is_premium_only,
        is_subscriber_only=payload.is_subscriber_only,
    )
    db.add(t)
    db.flush()
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
def list_tipster_accas(
    username: str,
    db: Session = Depends(get_db),
    viewer=Depends(optional_user),
):
    tip = db.query(Tipster).filter(Tipster.username == username).first()
    if not tip:
        raise HTTPException(404, "tipster not found")

    viewer_is_premium, viewer_email, viewer_user_id = _viewer_premium_status(db, viewer)
    tip_email = ((_email_of_tipster(tip) or "")).lower()
    is_owner = bool(viewer_email and viewer_email == tip_email)

    viewer_is_subscriber = _viewer_is_subscribed_to_tipster(
        db, viewer_user_id, tip.id
    )

    tickets = (
        db.query(AccaTicket)
        .filter(AccaTicket.source == "tipster", AccaTicket.tipster_id == tip.id)
        .order_by(AccaTicket.created_at.desc())
        .all()
    )

    out: list[dict] = []
    for t in tickets:
        is_prem_only = bool(getattr(t, "is_premium_only", False))
        is_sub_only = bool(getattr(t, "is_subscriber_only", False))

        locked = (
            (is_prem_only and not (viewer_is_premium or is_owner)) or
            (is_sub_only and not (viewer_is_subscriber or is_owner))
        )

        if locked:
            base = _to_acca_out(db, t)
            # strip sensitive bits
            base["legs"] = []
            base["combined_price"] = None
            base["profit"] = None
            base["stake_units"] = 0.0
            out.append(base)
        else:
            out.append(_to_acca_out(db, t))

    return out


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


class ConnectStatusOut(BaseModel):
    has_connect: bool
    charges_enabled: bool = False
    payouts_enabled: bool = False
    details_submitted: bool = False
    currently_due: list[str] = Field(default_factory=list)


@router.post("/{username}/connect/onboard", response_model=dict)
def start_connect_onboarding(
    username: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Owner-only: create or reuse a Stripe Express account and return onboarding URL.
    """
    # Confirm they own this tipster (email check, as before)
    tip = _require_owner(username, db, user)

    # Ensure we have a User row that matches this Firebase user
    owner_user = _get_or_create_user_by_claims(db, user)

    # (Optional belt-and-braces check)
    if tip.owner_user_id and tip.owner_user_id != owner_user.id:
        raise HTTPException(403, "not your profile")

    account_id = stripe_connect.ensure_express_account(db, tip, owner_user)
    url = stripe_connect.create_onboarding_link(account_id)
    return {"onboarding_url": url}


@router.get("/{username}/connect/status", response_model=ConnectStatusOut)
def connect_status(
    username: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    tip = _require_owner(username, db, user)

    if not tip.stripe_account_id:
        return ConnectStatusOut(has_connect=False)

    status = stripe_connect.get_connect_status(tip.stripe_account_id)
    return ConnectStatusOut(
        has_connect=True,
        charges_enabled=status["charges_enabled"],
        payouts_enabled=status["payouts_enabled"],
        details_submitted=status["details_submitted"],
        currently_due=status["currently_due"],
    )


@router.get("/{username}/connect/dashboard", response_model=dict)
def connect_dashboard_link(
    username: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    tip = _require_owner(username, db, user)

    if not tip.stripe_account_id:
        raise HTTPException(400, "Creator has no Stripe Connect account yet")

    url = stripe_connect.create_login_link(tip.stripe_account_id)
    return {"dashboard_url": url}