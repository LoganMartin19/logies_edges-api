# api/app/routers/tipster_subscriptions.py
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
import os

from ..db import get_db
from ..models import User, Tipster, TipsterSubscription
from ..auth_firebase import get_current_user
from ..services import stripe_client  # 👈 use shared Stripe helpers

import stripe  # ⬅️ Stripe SDK


router = APIRouter(prefix="/api/tipsters", tags=["tipster-subscriptions"])

# --- Stripe config ----------------------------------------------------------

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
FRONTEND_BASE_URL = os.getenv(
    "FRONTEND_BASE_URL", "https://charteredsportsbetting.com"
)

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY


class SubscriptionStatusOut(BaseModel):
    is_subscriber: bool
    status: Optional[str] = None        # active|canceled|past_due|None
    plan_name: Optional[str] = None
    price_cents: Optional[int] = None
    renews_at: Optional[datetime] = None
    canceled_at: Optional[datetime] = None
    # handy for UI
    tipster_username: str
    tipster_name: str
    subscriber_count: int
    subscriber_limit: Optional[int] = None
    is_open_for_new_subs: bool


class CheckoutSessionOut(BaseModel):
    checkout_url: str


# ---------- helpers: users / tipsters / counts ----------


def _get_user(db: Session, claims: dict) -> User:
    uid = claims.get("uid")
    email = (claims.get("email") or "").lower()

    if not uid:
        raise HTTPException(400, "Firebase uid missing in token")

    user = db.query(User).filter(User.firebase_uid == uid).first()
    if user:
        # sync email if changed
        if email and user.email != email:
            user.email = email
            db.commit()
            db.refresh(user)
        return user

    # create on first login
    user = User(firebase_uid=uid, email=email)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _get_tipster_by_username(db: Session, username: str) -> Tipster:
    tip = db.query(Tipster).filter(Tipster.username == username).first()
    if not tip:
        raise HTTPException(404, "tipster not found")
    return tip


def _count_active_subs(db: Session, tipster_id: int) -> int:
    return (
        db.query(TipsterSubscription)
        .filter(
            TipsterSubscription.tipster_id == tipster_id,
            TipsterSubscription.status == "active",
        )
        .count()
    )


# ---------- helpers: Stripe customer + price ----------


def _ensure_stripe_customer(db: Session, user: User) -> str:
    """
    Ensure the User has a stripe_customer_id, creating one if needed.
    Reuses the same customer for Premium + tipster subs.
    """
    if getattr(user, "stripe_customer_id", None):
        return user.stripe_customer_id

    if not STRIPE_SECRET_KEY:
        raise HTTPException(500, "Stripe not configured on server")

    customer = stripe.Customer.create(
        email=user.email or None,
        metadata={"user_id": str(user.id)},
    )
    user.stripe_customer_id = customer["id"]
    db.commit()
    db.refresh(user)
    return user.stripe_customer_id


def _ensure_tipster_price(db: Session, tip: Tipster) -> str:
    """
    Ensure the Tipster has a Stripe Price for monthly subs.
    Auto-generates / auto-syncs it from tip.default_price_cents.

    Behaviour:
      - No stripe_price_id -> create new Price and store it.
      - Existing stripe_price_id:
          * If amount/currency/interval match -> reuse.
          * If they differ (e.g. tipster changed default_price_cents) -> create new
            Price and update tip.stripe_price_id.
      - If retrieve fails (deleted / invalid) -> create new Price.
    """
    if not STRIPE_SECRET_KEY:
        raise HTTPException(500, "Stripe not configured on server")

    if not tip.default_price_cents or tip.default_price_cents <= 0:
        raise HTTPException(400, "Tipster has no price set")

    desired_amount = int(tip.default_price_cents)
    desired_currency = (tip.currency or "gbp").lower()
    desired_interval = "month"

    def _create_price() -> stripe.Price:
        return stripe.Price.create(
            unit_amount=desired_amount,
            currency=desired_currency,
            recurring={"interval": desired_interval},
            product_data={
                "name": f"{tip.name} (@{tip.username}) tipster subscription",
            },
        )

    # 1) If we already have a stripe_price_id, check that it still matches
    if getattr(tip, "stripe_price_id", None):
        try:
            price = stripe.Price.retrieve(tip.stripe_price_id)
            # Only reuse if amount / currency / interval all match
            curr_interval = (price.get("recurring") or {}).get("interval")
            if (
                int(price["unit_amount"]) == desired_amount
                and price["currency"] == desired_currency
                and curr_interval == desired_interval
            ):
                return price["id"]
            # Otherwise fall through to create a fresh price below
        except stripe.error.StripeError:
            # treat as missing / invalid -> create new
            pass

    # 2) Create a new Price and store it on the tipster
    new_price = _create_price()
    tip.stripe_price_id = new_price["id"]
    db.commit()
    db.refresh(tip)
    return tip.stripe_price_id


# =====================================================================
#   STATUS
# =====================================================================

@router.get("/{username}/subscription", response_model=SubscriptionStatusOut)
def get_subscription_status(
    username: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    tip = _get_tipster_by_username(db, username)
    viewer = _get_user(db, user)

    sub = (
        db.query(TipsterSubscription)
        .filter(
            TipsterSubscription.user_id == viewer.id,
            TipsterSubscription.tipster_id == tip.id,
        )
        .first()
    )

    active_count = _count_active_subs(db, tip.id)
    is_open = bool(tip.is_open_for_new_subs) and (
        tip.subscriber_limit is None or active_count < tip.subscriber_limit
    )

    return SubscriptionStatusOut(
        is_subscriber=bool(sub and sub.status == "active"),
        status=sub.status if sub else None,
        plan_name=sub.plan_name if sub else None,
        price_cents=sub.price_cents if sub else tip.default_price_cents,
        renews_at=sub.renews_at if sub else None,
        canceled_at=sub.canceled_at if sub else None,
        tipster_username=tip.username,
        tipster_name=tip.name,
        subscriber_count=active_count,
        subscriber_limit=tip.subscriber_limit,
        is_open_for_new_subs=is_open,
    )


# =====================================================================
#   STRIPE CHECKOUT (preferred path)
# =====================================================================

@router.post(
    "/{username}/subscription/checkout",
    response_model=CheckoutSessionOut,
)
def create_stripe_checkout_session(
    username: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    tip = _get_tipster_by_username(db, username)
    viewer = _get_user(db, user)

    if tip.owner_user_id and tip.owner_user_id == viewer.id:
        raise HTTPException(400, "You cannot subscribe to your own tipster account")

    active_count = _count_active_subs(db, tip.id)
    if tip.subscriber_limit is not None and active_count >= tip.subscriber_limit:
        raise HTTPException(400, "This tipster is full for now")

    if not bool(tip.is_open_for_new_subs):
        raise HTTPException(400, "This tipster is currently closed for new subscribers")

    if not tip.default_price_cents or tip.default_price_cents <= 0:
        raise HTTPException(400, "Tipster has no subscription price set yet")

    # ensure Stripe objects exist
    customer_id = _ensure_stripe_customer(db, viewer)
    price_id = _ensure_tipster_price(db, tip)

    success_url = f"{FRONTEND_BASE_URL}/tipsters/{tip.username}?sub=success"
    cancel_url = f"{FRONTEND_BASE_URL}/tipsters/{tip.username}?sub=cancelled"

    if not STRIPE_SECRET_KEY:
        raise HTTPException(500, "Stripe not configured on server")

    # 👉 this helper returns a plain string URL
    checkout_url = stripe_client.create_subscription_checkout_session(
        customer_id=customer_id,
        price_id=price_id,
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={
            "user_id": str(viewer.id),
            "tipster_id": str(tip.id),
            "tipster_username": tip.username,
        },
    )

    return CheckoutSessionOut(checkout_url=checkout_url)


# =====================================================================
#   MANUAL START / CANCEL (MVP + webhook helper)
# =====================================================================

@router.post("/{username}/subscription/start", response_model=SubscriptionStatusOut)
def start_subscription(
    username: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    MVP: manual start of a tipster subscription.

    You can:
      - keep calling this directly (no Stripe) for testing, OR
      - call it from your Stripe webhook handler once payment succeeds,
        by building a fake `user` claims dict that contains the viewer's
        Firebase uid/email, or by refactoring the core logic into a
        helper that takes (db, user_id, tipster_id).
    """
    tip = _get_tipster_by_username(db, username)
    viewer = _get_user(db, user)

    if tip.owner_user_id and tip.owner_user_id == viewer.id:
        raise HTTPException(400, "You cannot subscribe to your own tipster account")

    active_count = _count_active_subs(db, tip.id)
    if tip.subscriber_limit is not None and active_count >= tip.subscriber_limit:
        raise HTTPException(400, "This tipster is full for now")

    if not bool(tip.is_open_for_new_subs):
        raise HTTPException(400, "This tipster is currently closed for new subscribers")

    sub = (
        db.query(TipsterSubscription)
        .filter(
            TipsterSubscription.user_id == viewer.id,
            TipsterSubscription.tipster_id == tip.id,
        )
        .first()
    )

    now = datetime.utcnow()
    renews = now + timedelta(days=30)  # simple monthly period for now

    if sub:
        # revive / update existing sub
        sub.status = "active"
        sub.canceled_at = None
        sub.started_at = sub.started_at or now
        sub.renews_at = renews
        if tip.default_price_cents:
            sub.price_cents = tip.default_price_cents
        sub.provider = sub.provider or "manual"
    else:
        sub = TipsterSubscription(
            user_id=viewer.id,
            tipster_id=tip.id,
            plan_name="Monthly",
            price_cents=tip.default_price_cents,
            status="active",
            provider="manual",
            started_at=now,
            renews_at=renews,
        )
        db.add(sub)

    db.commit()
    db.refresh(sub)

    active_count = _count_active_subs(db, tip.id)

    return SubscriptionStatusOut(
        is_subscriber=True,
        status=sub.status,
        plan_name=sub.plan_name,
        price_cents=sub.price_cents,
        renews_at=sub.renews_at,
        canceled_at=sub.canceled_at,
        tipster_username=tip.username,
        tipster_name=tip.name,
        subscriber_count=active_count,
        subscriber_limit=tip.subscriber_limit,
        is_open_for_new_subs=bool(tip.is_open_for_new_subs),
    )


@router.post("/{username}/subscription/cancel", response_model=SubscriptionStatusOut)
def cancel_subscription(
    username: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    tip = _get_tipster_by_username(db, username)
    viewer = _get_user(db, user)

    sub = (
        db.query(TipsterSubscription)
        .filter(
            TipsterSubscription.user_id == viewer.id,
            TipsterSubscription.tipster_id == tip.id,
        )
        .first()
    )
    if not sub:
        raise HTTPException(404, "subscription not found")

    if sub.status != "active":
        raise HTTPException(400, "subscription already not active")

    # If this subscription is backed by Stripe, cancel it there too
    if sub.provider == "stripe" and sub.provider_sub_id:
        if not STRIPE_SECRET_KEY:
            raise HTTPException(500, "Stripe not configured on server")

        try:
            # Immediately cancel the Stripe subscription
            stripe.Subscription.delete(sub.provider_sub_id)
        except stripe.error.StripeError as e:
            # Bubble up a clean error so the UI can show a message
            raise HTTPException(
                status_code=500,
                detail=f"Failed to cancel Stripe subscription: {str(e)}",
            )

    # Local DB: mark as canceled and stop access immediately
    sub.status = "canceled"
    sub.canceled_at = datetime.utcnow()
    sub.renews_at = None  # MVP: access stops immediately
    db.commit()
    db.refresh(sub)

    active_count = _count_active_subs(db, tip.id)

    return SubscriptionStatusOut(
        is_subscriber=False,
        status=sub.status,
        plan_name=sub.plan_name,
        price_cents=sub.price_cents,
        renews_at=sub.renews_at,
        canceled_at=sub.canceled_at,
        tipster_username=tip.username,
        tipster_name=tip.name,
        subscriber_count=active_count,
        subscriber_limit=tip.subscriber_limit,
        is_open_for_new_subs=bool(tip.is_open_for_new_subs),
    )