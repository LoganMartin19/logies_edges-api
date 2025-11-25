# api/app/routers/tipster_subscriptions.py
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import User, Tipster, TipsterSubscription
from ..auth_firebase import get_current_user

router = APIRouter(prefix="/api/tipsters", tags=["tipster-subscriptions"])


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


@router.post("/{username}/subscription/start", response_model=SubscriptionStatusOut)
def start_subscription(
    username: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    MVP: manual start of a tipster subscription.
    Later you can call this from Stripe webhook / billing.py instead.
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