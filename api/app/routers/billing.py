# api/app/routers/billing.py
from __future__ import annotations

from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from ..db import get_db
from ..auth_firebase import get_current_user
from ..models import User, Tipster, TipsterSubscription
from ..settings import settings
from ..services import stripe_client

router = APIRouter(prefix="/billing", tags=["billing"])


# ============================================================
# Helper: ensure DB user exists
# ============================================================

def _get_or_create_db_user(db: Session, fb_claims: dict) -> User:
    uid = fb_claims.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="Missing Firebase uid")

    user = db.query(User).filter(User.firebase_uid == uid).first()
    if user:
        return user

    user = User(
        firebase_uid=uid,
        email=fb_claims.get("email"),
        display_name=fb_claims.get("name"),
        avatar_url=fb_claims.get("picture"),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# ============================================================
# Premium Checkout
# ============================================================

@router.post("/premium/checkout")
def create_premium_checkout(
    request: Request,
    fb_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Start Stripe Checkout for CSB Premium.
    Returns: { checkout_url: "https://..." }
    """
    user = _get_or_create_db_user(db, fb_user)

    if not settings.STRIPE_PREMIUM_PRICE_ID or not settings.STRIPE_SECRET_KEY:
        raise HTTPException(500, "Stripe is not configured on server")

    if not user.email:
        raise HTTPException(400, "User account must have an email to start checkout.")

    origin = request.headers.get("origin") or (
        request.url.scheme + "://" + request.url.netloc
    )
    success_url = f"{origin}/account?upgraded=1"
    cancel_url = f"{origin}/account?canceled=1"

    checkout_url = stripe_client.create_premium_checkout_session(
        customer_email=user.email,
        firebase_uid=user.firebase_uid,
        success_url=success_url,
        cancel_url=cancel_url,
    )

    return {"checkout_url": checkout_url}


# ============================================================
# Billing Portal
# ============================================================

@router.post("/customer-portal")
def create_billing_portal_session(
    request: Request,
    fb_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = _get_or_create_db_user(db, fb_user)

    if not user.email:
        raise HTTPException(400, "User account must have an email to manage billing.")

    # Ensure we have a Stripe customer ID; create if missing
    customer_id = user.stripe_customer_id
    if not customer_id:
        customer = stripe_client.get_or_create_customer(user.email)
        user.stripe_customer_id = customer.id
        db.commit()

    origin = request.headers.get("origin") or (
        request.url.scheme + "://" + request.url.netloc
    )
    return_url = f"{origin}/account"

    # Try to create a portal session; if customer_id is stale, recreate & retry once
    try:
        portal_url = stripe_client.create_billing_portal_session(
            customer_id=user.stripe_customer_id,
            return_url=return_url,
        )
    except Exception:
        try:
            # stale / invalid customer -> recreate and retry
            customer = stripe_client.get_or_create_customer(user.email)
            user.stripe_customer_id = customer.id
            db.commit()
            portal_url = stripe_client.create_billing_portal_session(
                customer_id=user.stripe_customer_id,
                return_url=return_url,
            )
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="Unable to create billing portal session. Please contact support.",
            )

    return {"url": portal_url}


# ============================================================
# Premium Status Endpoint
# ============================================================

@router.get("/status")
def get_billing_status(
    fb_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    uid = fb_user.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="Missing Firebase uid")

    user = db.query(User).filter(User.firebase_uid == uid).first()
    if not user:
        return {
            "is_premium": False,
            "stripe_customer_id": None,
            "premium_activated_at": None,
        }

    return {
        "is_premium": bool(user.is_premium),
        "stripe_customer_id": user.stripe_customer_id,
        "premium_activated_at": user.premium_activated_at,
    }


# ============================================================
# Stripe Webhook (Premium + Tipster subs)
# ============================================================

@router.post("/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """
    Handles both:
      1. CSB Premium
      2. Tipster subscriptions (Stripe Checkout)
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        event = stripe_client.parse_event(payload, sig_header)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid payload")

    event_type = event["type"]
    data = event["data"]["object"]
    metadata = data.get("metadata") or {}

    # Metadata from premium checkout
    firebase_uid = metadata.get("firebase_uid")

    # Metadata from tipster checkout
    user_id_meta = metadata.get("user_id")
    tipster_id_meta = metadata.get("tipster_id")

    # ============================================================
    # checkout.session.completed
    # ============================================================
    if event_type == "checkout.session.completed":

        # ---- CSB Premium ----
        if firebase_uid and not tipster_id_meta:
            user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
            if user:
                user.is_premium = True
                user.premium_activated_at = datetime.now(timezone.utc)
                if data.get("customer"):
                    user.stripe_customer_id = data["customer"]
                db.commit()
            return {"ok": True}

        # ---- Tipster subscription ----
        if user_id_meta and tipster_id_meta:
            try:
                viewer_id = int(user_id_meta)
                tipster_id = int(tipster_id_meta)
            except ValueError:
                # malformed metadata, ignore gracefully
                return {"ok": True}

            user = db.query(User).get(viewer_id)
            tipster = db.query(Tipster).get(tipster_id)
            if not user or not tipster:
                return {"ok": True}

            # Update customer ID if missing
            if data.get("customer"):
                user.stripe_customer_id = data["customer"]
                db.commit()

            # Create / update tipster subscription record
            subscription_id = data.get("subscription")
            now = datetime.now(timezone.utc)
            renews = now + timedelta(days=30)  # simple monthly period

            sub = (
                db.query(TipsterSubscription)
                .filter(
                    TipsterSubscription.user_id == user.id,
                    TipsterSubscription.tipster_id == tipster.id,
                )
                .first()
            )

            if not sub:
                sub = TipsterSubscription(
                    user_id=user.id,
                    tipster_id=tipster.id,
                    plan_name="Monthly",
                    price_cents=tipster.default_price_cents,
                    status="active",
                    provider="stripe",
                    provider_sub_id=subscription_id,
                    started_at=now,
                    renews_at=renews,
                )
                db.add(sub)
            else:
                sub.status = "active"
                sub.canceled_at = None
                sub.renews_at = renews
                sub.provider = "stripe"
                sub.provider_sub_id = subscription_id or sub.provider_sub_id
                if tipster.default_price_cents:
                    sub.price_cents = tipster.default_price_cents

            db.commit()
            return {"ok": True}

        # If it's some other Checkout session we don't care about
        return {"ok": True}

    # ============================================================
    # customer.subscription.updated / deleted
    # ============================================================
    if event_type in ("customer.subscription.updated", "customer.subscription.deleted"):
        customer_id = data.get("customer")
        subscription_id = data.get("id")
        status_str = data.get("status")  # active, past_due, canceled...

        # Try to detect which price this sub is for
        price_id = None
        items = (data.get("items") or {}).get("data") or []
        if items:
            price_obj = items[0].get("price") or {}
            price_id = price_obj.get("id")

        # ---- Premium status (only if this is the Premium price) ----
        if customer_id and price_id == settings.STRIPE_PREMIUM_PRICE_ID:
            user = (
                db.query(User)
                .filter(User.stripe_customer_id == customer_id)
                .first()
            )
            if user:
                is_active = status_str in ("active", "trialing")
                user.is_premium = bool(is_active)
                db.commit()

        # ---- Tipster subscription status (by provider_sub_id) ----
        if subscription_id:
            tip_sub = (
                db.query(TipsterSubscription)
                .filter(TipsterSubscription.provider_sub_id == subscription_id)
                .first()
            )
            if tip_sub:
                if status_str in ("active", "trialing"):
                    tip_sub.status = "active"
                else:
                    tip_sub.status = "canceled"
                    tip_sub.canceled_at = datetime.now(timezone.utc)
                    tip_sub.renews_at = None
                db.commit()

        return {"ok": True}

    # For all other events we don't care about yet
    return {"ok": True}