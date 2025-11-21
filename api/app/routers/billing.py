# api/app/routers/billing.py
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from ..db import get_db
from ..auth_firebase import get_current_user    # Firebase claims
from ..models import User
from ..settings import settings
from ..services import stripe_client

router = APIRouter(prefix="/billing", tags=["billing"])


def _get_or_create_db_user(db: Session, fb_claims: dict) -> User:
    uid = fb_claims.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="Missing Firebase uid")

    user = db.query(User).filter(User.firebase_uid == uid).first()
    if user:
        return user

    # Fallback – create a row the first time we see this Firebase user
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


# ---------- Checkout: start subscription ----------

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

    # Make sure env vars are set
    if not settings.STRIPE_PREMIUM_PRICE_ID or not settings.STRIPE_SECRET_KEY:
        raise HTTPException(
            status_code=500,
            detail="Stripe is not configured on the server."
        )

    if not user.email:
        raise HTTPException(
            status_code=400,
            detail="User account must have an email to start checkout."
        )

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


# ---------- Customer Portal: manage / cancel subscription ----------

@router.post("/customer-portal")
def create_billing_portal_session(
    request: Request,
    fb_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Create a Stripe Billing Portal session so the user can
    manage their subscription (update card, cancel, etc.).

    Returns: { url: "https://billing.stripe.com/..." }
    """
    user = _get_or_create_db_user(db, fb_user)

    if not user.email:
        raise HTTPException(
            status_code=400,
            detail="User account must have an email to manage billing."
        )

    # Ensure we have a Stripe Customer ID for this user
    customer_id = user.stripe_customer_id
    if not customer_id:
        # Try to find or create a Stripe customer by email
        customer = stripe_client.get_or_create_customer(user.email)
        customer_id = customer.id
        user.stripe_customer_id = customer_id
        db.add(user)
        db.commit()

    origin = request.headers.get("origin") or (
        request.url.scheme + "://" + request.url.netloc
    )
    return_url = f"{origin}/account"

    portal_url = stripe_client.create_billing_portal_session(
        customer_id=customer_id,
        return_url=return_url,
    )

    return {"url": portal_url}


# ---------- Status endpoint: used by frontend to show Premium badge ----------

@router.get("/status")
def get_billing_status(
    fb_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Return current premium / billing status for the logged-in user.
    """
    uid = fb_user.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="Missing Firebase uid")

    user = db.query(User).filter(User.firebase_uid == uid).first()
    if not user:
        # Not in DB yet => definitely not premium
        return {
            "is_premium": False,
            "stripe_customer_id": None,
            "premium_activated_at": None,
        }

    return {
        "is_premium": bool(getattr(user, "is_premium", False)),
        "stripe_customer_id": getattr(user, "stripe_customer_id", None),
        "premium_activated_at": getattr(user, "premium_activated_at", None),
    }


# ---------- Stripe Webhook ----------

@router.post("/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """
    Stripe webhook endpoint.
    Configure this URL in your Stripe dashboard:
      https://your-api-url.com/billing/webhook
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        event = stripe_client.parse_event(payload, sig_header)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid payload",
        )

    event_type = event["type"]
    data = event["data"]["object"]

    # We attach firebase_uid to the Checkout Session metadata
    firebase_uid = (data.get("metadata") or {}).get("firebase_uid")

    if event_type == "checkout.session.completed":
        # User finished checkout successfully -> mark as premium
        if firebase_uid:
            user = db.query(User).filter(
                User.firebase_uid == firebase_uid
            ).first()
            if user:
                user.is_premium = True
                user.premium_activated_at = datetime.now(timezone.utc)
                # store Stripe customer id if present
                if data.get("customer"):
                    user.stripe_customer_id = data["customer"]
                db.add(user)
                db.commit()

    elif event_type in ("customer.subscription.deleted", "customer.subscription.updated"):
        # Subscription changed or was canceled
        customer_id = data.get("customer")
        if customer_id:
            user = db.query(User).filter(
                User.stripe_customer_id == customer_id
            ).first()
            if user:
                status_str = data.get("status")
                # Stripe statuses: active, trialing, past_due, canceled, unpaid, etc.
                is_active = status_str in ("active", "trialing")
                user.is_premium = bool(is_active)
                db.add(user)
                db.commit()

    return {"ok": True}