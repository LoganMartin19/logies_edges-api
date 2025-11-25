# api/app/services/stripe_client.py
from __future__ import annotations
import os
import stripe
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from ..settings import settings

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

# Default platform cut (can override by env)
DEFAULT_PLATFORM_FEE_PERCENT = float(os.getenv("PLATFORM_FEE_PERCENT", "15.0"))


# ============================================================
# Premium Checkout
# ============================================================

def create_premium_checkout_session(
    customer_email: str,
    firebase_uid: str,
    success_url: str,
    cancel_url: str,
) -> dict:
    """
    Create a Stripe Checkout Session for the CSB Premium plan.
    Returns: { url: "...", id: "cs_test_..." }
    """
    if not settings.STRIPE_PREMIUM_PRICE_ID:
        raise ValueError("Missing STRIPE_PREMIUM_PRICE_ID")

    session = stripe.checkout.Session.create(
        mode="subscription",
        payment_method_types=["card"],
        line_items=[{
            "price": settings.STRIPE_PREMIUM_PRICE_ID,
            "quantity": 1,
        }],
        customer_email=customer_email,
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={
            "firebase_uid": firebase_uid,
            "product": "csb_premium",
        },
    )
    return {"url": session.url, "id": session.id}


# ============================================================
# General-purpose subscription checkout (Tipsters)
# ============================================================

def create_subscription_checkout_session(
    customer_id: str,
    price_id: str,
    success_url: str,
    cancel_url: str,
    metadata: dict | None = None,
    connect_account_id: str | None = None,
    application_fee_percent: float | None = None,
) -> str:
    """
    Create a Checkout Session for a subscription.
    If connect_account_id + application_fee_percent are provided,
    the subscription will route funds to the tipster and apply
    your platform cut.
    """
    metadata = metadata or {}

    # Fee defaults to 15% if not provided
    fee_pct = (
        float(application_fee_percent)
        if application_fee_percent is not None
        else DEFAULT_PLATFORM_FEE_PERCENT
    )

    subscription_data: Dict[str, Any] = {
        "metadata": metadata,
    }

    if connect_account_id:
        subscription_data["transfer_data"] = {
            "destination": connect_account_id,
        }
        subscription_data["application_fee_percent"] = fee_pct

    session = stripe.checkout.Session.create(
        mode="subscription",
        customer=customer_id,
        line_items=[{
            "price": price_id,
            "quantity": 1
        }],
        success_url=success_url,
        cancel_url=cancel_url,
        subscription_data=subscription_data,
        metadata=metadata,
    )

    return session.url