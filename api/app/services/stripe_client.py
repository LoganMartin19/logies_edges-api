# api/app/services/stripe_client.py
from __future__ import annotations

import os
import json
from typing import Optional, Dict, Any
from datetime import datetime, timezone

import stripe

from ..settings import settings

# ============================================================
# Stripe config
# ============================================================

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

# Default platform cut (can override by env)
DEFAULT_PLATFORM_FEE_PERCENT = float(os.getenv("PLATFORM_FEE_PERCENT", "15.0"))

# Webhook signing secret (for /api/billing/webhook)
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")


# ============================================================
# Webhook Parser (used by billing.py)
# ============================================================

def parse_event(payload: bytes, sig_header: str):
    """
    Parse + verify a Stripe webhook event using the Signing Secret.

    If STRIPE_WEBHOOK_SECRET is missing, falls back to plain JSON parse
    (useful for local dev / ngrok).
    """
    if STRIPE_WEBHOOK_SECRET:
        try:
            return stripe.Webhook.construct_event(
                payload=payload,
                sig_header=sig_header,
                secret=STRIPE_WEBHOOK_SECRET,
            )
        except stripe.error.SignatureVerificationError as e:
            print("❌ Stripe signature verification FAILED:", e)
            raise
        except Exception as e:
            print("❌ Stripe webhook parse FAILED:", e)
            raise

    # Dev fallback
    print("⚠️ STRIPE_WEBHOOK_SECRET not set — skipping signature verification")
    return stripe.Event.construct_from(json.loads(payload), stripe.api_key)


# ============================================================
# Helper: customers (used by billing portal etc.)
# ============================================================

def get_or_create_customer(email: str) -> stripe.Customer:
    """
    Simple helper: find an existing customer by email or create one.
    Used by the billing portal endpoint.
    """
    if not STRIPE_SECRET_KEY:
        raise ValueError("Stripe not configured")

    # Stripe's search API – safe for sandbox + small scale
    result = stripe.Customer.search(query=f"email:'{email}'")
    if result.data:
        return result.data[0]

    return stripe.Customer.create(email=email)


def create_billing_portal_session(customer_id: str, return_url: str) -> str:
    """
    Create a Stripe Billing Portal session so the user can manage
    all of their subscriptions (Premium + tipsters).

    Returns the portal URL as a plain string.
    """
    if not STRIPE_SECRET_KEY:
        raise ValueError("Stripe not configured")

    session = stripe.billing_portal.Session.create(
        customer=customer_id,
        return_url=return_url,
    )
    return session.url


# ============================================================
# Premium Checkout (CSB Premium plan)
# ============================================================

def create_premium_checkout_session(
    customer_email: str,
    firebase_uid: str,
    success_url: str,
    cancel_url: str,
) -> str:
    """
    Create a Stripe Checkout Session for the CSB Premium plan.
    Returns the session URL as a plain string.
    """
    if not settings.STRIPE_PREMIUM_PRICE_ID:
        raise ValueError("Missing STRIPE_PREMIUM_PRICE_ID")

    session = stripe.checkout.Session.create(
        mode="subscription",
        payment_method_types=["card"],
        line_items=[
            {
                "price": settings.STRIPE_PREMIUM_PRICE_ID,
                "quantity": 1,
            }
        ],
        customer_email=customer_email,
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={
            "firebase_uid": firebase_uid,
            "product": "csb_premium",
        },
    )

    # 👇 return just the URL
    return session.url


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
    the subscription will:
      - route funds to the tipster's Connect account
      - apply your platform cut via application_fee_percent.
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
        line_items=[
            {
                "price": price_id,
                "quantity": 1,
            }
        ],
        success_url=success_url,
        cancel_url=cancel_url,
        subscription_data=subscription_data,
        metadata=metadata,
    )

    # Tipster flow expects a plain URL string
    return session.url