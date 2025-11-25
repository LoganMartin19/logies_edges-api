# api/app/services/stripe_client.py
from __future__ import annotations
import stripe
from typing import Optional
from datetime import datetime, timezone

from ..settings import settings

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY


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
    If connect_account_id + application_fee_percent are provided, use
    Stripe Connect revenue share.
    """
    metadata = metadata or {}

    subscription_data: dict = {
        "metadata": metadata,
    }

    if connect_account_id:
        # send net revenue to tipster
        subscription_data["transfer_data"] = {
            "destination": connect_account_id,
        }
        # your platform cut in percent
        if application_fee_percent is not None:
            subscription_data["application_fee_percent"] = application_fee_percent

    session = stripe.checkout.Session.create(
        mode="subscription",
        customer=customer_id,
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=success_url,
        cancel_url=cancel_url,
        subscription_data=subscription_data,
        metadata=metadata,
    )

    return session["url"]


# ============================================================
# Shared helpers
# ============================================================

def get_or_create_customer(email: str) -> stripe.Customer:
    customers = stripe.Customer.list(email=email, limit=1).data
    if customers:
        return customers[0]
    return stripe.Customer.create(email=email)


def parse_event(payload: bytes, sig_header: str) -> stripe.Event:
    return stripe.Webhook.construct_event(
        payload=payload,
        sig_header=sig_header,
        secret=settings.STRIPE_WEBHOOK_SECRET,
    )


def create_billing_portal_session(customer_id: str, return_url: str) -> str:
    session = stripe.billing_portal.Session.create(
        customer=customer_id,
        return_url=return_url,
    )
    return session.url