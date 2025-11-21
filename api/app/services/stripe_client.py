# api/app/services/stripe_client.py
from __future__ import annotations
import stripe
from typing import Optional
from datetime import datetime, timezone

from ..settings import settings

stripe.api_key = settings.STRIPE_SECRET_KEY


def create_premium_checkout_session(
    customer_email: str,
    firebase_uid: str,
    success_url: str,
    cancel_url: str,
) -> str:
    """
    Create a Stripe Checkout Session for the CSB Premium plan.
    Returns: session.url (redirect URL)
    """
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
    return session.url


def get_or_create_customer(email: str) -> stripe.Customer:
    """
    Optional helper if you later want explicit Customer records.
    """
    customers = stripe.Customer.list(email=email, limit=1).data
    if customers:
        return customers[0]
    return stripe.Customer.create(email=email)


def parse_event(payload: bytes, sig_header: str) -> stripe.Event:
    """
    Verify and parse a Stripe webhook event.
    """
    return stripe.Webhook.construct_event(
        payload=payload,
        sig_header=sig_header,
        secret=settings.STRIPE_WEBHOOK_SECRET,
    )

def create_billing_portal_session(customer_id: str, return_url: str) -> str:
    """
    Create a Stripe Billing Portal session so the user can manage
    their subscription (update card, cancel, etc.).
    """
    session = stripe.billing_portal.Session.create(
        customer=customer_id,
        return_url=return_url,
    )
    return session.url