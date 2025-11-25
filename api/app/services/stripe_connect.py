# api/app/services/stripe_connect.py
from __future__ import annotations
import os
import stripe

from ..models import Tipster, User
from sqlalchemy.orm import Session
from fastapi import HTTPException

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
FRONTEND_BASE_URL = os.getenv(
    "FRONTEND_BASE_URL", "https://charteredsportsbetting.com"
)

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY


def ensure_express_account(db: Session, tipster: Tipster, owner: User) -> str:
    """
    Ensure this tipster has a Stripe Connect Express account.
    Returns stripe_account_id.
    """
    if tipster.stripe_account_id:
        return tipster.stripe_account_id

    if not STRIPE_SECRET_KEY:
        raise HTTPException(500, "Stripe not configured on server")

    acct = stripe.Account.create(
        type="express",
        country="GB",  # adjust if you’ll have non-UK tipsters
        email=owner.email or None,
        capabilities={
            "card_payments": {"requested": True},
            "transfers": {"requested": True},
        },
        business_type="individual",
        business_profile={
            "product_description": "Sports betting tips subscription",
            "url": FRONTEND_BASE_URL,
        },
        metadata={
            "tipster_id": str(tipster.id),
            "tipster_username": tipster.username,
        },
    )

    tipster.stripe_account_id = acct["id"]
    db.commit()
    db.refresh(tipster)
    return tipster.stripe_account_id


def create_onboarding_link(stripe_account_id: str) -> str:
    """
    Create an onboarding link for Express account.
    """
    if not STRIPE_SECRET_KEY:
        raise HTTPException(500, "Stripe not configured on server")

    link = stripe.AccountLink.create(
        account=stripe_account_id,
        refresh_url=f"{FRONTEND_BASE_URL}/creator/onboarding?refresh=1",
        return_url=f"{FRONTEND_BASE_URL}/creator/onboarding?complete=1",
        type="account_onboarding",
    )
    return link["url"]


def get_connect_status(stripe_account_id: str) -> dict:
    """
    Small status snapshot for UI.
    """
    if not STRIPE_SECRET_KEY:
        raise HTTPException(500, "Stripe not configured on server")

    acct = stripe.Account.retrieve(stripe_account_id)
    return {
        "id": acct["id"],
        "charges_enabled": acct.get("charges_enabled", False),
        "payouts_enabled": acct.get("payouts_enabled", False),
        "details_submitted": acct.get("details_submitted", False),
        "currently_due": acct.get("requirements", {}).get("currently_due", []),
    }

def create_login_link(stripe_account_id: str) -> str:
    """
    Generate a login link for the Stripe Express Dashboard.
    Tipsters use this to manage payout details.
    """
    if not STRIPE_SECRET_KEY:
        raise HTTPException(500, "Stripe not configured on server")

    link = stripe.Account.create_login_link(stripe_account_id)
    return link["url"]