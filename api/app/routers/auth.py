# api/app/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from ..auth_firebase import get_current_user
from ..db import get_db
from ..models import User

# 👇 still used for the manual test endpoint
from ..services.email import send_email
from ..templates.welcome_email import welcome_email_html

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/me")
def me(
    user = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Returns merged Firebase + DB info for the current user.
    Frontend uses this for account/profile/premium status.
    Includes email_picks_opt_in from the local User row.
    """
    db_user = None
    db_user_id = user.get("db_user_id")

    if db_user_id:
        db_user = db.query(User).get(db_user_id)

    return {
        "firebase_uid": user.get("uid"),
        "email": user.get("email"),
        "display_name": user.get("name"),
        "avatar_url": user.get("picture"),
        "db_user_id": db_user_id,
        "is_admin": user.get("is_admin", False),
        "is_premium": user.get("is_premium", False),
        # ⭐ NEW: email picks preference (default True if missing)
        "email_picks_opt_in": bool(db_user.email_picks_opt_in)
        if db_user and db_user.email_picks_opt_in is not None
        else True,
    }


# ---------- Email preferences ----------

class EmailPrefsIn(BaseModel):
    email_picks_opt_in: bool


@router.post("/email-preferences")
def update_email_preferences(
    payload: EmailPrefsIn,
    user = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Update the logged-in user's email preferences.

    Currently only:
      - email_picks_opt_in: whether they want featured/premium pick emails.
    """
    db_user_id = user.get("db_user_id")
    if not db_user_id:
        raise HTTPException(status_code=400, detail="No linked user row")

    db_user = db.query(User).get(db_user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    db_user.email_picks_opt_in = payload.email_picks_opt_in
    db.commit()
    db.refresh(db_user)

    return {
        "ok": True,
        "email_picks_opt_in": bool(db_user.email_picks_opt_in),
    }


# ---------- Test welcome email endpoint (manual trigger) ----------

@router.post("/welcome-email/test")
def send_test_welcome_email(user = Depends(get_current_user)):
    """
    Sends the welcome email to the currently logged-in user.
    Handy for testing Resend without creating new accounts.
    """
    email = user.get("email")
    name = user.get("name") or email

    if not email:
        raise HTTPException(status_code=400, detail="No email on your account")

    html = welcome_email_html(name)
    send_email(
        to=email,
        subject="Test: Welcome to Chartered Sports Betting",
        html=html,
    )

    return {"ok": True}