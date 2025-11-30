# api/app/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException
from ..auth_firebase import get_current_user

# 👇 NEW imports
from ..services.email import send_email
from ..templates.welcome_email import welcome_email_html

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/me")
def me(user = Depends(get_current_user)):
    """
    Returns merged Firebase + DB info for the current user.
    Frontend uses this for account/profile/premium status.
    """
    return {
        "firebase_uid": user.get("uid"),
        "email": user.get("email"),
        "display_name": user.get("name"),
        "avatar_url": user.get("picture"),
        "db_user_id": user.get("db_user_id"),
        "is_admin": user.get("is_admin", False),
        "is_premium": user.get("is_premium", False),
    }


# 👇 NEW: test welcome email endpoint
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