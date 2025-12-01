# api/app/auth_firebase.py
from __future__ import annotations

from typing import Optional
from datetime import datetime

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .services.firebase import verify_id_token
from .db import get_db
from .models import User

# 👇 welcome email bits
from .services.email import send_email
from .templates.welcome_email import welcome_email_html

bearer = HTTPBearer(auto_error=False)


def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
    db: Session = Depends(get_db),
) -> dict:
    """
    Strict auth:
      - Requires a valid Firebase ID token
      - Ensures a local User row exists (auto-create)
      - Returns a DICT of Firebase claims + db_user_id + is_premium/is_admin
      - Sends a one-time welcome email on first creation
    """
    if not creds or not creds.scheme.lower().startswith("bearer"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
        )

    # Verify Firebase ID token
    claims = verify_id_token(creds.credentials)
    if not isinstance(claims, dict) or not claims.get("uid"):
        # add a print so we can see what's wrong in Render logs
        print("verify_id_token returned:", claims)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Firebase token",
        )

    uid = claims.get("uid")
    email = (claims.get("email") or "").lower()
    display_name = claims.get("name")
    avatar_url = claims.get("picture")

    # Sync / upsert local User row
    try:
        user = db.query(User).filter(User.firebase_uid == uid).first()
        now = datetime.utcnow()
        is_new = user is None  # 👈 track if we're creating for the first time

        if not user:
            user = User(
                firebase_uid=uid,
                email=email,
                display_name=display_name,
                avatar_url=avatar_url,
                created_at=now,
                updated_at=now,
            )
            db.add(user)
        else:
            changed = False
            if email and user.email != email:
                user.email = email
                changed = True
            if display_name and user.display_name != display_name:
                user.display_name = display_name
                changed = True
            if avatar_url and user.avatar_url != avatar_url:
                user.avatar_url = avatar_url
                changed = True
            if changed:
                user.updated_at = now

        db.commit()
        db.refresh(user)

        # 👇 Send welcome email only once, on first creation
        if is_new and email:
            try:
                html = welcome_email_html(display_name or email or "there")
                send_email(
                    to=email,
                    subject="Welcome to Chartered Sports Betting",
                    html=html,
                )
            except Exception as e:
                # Never block login for email issues
                print("Failed to send welcome email:", e)

        # Attach DB info back onto claims so the rest of the app can use it
        claims["db_user_id"] = user.id
        claims["is_premium"] = bool(user.is_premium)
        claims["is_admin"] = bool(user.is_admin)

    except Exception as e:
        print("Error syncing user from Firebase claims:", e)
        db.rollback()
        claims.setdefault("db_user_id", None)
        claims.setdefault("is_premium", False)
        claims.setdefault("is_admin", False)

    return claims


def optional_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
    db: Session = Depends(get_db),
) -> Optional[dict]:
    """
    Soft auth:
      - If no/invalid token → returns None
      - If valid → returns the SAME claims dict as get_current_user
        (but without sending emails / creating users)
    """
    if not creds or not creds.scheme.lower().startswith("bearer"):
        return None

    claims = verify_id_token(creds.credentials)
    if not isinstance(claims, dict) or not claims.get("uid"):
        return None

    uid = claims.get("uid")
    email = (claims.get("email") or "").lower()

    user = db.query(User).filter(User.firebase_uid == uid).first()
    if user:
        claims["db_user_id"] = user.id
        claims["is_premium"] = bool(user.is_premium)
        claims["is_admin"] = bool(user.is_admin)
    else:
        claims["db_user_id"] = None
        claims["is_premium"] = False
        claims["is_admin"] = False

    return claims


def require_premium(user=Depends(get_current_user)):
    """
    Guard for premium-only routes.
    """
    if not user.get("is_premium"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required",
        )
    return user