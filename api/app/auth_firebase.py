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

bearer = HTTPBearer(auto_error=False)


def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
    db: Session = Depends(get_db),
):
    """
    Strict auth:
      - Requires a valid Firebase ID token
      - Ensures a local User row exists (auto-create)
      - Returns a DICT of Firebase claims + db_user_id + is_premium

    All callers should treat the return as a dict, e.g. user.get("email").
    """
    if not creds or not creds.scheme.lower().startswith("bearer"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
        )

    try:
        claims = verify_id_token(creds.credentials)  # Firebase decoded token (dict)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Firebase token",
        )

    uid = claims.get("uid")
    if not uid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Firebase token (uid missing)",
        )

    email = (claims.get("email") or "").lower()
    display_name = claims.get("name")
    avatar_url = claims.get("picture")

    # Sync / upsert local User row
    try:
        user = db.query(User).filter(User.firebase_uid == uid).first()
        now = datetime.utcnow()

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

        claims["db_user_id"] = user.id
        claims["is_premium"] = bool(user.is_premium)
        claims["is_admin"] = bool(user.is_admin)

    except Exception:
        db.rollback()
        # Fallback – still let them through with claims only
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
    """
    if not creds or not creds.scheme.lower().startswith("bearer"):
        return None

    try:
        claims = verify_id_token(creds.credentials)
    except Exception:
        return None

    uid = claims.get("uid")
    if not uid:
        return None

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
    Guard for premium-only endpoints.
    Use in routes as:  user = Depends(require_premium)
    """
    if not user.get("is_premium"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required",
        )
    return user