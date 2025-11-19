# api/app/auth_firebase.py
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .services.firebase import verify_id_token
from .db import get_db
from .models import User

# Bearer auth scheme (reads "Authorization: Bearer <token>")
bearer = HTTPBearer(auto_error=False)


def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
    db=Depends(get_db),
) -> User:
    """
    Strict auth:
      - Requires a valid Firebase ID token
      - Resolves (or auto-creates) a local User row
      - Returns the SQLAlchemy User instance

    Use this for routes that MUST be logged in.
    """
    if not creds or not creds.scheme.lower().startswith("bearer"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
        )

    try:
        decoded = verify_id_token(creds.credentials)  # Firebase decoded token
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Firebase token",
        )

    firebase_uid = decoded.get("uid")
    if not firebase_uid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing UID",
        )

    # Look up local user
    user: Optional[User] = (
        db.query(User).filter(User.firebase_uid == firebase_uid).first()
    )

    # Option A: auto-create if missing
    if not user:
        user = User(
            firebase_uid=firebase_uid,
            email=decoded.get("email"),
            display_name=decoded.get("name"),
            avatar_url=decoded.get("picture"),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    return user


def optional_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
    db=Depends(get_db),
) -> Optional[User]:
    """
    Soft auth:
      - If no/invalid token → returns None
      - If valid → returns User row

    Use this on endpoints where login is OPTIONAL:
      e.g. featured picks, tipster cards, dashboard, etc.
    """
    if not creds or not creds.scheme.lower().startswith("bearer"):
        return None

    try:
        decoded = verify_id_token(creds.credentials)
    except Exception:
        return None

    firebase_uid = decoded.get("uid")
    if not firebase_uid:
        return None

    return db.query(User).filter(User.firebase_uid == firebase_uid).first()


def require_premium(user: User = Depends(get_current_user)) -> User:
    """
    Guard for premium-only endpoints.
    Use like: Depends(require_premium)
    """
    if not user.is_premium:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required",
        )
    return user