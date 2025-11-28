# api/app/routers/notifications.py
from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import User, PushToken
from ..auth_firebase import get_current_user
from .tipsters import _get_or_create_user_by_claims  # reuse helper

router = APIRouter(prefix="/api/notifications", tags=["notifications"])


class RegisterTokenIn(BaseModel):
    token: str
    platform: Literal["web", "ios", "android"] = "web"


@router.post("/register-token")
def register_push_token(
    payload: RegisterTokenIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Store or update a push token for the current user.
    Called from the frontend after FCM returns a registration token.
    """
    if not payload.token:
        raise HTTPException(400, "token is required")

    # ensure user row exists
    user_row: User = _get_or_create_user_by_claims(db, user)

    existing = (
        db.query(PushToken)
        .filter(
            PushToken.user_id == user_row.id,
            PushToken.token == payload.token,
        )
        .first()
    )

    now = datetime.utcnow()
    if existing:
        existing.platform = payload.platform
        existing.is_active = True
        existing.last_used_at = now
        db.commit()
        return {"ok": True, "status": "updated"}

    db.add(
        PushToken(
            user_id=user_row.id,
            token=payload.token,
            platform=payload.platform,
            is_active=True,
            created_at=now,
        )
    )
    db.commit()
    return {"ok": True, "status": "created"}


class UnregisterTokenIn(BaseModel):
    token: str


@router.post("/unregister-token")
def unregister_push_token(
    payload: UnregisterTokenIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Optional: turn off a specific token for this user (e.g. logout).
    """
    user_row: User = _get_or_create_user_by_claims(db, user)

    row = (
        db.query(PushToken)
        .filter(
            PushToken.user_id == user_row.id,
            PushToken.token == payload.token,
        )
        .first()
    )
    if row:
        row.is_active = False
        db.commit()

    return {"ok": True}