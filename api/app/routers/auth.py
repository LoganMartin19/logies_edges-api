# api/app/routers/auth.py
from fastapi import APIRouter, Depends
from ..auth_firebase import get_current_user
from ..models import User

router = APIRouter(prefix="/auth", tags=["auth"])

@router.get("/me")
def me(user: User = Depends(get_current_user)):
    return {
        "id": user.id,
        "firebase_uid": user.firebase_uid,
        "email": user.email,
        "display_name": user.display_name,
        "avatar_url": user.avatar_url,
        "is_admin": user.is_admin,
        "is_premium": user.is_premium,
        "premium_activated_at": (
            user.premium_activated_at.isoformat() if user.premium_activated_at else None
        ),
        "tipster_id": user.tipster_id,
    }