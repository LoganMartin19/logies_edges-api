# api/app/routers/auth.py
from fastapi import APIRouter, Depends
from ..auth_firebase import get_current_user

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