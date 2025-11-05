from fastapi import APIRouter, Depends
from ..deps.current_user import current_user
from ..models import User

router = APIRouter(prefix="/api/users", tags=["users"])

@router.get("/me")
def me(user: User = Depends(current_user)):
    return {
        "id": user.id,
        "display_name": user.display_name,
        "email": user.email,
        "avatar_url": user.avatar_url,
        "firebase_uid": user.firebase_uid,
    }