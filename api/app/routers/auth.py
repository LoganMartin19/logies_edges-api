# api/app/routers/auth.py
from fastapi import APIRouter, Depends
from ..auth_firebase import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])

@router.get("/me")
def me(user=Depends(get_current_user)):
    return {
        "uid": user.get("uid"),
        "email": user.get("email"),
        "name": user.get("name"),
        "picture": user.get("picture"),
    }