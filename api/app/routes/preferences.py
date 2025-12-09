# api/app/routes/preferences.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import UserPreference
from ..auth_firebase import get_current_user  # ✅ use firebase auth

router = APIRouter(prefix="/me/preferences", tags=["preferences"])


class PrefsIn(BaseModel):
    favorite_sports: list[str] = []      # 👈 NEW
    favorite_teams: list[str] = []
    favorite_leagues: list[str] = []


class PrefsOut(PrefsIn):
    pass


@router.get("", response_model=PrefsOut)
def get_my_prefs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    prefs = (
        db.query(UserPreference)
        .filter(UserPreference.user_id == current_user.id)
        .one_or_none()
    )
    if not prefs:
        return PrefsOut(
            favorite_sports=[],
            favorite_teams=[],
            favorite_leagues=[],
        )

    return PrefsOut(
        favorite_sports=prefs.favorite_sports or [],
        favorite_teams=prefs.favorite_teams or [],
        favorite_leagues=prefs.favorite_leagues or [],
    )


@router.post("", response_model=PrefsOut)
def set_my_prefs(
    payload: PrefsIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    prefs = (
        db.query(UserPreference)
        .filter(UserPreference.user_id == current_user.id)
        .one_or_none()
    )

    if not prefs:
        prefs = UserPreference(
            user_id=current_user.id,
            favorite_sports=payload.favorite_sports,
            favorite_teams=payload.favorite_teams,
            favorite_leagues=payload.favorite_leagues,
        )
        db.add(prefs)
    else:
        prefs.favorite_sports = payload.favorite_sports
        prefs.favorite_teams = payload.favorite_teams
        prefs.favorite_leagues = payload.favorite_leagues

    db.commit()
    db.refresh(prefs)

    return PrefsOut(
        favorite_sports=prefs.favorite_sports or [],
        favorite_teams=prefs.favorite_teams or [],
        favorite_leagues=prefs.favorite_leagues or [],
    )