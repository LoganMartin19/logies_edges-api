# api/app/routes/preferences.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import UserPreference
from ..auth_firebase import get_current_user  # ✅ use Firebase auth

router = APIRouter(prefix="/me/preferences", tags=["preferences"])


class PrefsIn(BaseModel):
    favorite_sports: list[str] = []   # e.g. ["football", "nba"]
    favorite_teams: list[str] = []    # e.g. ["Celtic", "Arsenal"]
    favorite_leagues: list[str] = []  # e.g. ["EPL", "SCO_PREM"]


class PrefsOut(PrefsIn):
    pass


@router.get("", response_model=PrefsOut)
def get_my_prefs(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),  # dict from auth_firebase
):
    user_id = current_user.get("db_user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User not found")

    prefs = (
        db.query(UserPreference)
        .filter(UserPreference.user_id == user_id)
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
    current_user = Depends(get_current_user),
):
    user_id = current_user.get("db_user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User not found")

    prefs = (
        db.query(UserPreference)
        .filter(UserPreference.user_id == user_id)
        .one_or_none()
    )

    if not prefs:
        prefs = UserPreference(
            user_id=user_id,
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