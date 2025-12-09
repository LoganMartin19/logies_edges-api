# api/app/routes/preferences.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import User, UserPreference
from ..auth import get_current_user  # adjust import to your auth helper

router = APIRouter(prefix="/me/preferences", tags=["preferences"])


class PrefsIn(BaseModel):
    favorite_teams: list[str] = []
    favorite_leagues: list[str] = []  # use your comp codes like "EPL","SCO_PRM"


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
        return PrefsOut(favorite_teams=[], favorite_leagues=[])
    return PrefsOut(
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
            favorite_teams=payload.favorite_teams,
            favorite_leagues=payload.favorite_leagues,
        )
        db.add(prefs)
    else:
        prefs.favorite_teams = payload.favorite_teams
        prefs.favorite_leagues = payload.favorite_leagues

    db.commit()
    db.refresh(prefs)

    return PrefsOut(
        favorite_teams=prefs.favorite_teams or [],
        favorite_leagues=prefs.favorite_leagues or [],
    )