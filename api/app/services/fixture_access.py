# api/app/services/fixture_access.py
from __future__ import annotations

from datetime import datetime, date
from typing import Tuple

from sqlalchemy.orm import Session

from ..models import User, Fixture, UserFixtureAccess

# 🚧 tweak if you ever want 5 / 20 etc
FREE_DAILY_FIXTURE_LIMIT = 10


def _today_utc() -> date:
    """Return today's date in UTC."""
    return datetime.utcnow().date()


def user_has_fixture_access(
    db: Session,
    user_id: int | None,
    fixture_id: int | None,
) -> bool:
    """
    Simple read-only check: has this user already unlocked this fixture?

    Used so you can:
      - show full edges if True
      - show locked / teaser view if False (and don't auto-spend a token)
    """
    if not user_id or not fixture_id:
        return False

    row = (
        db.query(UserFixtureAccess)
        .filter(
            UserFixtureAccess.user_id == user_id,
            UserFixtureAccess.fixture_id == fixture_id,
        )
        .first()
    )
    return row is not None


def user_daily_access_count(
    db: Session,
    user_id: int | None,
    day: date | None = None,
) -> int:
    """
    Count how many distinct fixtures this user has unlocked *today*.
    This is what we compare to FREE_DAILY_FIXTURE_LIMIT.
    """
    if not user_id:
        return 0

    if day is None:
        day = _today_utc()

    return (
        db.query(UserFixtureAccess)
        .filter(
            UserFixtureAccess.user_id == user_id,
            UserFixtureAccess.day == day,
        )
        .count()
    )


def can_unlock_new_fixture(
    db: Session,
    user_id: int | None,
    day: date | None = None,
    *,
    limit: int = FREE_DAILY_FIXTURE_LIMIT,
) -> Tuple[bool, int, int]:
    """
    Convenience helper:
      returns (can_unlock, used_today, limit)

    - Guests (no user_id) → (False, 0, limit)
    """
    if not user_id:
        return False, 0, limit

    used = user_daily_access_count(db, user_id, day=day)
    return used < limit, used, limit


def ensure_fixture_access(
    db: Session,
    user: User | None,
    fixture: Fixture | None,
    *,
    is_premium: bool,
    limit: int = FREE_DAILY_FIXTURE_LIMIT,
) -> Tuple[bool, int, int]:
    """
    Main write-path for gating when user clicks into a fixture.

    Returns:
      (has_access, used_today, limit)

    Behaviour:
      - If no logged-in user → no access (caller should show teaser / login CTA).
      - If premium user → always has access (no quota); we still create a row
        the first time so you can track total unlocked if you want.
      - If already unlocked → has access, doesn't change today's count.
      - Else:
          * if under daily limit → create UserFixtureAccess row, grant access
          * if at/over limit → no new row, caller should show "limit reached" UI
    """
    if not user or not user.id or not fixture:
        return False, 0, limit

    user_id = user.id
    fixture_id = fixture.id

    today = _today_utc()

    # Premium users → unlimited views, but still mark first access if you like
    if is_premium:
        # if we already have a row, just return
        if user_has_fixture_access(db, user_id, fixture_id):
            used_today = user_daily_access_count(db, user_id, day=today)
            return True, used_today, limit

        # create one row for analytics / "recently viewed", but ignore limit
        access = UserFixtureAccess(
            user_id=user_id,
            fixture_id=fixture_id,
            day=today,
        )
        db.add(access)
        db.commit()

        used_today = user_daily_access_count(db, user_id, day=today)
        return True, used_today, limit

    # Non-premium: check if already unlocked
    if user_has_fixture_access(db, user_id, fixture_id):
        used_today = user_daily_access_count(db, user_id, day=today)
        return True, used_today, limit

    # Not unlocked yet → enforce daily quota
    used_today = user_daily_access_count(db, user_id, day=today)
    if used_today >= limit:
        # hit the wall, do NOT create a row
        return False, used_today, limit

    # under limit → create new access row and grant
    access = UserFixtureAccess(
        user_id=user_id,
        fixture_id=fixture_id,
        day=today,
    )
    db.add(access)
    db.commit()

    return True, used_today + 1, limit