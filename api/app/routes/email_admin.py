# api/app/routes/email_admin.py
from __future__ import annotations

from datetime import date
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import FeaturedPick, User
from ..auth_firebase import get_current_user
from ..services.email import send_email
from ..templates.featured_picks_email import featured_picks_email_html

router = APIRouter(prefix="/admin/email", tags=["admin-email"])


def require_admin(user=Depends(get_current_user)):
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin only")
    return user


@router.post("/featured-picks")
def send_featured_picks_digest(
    day: date | None = Query(default=None, description="UTC day for the card (YYYY-MM-DD)"),
    premium_only: bool = Query(
        default=False,
        description="If true, only email users with is_premium = true",
    ),
    db: Session = Depends(get_db),
    _admin=Depends(require_admin),
):
    """
    Send a Featured Picks digest to CSB users.

    Behaviour:
      - Picks are split into free vs premium using FeaturedPick.is_premium_only.
      - Premium users always receive *all* picks.
      - Free users receive only free picks, plus a teaser saying how many
        extra premium picks are live on the dashboard.
      - If `premium_only=true`, we *only* target premium users.
    """
    if day is None:
        day = date.today()

    # 1) Load featured picks for the day
    picks: List[FeaturedPick] = (
        db.query(FeaturedPick)
        .filter(FeaturedPick.day == day)
        .order_by(FeaturedPick.kickoff_utc.asc())
        .all()
    )
    if not picks:
        raise HTTPException(status_code=404, detail="No featured picks for that day")

    # Convert to simple dicts for the template, including premium flag
    pick_dicts = [
        {
            "comp": fp.comp,
            "home_team": fp.home_team,
            "away_team": fp.away_team,
            "kickoff_utc": fp.kickoff_utc,
            "market": fp.market,
            "bookmaker": fp.bookmaker,
            "price": fp.price,
            "edge": fp.edge,
            "is_premium_only": bool(fp.is_premium_only),
        }
        for fp in picks
    ]

    # Pre-split counts for convenience
    free_picks_all = [p for p in pick_dicts if not p["is_premium_only"]]
    premium_picks_all = [p for p in pick_dicts if p["is_premium_only"]]
    free_count_all = len(free_picks_all)
    premium_count_all = len(premium_picks_all)

    # 2) Find recipients
    q = db.query(User).filter(User.email.isnot(None))
    if premium_only:
        q = q.filter(User.is_premium.is_(True))

    recipients: List[User] = q.all()
    if not recipients:
        raise HTTPException(status_code=404, detail="No recipients found")

    sent = 0
    skipped_no_free = 0

    for u in recipients:
        is_premium_user = bool(u.is_premium)
        email = u.email
        if not email:
            continue

        # Determine which picks this user actually sees
        if is_premium_user:
            user_picks = pick_dicts
            user_free_count = free_count_all
            user_premium_count = premium_count_all
        else:
            # Free user → only free picks
            user_picks = free_picks_all
            user_free_count = free_count_all
            user_premium_count = premium_count_all

            # If no free picks today, skip emailing this free user
            if not user_picks:
                skipped_no_free += 1
                continue

        # Build subject line per user
        day_str = day.strftime("%d %b %Y")
        if is_premium_user:
            if user_free_count and user_premium_count:
                subject = f"CSB Featured & Premium Picks — {day_str}"
            elif user_premium_count:
                subject = f"CSB Premium Picks — {day_str}"
            else:
                subject = f"CSB Featured Picks — {day_str}"
        else:
            if user_premium_count:
                subject = f"CSB Free Picks (+{user_premium_count} premium) — {day_str}"
            else:
                subject = f"CSB Free Picks — {day_str}"

        try:
            html = featured_picks_email_html(
                day=day,
                picks=user_picks,
                recipient_name=u.display_name or (email.split("@")[0]),
                is_premium_user=is_premium_user,
                free_count=user_free_count,
                premium_count=user_premium_count,
            )
            send_email(
                to=email,
                subject=subject,
                html=html,
            )
            sent += 1
        except Exception as e:
            # don't blow up whole run if one address explodes
            print(f"[email_admin] Failed to email {email}: {e}")

    return {
        "ok": True,
        "day": str(day),
        "picks_total": len(picks),
        "free_picks": free_count_all,
        "premium_picks": premium_count_all,
        "recipients": len(recipients),
        "sent": sent,
        "skipped_no_free": skipped_no_free,
        "premium_only_param": premium_only,
    }