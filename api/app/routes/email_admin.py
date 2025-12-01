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
    day: date | None = Query(
        default=None,
        description="UTC day for the card (YYYY-MM-DD)",
    ),
    premium_only: bool = Query(
        default=True,
        description="If true, only email users with is_premium = true",
    ),
    db: Session = Depends(get_db),
    _admin=Depends(require_admin),
):
    """
    Send a Featured Picks digest to CSB users.

    - Default: today's card, premium users only.
    - You can override `day` and `premium_only` via query params.
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

    # Convert to simple dicts for the template
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
            # 👇 NEW: let the template see which picks are premium-only
            "is_premium_only": fp.is_premium_only,
        }
        for fp in picks
    ]

    # 2) Find recipients
    q = db.query(User).filter(User.email.isnot(None))
    if premium_only:
        q = q.filter(User.is_premium.is_(True))

    recipients: List[User] = q.all()
    if not recipients:
        raise HTTPException(status_code=404, detail="No recipients found")

    subject = f"CSB Featured Picks — {day.strftime('%d %b %Y')}"

    sent = 0
    for u in recipients:
        try:
            html = featured_picks_email_html(
                day=day,
                picks=pick_dicts,
                recipient_name=u.display_name or (u.email or "").split("@")[0],
                # 👇 Let the template auto-detect free/premium/mixed
                premium_only=None,
            )
            send_email(
                to=u.email,
                subject=subject,
                html=html,
            )
            sent += 1
        except Exception as e:
            # don't blow up whole run if one address explodes
            print(f"[email_admin] Failed to email {u.email}: {e}")

    return {
        "ok": True,
        "day": str(day),
        "premium_only": premium_only,
        "picks": len(picks),
        "recipients": len(recipients),
        "sent": sent,
    }