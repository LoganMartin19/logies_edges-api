# api/app/services/push.py
from typing import Iterable, Optional

from sqlalchemy.orm import Session

from .firebase import send_web_push_to_tokens
from ..models import (
    User,
    Tipster,
    TipsterPick,
    TipsterFollow,
    TipsterSubscription,
    PushToken,
    Fixture,
)


def _send_fcm_to_tokens(
    tokens: Iterable[str],
    title: str,
    body: str,
    data: Optional[dict] = None,
) -> None:
    """
    Thin wrapper over firebase.send_web_push_to_tokens so the rest of this
    module doesn’t need to know about Firebase internals.
    """
    token_list = [t for t in tokens if t]
    if not token_list:
        return

    # data payload will be coerced to strings inside firebase.send_web_push_to_tokens
    send_web_push_to_tokens(token_list, title, body, data or {})


def _eligible_users_for_pick(db: Session, tip: Tipster, pick: TipsterPick) -> list[User]:
    """
    Returns User rows who:
      • follow this tipster
      • AND satisfy premium/subscription gating for this pick
    """
    follower_rels = (
        db.query(TipsterFollow)
        .filter(TipsterFollow.tipster_id == tip.id)
        .all()
    )
    follower_emails = {
        fr.follower_email.lower()
        for fr in follower_rels
        if fr.follower_email
    }

    if not follower_emails:
        return []

    users = (
        db.query(User)
        .filter(User.email.in_(follower_emails))
        .all()
    )

    active_subs = (
        db.query(TipsterSubscription)
        .filter(
            TipsterSubscription.tipster_id == tip.id,
            TipsterSubscription.status == "active",
        )
        .all()
    )
    sub_user_ids = {s.user_id for s in active_subs}

    gated: list[User] = []
    for u in users:
        is_premium = bool(getattr(u, "is_premium", False))
        is_subscriber = u.id in sub_user_ids

        if pick.is_subscriber_only:
            if is_subscriber:
                gated.append(u)
        elif pick.is_premium_only:
            if is_premium or is_subscriber:
                gated.append(u)
        else:
            gated.append(u)

    return gated


def send_new_pick_push(
    db: Session,
    tip: Tipster,
    pick: TipsterPick,
    fixture: Optional[Fixture],
) -> None:
    """
    High-level: find eligible users → fetch active web push tokens → send FCM.
    """
    users = _eligible_users_for_pick(db, tip, pick)
    if not users:
        return

    user_ids = [u.id for u in users]

    tokens = (
        db.query(PushToken)
        .filter(
            PushToken.user_id.in_(user_ids),
            PushToken.is_active == True,   # noqa: E712
            PushToken.platform == "web",
        )
        .all()
    )
    token_strings = [t.token for t in tokens if t.token]

    if not token_strings:
        return

    fixture_label = "New pick"
    fixture_path = None
    if fixture:
        home = fixture.home_team or ""
        away = fixture.away_team or ""
        if home or away:
            fixture_label = f"{home} vs {away}"
        sport = (fixture.sport or "").lower()
        if sport in ("football", "soccer"):
            fixture_path = f"/fixture/{fixture.id}"
        elif sport == "nhl":
            fixture_path = f"/nhl/game/{fixture.id}"
        elif sport in ("cfb", "ncaaf"):
            fixture_path = f"/cfb/fixture/{fixture.id}"
        elif sport in ("nba", "basketball"):
            fixture_path = f"/basketball/game/{fixture.id}"

    price = float(pick.price or 0.0)

    title = f"New pick from {tip.name} (@{tip.username})"
    body = f"{fixture_label}: {pick.market} @ {price:.2f}"

    data = {
        "type": "new_pick",
        "tipster_username": tip.username,
        "fixture_id": pick.fixture_id,
        "fixture_path": fixture_path or "",
        "market": pick.market,
        "price": f"{price:.2f}",
        "is_premium_only": bool(pick.is_premium_only),
        "is_subscriber_only": bool(getattr(pick, "is_subscriber_only", False)),
    }

    _send_fcm_to_tokens(token_strings, title, body, data)