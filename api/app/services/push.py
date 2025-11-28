# api/app/services/push.py
import os
import json
from typing import Iterable, Optional

import requests
from sqlalchemy.orm import Session

from ..models import User, Tipster, TipsterPick, TipsterFollow, TipsterSubscription, PushToken, Fixture

FCM_SERVER_KEY = os.getenv("FIREBASE_FCM_SERVER_KEY")  # from Firebase console
FCM_ENDPOINT = "https://fcm.googleapis.com/fcm/send"


def _send_fcm_to_tokens(
    tokens: Iterable[str],
    title: str,
    body: str,
    data: Optional[dict] = None,
) -> None:
    """
    Low-level sender to FCM. If no server key, logs and bails.
    """
    tokens = [t for t in tokens if t]
    if not tokens:
        return

    if not FCM_SERVER_KEY:
        print("[push] No FIREBASE_FCM_SERVER_KEY – skipping push.")
        print("Title:", title)
        print("Body:", body)
        print("Tokens:", tokens[:3], "…")
        return

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"key={FCM_SERVER_KEY}",
    }

    payload = {
        "registration_ids": tokens,
        "notification": {
            "title": title,
            "body": body,
        },
        "data": data or {},
        "priority": "high",
    }

    try:
        resp = requests.post(FCM_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=5)
        if resp.status_code != 200:
            print("[push] FCM error:", resp.status_code, resp.text[:500])
    except Exception as e:
        print("[push] Exception sending FCM:", e)


def _eligible_users_for_pick(db: Session, tip: Tipster, pick: TipsterPick) -> list[User]:
    """
    Returns User rows who:
      • follow this tipster
      • AND satisfy premium/subscription gating for this pick
    """
    # 1) followers (by email)
    follower_rels = (
        db.query(TipsterFollow)
        .filter(TipsterFollow.tipster_id == tip.id)
        .all()
    )
    follower_emails = {fr.follower_email.lower() for fr in follower_rels if fr.follower_email}

    if not follower_emails:
        return []

    # 2) matching users
    users = (
        db.query(User)
        .filter(User.email.in_(follower_emails))
        .all()
    )

    # 3) subscribers for this tipster (for extra gating)
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

        # gating logic
        if pick.is_subscriber_only:
            if is_subscriber:
                gated.append(u)
        elif pick.is_premium_only:
            if is_premium or is_subscriber:
                gated.append(u)
        else:
            # normal public pick
            gated.append(u)

    return gated


def send_new_pick_push(
    db: Session,
    tip: Tipster,
    pick: TipsterPick,
    fixture: Optional[Fixture],
) -> None:
    """
    High-level function: find eligible users -> grab their tokens -> send push.
    """
    users = _eligible_users_for_pick(db, tip, pick)
    if not users:
        return

    user_ids = [u.id for u in users]

    tokens = (
        db.query(PushToken)
        .filter(
            PushToken.user_id.in_(user_ids),
            PushToken.is_active == True,
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

    title = f"New pick from {tip.name} (@{tip.username})"
    body = f"{fixture_label}: {pick.market} @ {pick.price:.2f}"

    data = {
        "type": "new_pick",
        "tipster_username": tip.username,
        "fixture_id": pick.fixture_id,
        "fixture_path": fixture_path or "",
        "market": pick.market,
        "price": f"{pick.price:.2f}",
        "is_premium_only": bool(pick.is_premium_only),
        "is_subscriber_only": bool(getattr(pick, "is_subscriber_only", False)),
    }

    _send_fcm_to_tokens(token_strings, title, body, data)