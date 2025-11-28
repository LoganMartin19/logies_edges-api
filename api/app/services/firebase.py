# api/app/services/firebase.py
import os
import json
from typing import Optional, Sequence

try:
    import firebase_admin
    from firebase_admin import (
        auth as fb_auth,
        credentials as fb_credentials,
        messaging as fb_messaging,
    )
except Exception:  # package not installed locally
    firebase_admin = None
    fb_auth = None
    fb_credentials = None
    fb_messaging = None

_initialized = False


def _ensure_init() -> None:
    """
    Initialise the Firebase Admin SDK once, using either:

    - FIREBASE_SERVICE_ACCOUNT_JSON (recommended in production), or
    - default credentials (for local dev if you have them configured).
    """
    global _initialized
    if _initialized or firebase_admin is None:
        return

    if not firebase_admin._apps:
        svc_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        try:
            if svc_json:
                # JSON string in env -> dict -> Certificate
                cred = fb_credentials.Certificate(json.loads(svc_json))
            else:
                # Fall back to ADC if you have it configured locally
                cred = fb_credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred)
        except Exception as e:
            # Don't crash the whole API if Firebase can't init; just log.
            print("Firebase admin init error:", repr(e))
            return

    _initialized = True


def ensure_firebase() -> None:
    """
    Public helper for other modules to ensure Firebase is ready
    before calling messaging/auth.
    """
    _ensure_init()


# ---------- AUTH HELPERS ----------


def verify_id_token(id_token: str) -> Optional[dict]:
    """
    Returns decoded claims or None if invalid/unavailable.
    Keeps API alive even if Firebase isn't configured yet.
    """
    try:
        _ensure_init()
        if fb_auth is None:
            print("Firebase admin not initialised (auth is None)")
            return None
        return fb_auth.verify_id_token(id_token)
    except Exception as e:
        print("verify_id_token() error:", repr(e))
        return None


def get_current_user(authorization_header: Optional[str]) -> Optional[dict]:
    """
    SOFT helper for places like tipster detail:

      - Accepts raw "Authorization: Bearer <token>" header
      - Returns decoded claims dict on success
      - Returns None on any failure (no header, bad format, invalid token)

    This should NEVER raise; callers treat None as "viewer is anonymous".
    """
    if not authorization_header:
        return None

    parts = authorization_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None

    token = parts[1]

    try:
        return verify_id_token(token)
    except Exception:
        # Any verification error => treat as unauthenticated viewer
        return None


# ---------- FCM / WEB PUSH HELPERS ----------


def send_web_push_to_tokens(
    tokens: Sequence[str],
    title: str,
    body: str,
    data: Optional[dict] = None,
):
    """
    Send a web push notification via Firebase Cloud Messaging
    to a set of FCM tokens.

    - tokens: list of FCM registration tokens (strings)
    - title/body: notification content
    - data: optional key/value payload (must be strings)
    """
    if not tokens:
        return None

    _ensure_init()
    if fb_messaging is None:
        print("Firebase admin messaging not available")
        return None

    # FCM data payload must be string-to-string
    safe_data = {str(k): str(v) for k, v in (data or {}).items()}

    message = fb_messaging.MulticastMessage(
        notification=fb_messaging.Notification(
            title=title,
            body=body,
        ),
        data=safe_data,
        tokens=list(tokens),
    )

    try:
        resp = fb_messaging.send_multicast(message)
        # Optional: log failures so we can later prune dead tokens
        if resp.failure_count:
            print(
                f"FCM multicast: {resp.success_count} success, "
                f"{resp.failure_count} failures"
            )
        return resp
    except Exception as e:
        print("send_web_push_to_tokens() error:", repr(e))
        return None