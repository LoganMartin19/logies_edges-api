# api/app/services/firebase.py
import os
import json
from typing import Optional

try:
    import firebase_admin
    from firebase_admin import auth, credentials
except Exception:  # package not installed locally
    firebase_admin = None
    auth = None
    credentials = None

_initialized = False


def _ensure_init():
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
        if svc_json:
            # JSON string in env -> dict -> Certificate
            cred = credentials.Certificate(json.loads(svc_json))
            firebase_admin.initialize_app(cred)
        else:
            # Fall back to default credentials (or error if none)
            firebase_admin.initialize_app()

    _initialized = True


def verify_id_token(id_token: str) -> dict:
    """
    STRICT verify:

    - Ensures Firebase Admin is initialised
    - Verifies the token
    - Returns decoded claims (dict)
    - Raises on failure

    This is what auth_firebase.get_current_user expects: an exception
    when the token is invalid / missing / Firebase not configured.
    """
    _ensure_init()

    if auth is None:
        raise RuntimeError("Firebase Admin SDK not configured")

    claims = auth.verify_id_token(id_token)
    if not claims:
        # Defensive: if verify_id_token returned falsy
        raise ValueError("Invalid Firebase token")

    return claims


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