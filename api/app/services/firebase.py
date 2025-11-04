import os, json
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
    global _initialized
    if _initialized or firebase_admin is None:
        return
    if not firebase_admin._apps:
        svc_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        if svc_json:
            cred = credentials.Certificate(json.loads(svc_json))
            firebase_admin.initialize_app(cred)
        else:
            # fall back to default creds; or skip init
            firebase_admin.initialize_app()
    _initialized = True

def verify_id_token(id_token: str) -> Optional[dict]:
    """
    Returns decoded claims or None if invalid/unavailable.
    Keeps API alive even if Firebase isn't configured yet.
    """
    try:
        _ensure_init()
        if auth is None:
            return None
        return auth.verify_id_token(id_token)
    except Exception:
        return None

def get_current_user(authorization_header: Optional[str]) -> Optional[dict]:
    """
    Helper if you pass `Authorization: Bearer <token>`
    """
    if not authorization_header:
        return None
    parts = authorization_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return verify_id_token(parts[1])
    return None