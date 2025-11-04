# api/app/services/firebase.py
import json, os
import firebase_admin
from firebase_admin import credentials, auth as fb_auth
from functools import lru_cache

@lru_cache(maxsize=1)
def _init():
    if not firebase_admin._apps:
        raw = os.getenv("FIREBASE_CREDENTIALS_JSON")
        if raw:
            cred = credentials.Certificate(json.loads(raw))
        else:
            path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if not path:
                raise RuntimeError("Provide FIREBASE_CREDENTIALS_JSON or GOOGLE_APPLICATION_CREDENTIALS")
            cred = credentials.Certificate(path)
        firebase_admin.initialize_app(cred)

def verify_id_token(id_token: str) -> dict:
    _init()
    # you can add check_revoked=True later
    return fb_auth.verify_id_token(id_token, clock_skew_seconds=60)