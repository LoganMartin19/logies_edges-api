# api/app/auth_firebase.py
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from .services.firebase import verify_id_token

bearer = HTTPBearer(auto_error=False)

def get_current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer)):
    if not creds or not creds.scheme.lower().startswith("bearer"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token")
    try:
        return verify_id_token(creds.credentials)  # contains uid, email, name, picture...
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Firebase token")