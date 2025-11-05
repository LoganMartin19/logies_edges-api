# api/app/deps/current_user.py
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import User
from ..auth_firebase import get_current_user as get_fb_claims

def current_user(db: Session = Depends(get_db),
                 claims: dict = Depends(get_fb_claims)) -> User:
    if not claims:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthenticated")

    uid = claims.get("uid")
    if not uid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Firebase token")

    email = claims.get("email")
    name = claims.get("name")
    avatar = claims.get("picture")

    # Upsert user
    user = db.query(User).filter(User.firebase_uid == uid).first()
    if not user:
        user = User(firebase_uid=uid, email=email, display_name=name, avatar_url=avatar)
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        changed = False
        if email and user.email != email:
            user.email = email; changed = True
        if name and user.display_name != name:
            user.display_name = name; changed = True
        if avatar and user.avatar_url != avatar:
            user.avatar_url = avatar; changed = True
        if changed:
            db.commit()
            db.refresh(user)

    return user