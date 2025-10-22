# routes/poll.py
from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from ..db import get_db
from ..models import PollVote
import hashlib

router = APIRouter()

@router.post("/poll/vote")
def vote_poll(fixture_id: int, choice: str, request: Request, db: Session = Depends(get_db)):
    ip = request.client.host
    ip_hash = hashlib.sha256(ip.encode()).hexdigest()

    # Prevent multiple votes from same IP+fixture
    existing = db.query(PollVote).filter_by(fixture_id=fixture_id, ip_hash=ip_hash).first()
    if existing:
        return {"status": "already_voted"}

    vote = PollVote(fixture_id=fixture_id, choice=choice, ip_hash=ip_hash)
    db.add(vote)
    db.commit()
    return {"status": "ok"}

@router.get("/poll/results")
def get_poll_results(fixture_id: int, db: Session = Depends(get_db)):
    votes = db.query(PollVote).filter(PollVote.fixture_id == fixture_id).all()

    total = len(votes)
    home = sum(1 for v in votes if v.choice == "home")
    draw = sum(1 for v in votes if v.choice == "draw")
    away = sum(1 for v in votes if v.choice == "away")

    results = {"home": 0, "draw": 0, "away": 0}
    if total:
        results = {
            "home": home / total * 100,
            "draw": draw / total * 100,
            "away": away / total * 100,
        }
    return {"results": results, "total": total}