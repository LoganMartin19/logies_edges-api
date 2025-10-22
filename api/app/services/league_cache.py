# api/app/services/league_cache.py
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from ..models import LeagueTableCache
from ..services.apifootball import get_league_table  # your existing API fetcher

def get_or_update_league_table(db: Session, league: str, refresh_hours: int = 6):
    row = db.query(LeagueTableCache).filter(LeagueTableCache.league == league).first()
    now = datetime.utcnow()

    if row and (now - row.updated_at).total_seconds() < refresh_hours * 3600:
        return row.table_json

    # fetch fresh
    table = get_league_table(league)
    if not row:
        row = LeagueTableCache(league=league, table_json=table, updated_at=now)
        db.add(row)
    else:
        row.table_json = table
        row.updated_at = now
    db.commit()
    return table