from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
import requests

from ..models import (
    FixtureDetailCache,
    FixtureStatsCache,
    FixtureEventsCache,
)
from ..services.apifootball import BASE_URL, HEADERS

FRESH_FOR = timedelta(hours=12)


def _now():
    # timezone-aware UTC so it plays nicely with timestamptz columns
    return datetime.now(timezone.utc)


# -------------------------
#  /fixtures?id=ID
# -------------------------
def get_fixture_detail_cached(db: Session, provider_fixture_id: int) -> dict:
    fid = int(provider_fixture_id or 0)
    if fid <= 0:
        return {}

    row = db.query(FixtureDetailCache).filter_by(fixture_provider_id=fid).one_or_none()

    # If fresh → return
    if row and (_now() - row.updated_at) < FRESH_FOR:
        return row.payload or {}

    # Hit API-Football
    try:
        r = requests.get(
            f"{BASE_URL}/fixtures",
            headers=HEADERS,
            params={"id": fid},
            timeout=20,
        )
        data = r.json() or {}
    except Exception:
        data = {}

    # If API fails, fall back to stale cache if present
    if not data.get("response"):
        return row.payload if row else {}

    # Upsert
    if row is None:
        row = FixtureDetailCache(
            fixture_provider_id=fid,
            payload=data,
            updated_at=_now(),
        )
        db.add(row)
    else:
        row.payload = data
        row.updated_at = _now()

    db.commit()
    return data


# -------------------------
#  /fixtures/statistics?fixture=ID
# -------------------------
def get_fixture_stats_cached(db: Session, provider_fixture_id: int) -> dict:
    fid = int(provider_fixture_id or 0)
    if fid <= 0:
        return {}

    row = db.query(FixtureStatsCache).filter_by(fixture_provider_id=fid).one_or_none()

    if row and (_now() - row.updated_at) < FRESH_FOR:
        return row.payload or {}

    try:
        r = requests.get(
            f"{BASE_URL}/fixtures/statistics",
            headers=HEADERS,
            params={"fixture": fid},
            timeout=20,
        )
        data = r.json() or {}
    except Exception:
        data = {}

    if not data.get("response"):
        return row.payload if row else {}

    if row is None:
        row = FixtureStatsCache(
            fixture_provider_id=fid,
            payload=data,
            updated_at=_now(),
        )
        db.add(row)
    else:
        row.payload = data
        row.updated_at = _now()

    db.commit()
    return data


# -------------------------
#  /fixtures/events?fixture=ID
# -------------------------
def get_fixture_events_cached(db: Session, provider_fixture_id: int) -> dict:
    fid = int(provider_fixture_id or 0)
    if fid <= 0:
        return {}

    row = db.query(FixtureEventsCache).filter_by(fixture_provider_id=fid).one_or_none()

    # Short TTL recommended for events (5 mins)
    short_ttl = timedelta(minutes=5)
    if row and (_now() - row.updated_at) < short_ttl:
        return row.payload or {}

    try:
        r = requests.get(
            f"{BASE_URL}/fixtures/events",
            headers=HEADERS,
            params={"fixture": fid},
            timeout=20,
        )
        data = r.json() or {}
    except Exception:
        data = {}

    if not data.get("response"):
        return row.payload if row else {}

    if row is None:
        row = FixtureEventsCache(
            fixture_provider_id=fid,
            payload=data,
            updated_at=_now(),
        )
        db.add(row)
    else:
        row.payload = data
        row.updated_at = _now()

    db.commit()
    return data