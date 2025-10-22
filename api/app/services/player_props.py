# api/app/services/player_props.py
from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from ..models import PlayerOdds, Fixture
from .apifootball import fetch_odds_for_fixture

# Map provider market names to our canonical names, and capture whether they use lines
# (you can extend this as you discover more markets)
MARKET_ALIASES: Dict[str, str] = {
    "Shots on Target": "SOT",          # line (e.g., 1+, 2+, 1.5)
    "Shots": "SHOTS",                   # line
    "Yellow Cards": "YC",               # line (often 1+, sometimes 0.5)
    "Red Cards": "RC",                  # line
    "Fouls": "FOULS",                   # line
    # vendor quirks
    "Player Shots on Target": "SOT",
    "Player Shots": "SHOTS",
    "Player Fouls": "FOULS",
    "Player To Be Booked": "YC",
}

def _canon_market(name: str | None) -> Optional[str]:
    if not name:
        return None
    return MARKET_ALIASES.get(name.strip(), None)

def _parse_line(raw: Any) -> Optional[float]:
    """
    Try to normalize provider 'handicap'/'value'/'line' fields into a float.
    Providers may send '1.5', '2+', '1+', 1, None, etc.
    """
    if raw is None:
        return None
    s = str(raw).strip().replace("+", "")
    try:
        return float(s)
    except Exception:
        # Sometimes props are expressed as integers for "x+" (e.g., 2+)
        try:
            return float(int(s))
        except Exception:
            return None

def _extract_player_rows(odds_payload: List[dict]) -> List[Dict[str, Any]]:
    """
    API-Football odds fixture response -> list of normalized player prop rows.
    We expect payload like:
      [{ "bookmakers": [...{"bets":[{"name": "...", "values":[{"value": "1.5", "odd": "2.10", "player": {...}}]}]}] }]
    This function is defensive to different nested shapes.
    """
    rows: List[Dict[str, Any]] = []
    if not isinstance(odds_payload, list):
        return rows

    for offer in odds_payload:
        bms = offer.get("bookmakers") or []
        for bm in bms:
            bookmaker = bm.get("name") or bm.get("key") or "Unknown"
            bets = bm.get("bets") or []
            for bet in bets:
                raw_market = bet.get("name")
                market = _canon_market(raw_market)
                if not market:
                    continue  # skip markets we don't track yet

                values = bet.get("values") or []
                for v in values:
                    price = v.get("odd") or v.get("price")
                    try:
                        price_f = float(price)
                    except Exception:
                        continue

                    line = _parse_line(v.get("value") or v.get("handicap") or v.get("line"))

                    # API sometimes nests player info under 'player' or embeds in 'value'
                    pinfo = v.get("player") or {}
                    player_id = pinfo.get("id")
                    player_name = pinfo.get("name") or v.get("participant")

                    if not player_name:
                        # if player name truly missing, skip (props must be player-specific)
                        continue

                    rows.append({
                        "player_id": player_id,
                        "player_name": player_name,
                        "market": market,
                        "line": line,
                        "bookmaker": bookmaker,
                        "price": price_f,
                    })
    return rows

def ingest_player_odds_for_fixture(db: Session, fixture_id: int) -> int:
    """
    Fetch player props odds for a fixture from API-Football, normalize, and upsert into PlayerOdds.
    Strategy: replace-by-fixture (delete then insert) for freshness and simplicity.
    Returns number of rows inserted.
    """
    fx: Fixture | None = db.query(Fixture).filter(Fixture.id == fixture_id).one_or_none()
    if not fx or not fx.provider_fixture_id:
        return 0

    provider_fixture_id = int(fx.provider_fixture_id)
    payload = fetch_odds_for_fixture(provider_fixture_id)  # list[dict]
    rows = _extract_player_rows(payload)

    # wipe old
    db.query(PlayerOdds).filter(PlayerOdds.fixture_id == fixture_id).delete(synchronize_session=False)

    now = datetime.utcnow()
    to_insert: List[PlayerOdds] = []
    for r in rows:
        to_insert.append(PlayerOdds(
            fixture_id=fixture_id,
            player_id=r.get("player_id"),
            player_name=r["player_name"],
            market=r["market"],
            line=r.get("line"),
            bookmaker=r["bookmaker"],
            price=r["price"],
            last_seen=now,
        ))

    if to_insert:
        db.add_all(to_insert)
    db.commit()
    return len(to_insert)