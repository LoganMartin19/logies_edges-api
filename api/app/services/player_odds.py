# api/app/services/player_odds.py
from sqlalchemy.orm import Session
from ..models import PlayerOdds, Fixture
from ..services.apifootball import _get, BASE_URL
from datetime import datetime

def fetch_player_odds_from_api(fixture_provider_id: int) -> list[dict]:
    url = f"{BASE_URL}/odds"
    params = {"fixture": fixture_provider_id, "type": "player"}
    res = _get(url, params) or []
    print(f"[DEBUG] Raw player odds response for fixture {fixture_provider_id}: {res}")
    return res

def ingest_player_odds(db: Session, fixture: Fixture) -> int:
    """
    Fetch player odds for a fixture and insert/update in DB.
    Returns count of odds rows upserted.
    """
    if not fixture.provider_fixture_id:
        return 0

    provider_id = int(fixture.provider_fixture_id)
    offers = fetch_player_odds_from_api(provider_id)

    count = 0
    for market_block in offers:
        bookmaker = (market_block.get("bookmaker") or {}).get("name")
        bets = market_block.get("bets") or []

        for bet in bets:
            market = bet.get("name") or ""
            line = float(bet.get("line") or 0.0)
            outcomes = bet.get("values") or []

            for outcome in outcomes:
                player_id = outcome.get("id") or outcome.get("player_id")
                player_name = outcome.get("value") or outcome.get("player_name")
                price = float(outcome.get("odd") or 0.0)

                if not (player_id and player_name and price > 0):
                    continue

                row = PlayerOdds(
                    fixture_id=fixture.id,
                    player_id=int(player_id),
                    player_name=player_name,
                    market=market.lower(),
                    line=line,
                    bookmaker=bookmaker,
                    price=price,
                    last_seen=datetime.utcnow(),
                )
                db.add(row)
                count += 1

    db.commit()
    print(f"[DEBUG] Ingested {count} player odds for fixture_id={fixture.id}")  # ✅ add debug
    return count