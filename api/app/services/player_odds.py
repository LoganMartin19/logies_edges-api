# api/app/services/player_odds.py
from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ..models import PlayerOdds, Fixture
from ..services.apifootball import _get, BASE_URL


_LINE_RE = re.compile(r"(-?\d+(?:\.\d+)?)")  # 4.5 in "Over 4.5" or "-0.5" etc

def _safe_float(x, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default

def _safe_int(x, default: Optional[int] = None) -> Optional[int]:
    if x is None:
        return default
    try:
        return int(x)
    except Exception:
        return default

def _extract_line_from_value(value_str: str) -> Optional[float]:
    if not value_str:
        return None
    m = _LINE_RE.search(value_str)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def _normalize_market_label(label: str) -> str:
    s = (label or "").strip().lower()

    # IMPORTANT: check "shots on target" before "shot"
    if "shot on target" in s or "sot" in s:
        return "shots_on_target"
    if "shot" in s:
        return "shots"
    if "foul" in s:
        return "fouls"
    if "pass" in s:
        return "passes"
    if "tackle" in s:
        return "tackles"
    if "assist" in s:
        return "assists"
    if "anytime goal" in s:
        return "anytime_goalscorer"
    if "first goal" in s:
        return "first_goalscorer"
    if "last goal" in s:
        return "last_goalscorer"

    if "player singles" in s:
        return "player_singles"
    if "player doubles" in s:
        return "player_doubles"
    if "player triples" in s:
        return "player_triples"

    return s.replace(" ", "_")

def fetch_player_odds_from_api(fixture_provider_id: int) -> List[Dict[str, Any]]:
    url = f"{BASE_URL}/odds"
    params = {"fixture": fixture_provider_id, "type": "player"}
    payload = _get(url, params) or {}

    if isinstance(payload, list):
        return payload

    resp = payload.get("response") or []
    return resp if isinstance(resp, list) else []

def ingest_player_odds(db: Session, fixture: Fixture) -> int:
    if not fixture.provider_fixture_id:
        return 0

    provider_id = int(fixture.provider_fixture_id)
    response_rows = fetch_player_odds_from_api(provider_id)

    upserts = 0
    now = datetime.utcnow()

    for fx_block in response_rows:
        bookmakers = fx_block.get("bookmakers") or []
        if not isinstance(bookmakers, list):
            continue

        for bm in bookmakers:
            bookmaker = bm.get("name") or bm.get("bookmaker") or "unknown"
            bets = bm.get("bets") or []
            if not isinstance(bets, list):
                continue

            for bet in bets:
                bet_name = (bet.get("name") or "").strip()
                bet_id = bet.get("id")
                values = bet.get("values") or []
                if not isinstance(values, list):
                    continue

                market_norm = _normalize_market_label(bet_name)
                # ✅ keep bet_id so "Player Singles" can be decoded later
                market_key = f"{market_norm}:{bet_id}" if bet_id is not None else market_norm

                for outcome in values:
                    raw_value = (outcome.get("value") or outcome.get("player_name") or "").strip()
                    price = _safe_float(outcome.get("odd"), 0.0) or 0.0
                    player_id = _safe_int(outcome.get("id") or outcome.get("player_id"))

                    if not raw_value or price <= 0:
                        continue

                    # line:
                    # - sometimes in bet["line"]
                    # - otherwise often embedded in outcome["value"]
                    raw_line = bet.get("line")
                    line = _safe_float(raw_line)
                    if line is None:
                        line = _extract_line_from_value(raw_value)  # may remain None

                    # clean player name only when we *derived* line from the value string
                    player_name = raw_value
                    if raw_line is None and line is not None:
                        player_name_clean = _LINE_RE.sub("", raw_value, count=1).strip()
                        player_name_clean = re.sub(r"\s{2,}", " ", player_name_clean).strip()
                        if len(player_name_clean) >= 3:
                            player_name = player_name_clean

                    existing = (
                        db.query(PlayerOdds)
                        .filter(
                            PlayerOdds.fixture_id == fixture.id,
                            PlayerOdds.player_id == player_id,
                            PlayerOdds.player_name == player_name,
                            PlayerOdds.market == market_key,
                            PlayerOdds.line == line,
                            PlayerOdds.bookmaker == bookmaker,
                        )
                        .first()
                    )

                    if existing:
                        if float(existing.price) != float(price):
                            existing.price = float(price)
                        existing.last_seen = now
                        db.add(existing)
                        upserts += 1
                    else:
                        db.add(PlayerOdds(
                            fixture_id=fixture.id,
                            player_id=player_id,        # can be None
                            player_name=player_name,
                            market=market_key,
                            line=line,                  # keep None if unknown/not applicable
                            bookmaker=bookmaker,
                            price=float(price),
                            last_seen=now,
                        ))
                        upserts += 1

    db.commit()
    print(f"[player_odds] upserted={upserts} fixture_id={fixture.id} provider={fixture.provider_fixture_id}")
    return upserts