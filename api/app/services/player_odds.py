# api/app/routes/player_odds.py
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Fixture, PlayerOdds
from ..services.apifootball import get_fixture  # fixture details cache
from ..services.player_cache import (
    get_team_season_players_cached,  # cached season players
)

router = APIRouter(prefix="/api/player-odds", tags=["player-odds"])


# -----------------------------
# Normalization / parsing utils
# -----------------------------

_space_re = re.compile(r"\s+")
_dash_line_re = re.compile(r"\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*$")


def _norm_name(s: str) -> str:
    """
    Normalize names to match reliably across:
      - "A. Bastoni"
      - "Alessandro Bastoni"
      - "Alessandro Bastoni - 1.5"
      - weird unicode dashes
    """
    if not s:
        return ""
    s = str(s)
    # unify dashes
    s = s.replace("–", "-").replace("—", "-")
    # keep word chars, spaces, dots, hyphens, apostrophes
    s = re.sub(r"[^\w\s\.\-']", " ", s, flags=re.UNICODE)
    s = s.lower().strip()
    s = _space_re.sub(" ", s)
    return s


def _strip_line_suffix(value_str: str) -> Tuple[str, Optional[float]]:
    """
    Given "Alessandro Bastoni - 1.5" -> ("Alessandro Bastoni", 1.5)
    If no suffix -> (value_str, None)
    """
    if not value_str:
        return "", None
    s = str(value_str).replace("–", "-").replace("—", "-").strip()
    m = _dash_line_re.search(s)
    if not m:
        return s, None
    line = None
    try:
        line = float(m.group(1))
    except Exception:
        line = None
    name_part = s[: m.start()].strip()
    return name_part, line


def _split_first_last(full: str) -> Tuple[Optional[str], Optional[str]]:
    """
    "A. Bastoni" => ("a", "bastoni")
    "Alessandro Bastoni" => ("alessandro", "bastoni")
    "Luis Henrique de Lima" => ("luis", "lima")  (best-effort: last token)
    """
    s = _norm_name(full)
    if not s:
        return None, None

    s = s.strip("-").strip()
    parts = [p for p in s.split(" ") if p]
    if len(parts) < 2:
        return (parts[0], None) if parts else (None, None)

    first = parts[0].replace(".", "")
    last = parts[-1].replace(".", "")
    return first or None, last or None


# -----------------------------------------
# Build fixture-scoped season player index
# -----------------------------------------


def _build_fixture_player_index(db: Session, fixture_id: int) -> Dict[str, int]:
    """
    Uses the SAME backend cache your UI relies on (season-players)
    to build aliases for name->player_id.

    Crucially, uses API-Football's player.firstname/lastname to build:
      - "a. bastoni" / "a bastoni"
      - "alessandro bastoni"
      - "bastoni" (only if unique in this fixture)

    This is what resolves odds strings like "Alessandro Bastoni - 1.5"
    even when the UI shows "A. Bastoni".
    """
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).one_or_none()
    if not fx or not fx.provider_fixture_id:
        raise HTTPException(status_code=404, detail="Fixture not found")

    pfx = int(fx.provider_fixture_id)
    fjson = get_fixture(pfx) or {}
    core = (fjson.get("response") or [None])[0] or {}
    lg = core.get("league") or {}
    season = int(lg.get("season") or 0)

    teams = core.get("teams") or {}
    home_id = int(((teams.get("home") or {}).get("id")) or 0)
    away_id = int(((teams.get("away") or {}).get("id")) or 0)

    if not (season and home_id and away_id):
        raise HTTPException(status_code=400, detail="Fixture missing season/team ids")

    home_rows = get_team_season_players_cached(db, home_id, season) or []
    away_rows = get_team_season_players_cached(db, away_id, season) or []

    alias_to_id: Dict[str, int] = {}

    def _add_alias(key: str, pid: int):
        k = _norm_name(key)
        if not k:
            return
        alias_to_id.setdefault(k, pid)

    def _process(rows: List[dict]):
        for row in rows:
            pl = (row.get("player") or {}) or {}
            pid = int(pl.get("id") or 0)
            if not pid:
                continue

            raw_name = (pl.get("name") or "").strip()  # often "A. Bastoni"
            first = (pl.get("firstname") or "").strip()
            last = (pl.get("lastname") or "").strip()

            # 1) always alias provider display name
            if raw_name:
                _add_alias(raw_name, pid)

            # 2) firstname/lastname → full + initial aliases
            if first and last:
                _add_alias(f"{first} {last}", pid)  # "Alessandro Bastoni"
                fi = first[:1]
                if fi:
                    _add_alias(f"{fi} {last}", pid)   # "A Bastoni"
                    _add_alias(f"{fi}. {last}", pid)  # "A. Bastoni"
                # marker for unique last-name only alias (handled later)
                _add_alias(f"__LAST__:{last}", pid)

            # 3) fallback: split raw_name if firstname/lastname missing
            if (not first or not last) and raw_name:
                f, l = _split_first_last(raw_name)
                if f and l:
                    _add_alias(f"{f} {l}", pid)
                    _add_alias(f"{f}. {l}", pid)
                    _add_alias(f"__LAST__:{l}", pid)

    _process(home_rows)
    _process(away_rows)

    # Build last-name-only aliases only if unique for this fixture
    last_name_to_ids: Dict[str, set] = {}
    for k, pid in list(alias_to_id.items()):
        # NOTE: _add_alias lowercases, so "__LAST__:" becomes "__last__:"
        if k.startswith("__last__:"):
            ln = _norm_name(k.split(":", 1)[1])
            if ln:
                last_name_to_ids.setdefault(ln, set()).add(pid)

    for ln, ids in last_name_to_ids.items():
        if len(ids) == 1:
            alias_to_id[ln] = list(ids)[0]

    # Remove internal markers
    for k in [k for k in list(alias_to_id.keys()) if k.startswith("__last__:")]:
        alias_to_id.pop(k, None)

    return alias_to_id


# -----------------------------
# DB upsert helper
# -----------------------------


def _upsert_player_odds(
    db: Session,
    *,
    fixture_id: int,
    player_id: Optional[int],
    player_name: str,
    market: str,
    line: Optional[float],
    bookmaker: str,
    price: float,
    last_seen: datetime,
):
    """
    Upsert by NAME-key (fixture_id, player_name, market, line, bookmaker)
    and upgrade player_id from NULL -> resolved id when available.

    If both exist (a legacy NULL-id row AND a resolved-id row), keep the resolved-id row.
    """

    # -------------------------
    # 1) Find row by NAME-key
    # -------------------------
    q_name = (
        db.query(PlayerOdds)
        .filter(PlayerOdds.fixture_id == fixture_id)
        .filter(PlayerOdds.player_name == player_name)
        .filter(PlayerOdds.market == market)
        .filter(PlayerOdds.bookmaker == bookmaker)
    )
    if line is None:
        q_name = q_name.filter(PlayerOdds.line.is_(None))
    else:
        q_name = q_name.filter(PlayerOdds.line == float(line))

    row_name = q_name.one_or_none()

    # -------------------------
    # 2) Find row by PID-key (only if player_id resolved)
    # -------------------------
    row_pid = None
    if player_id:
        q_pid = (
            db.query(PlayerOdds)
            .filter(PlayerOdds.fixture_id == fixture_id)
            .filter(PlayerOdds.player_id == int(player_id))
            .filter(PlayerOdds.market == market)
            .filter(PlayerOdds.bookmaker == bookmaker)
        )
        if line is None:
            q_pid = q_pid.filter(PlayerOdds.line.is_(None))
        else:
            q_pid = q_pid.filter(PlayerOdds.line == float(line))

        row_pid = q_pid.one_or_none()

    # -------------------------
    # 3) If we have BOTH rows, prefer the PID row, delete legacy NULL row
    # -------------------------
    if row_name and row_pid and row_name.id != row_pid.id:
        # keep pid-row; if pid-row name empty/odd, optionally update it
        row_pid.price = float(price)
        row_pid.last_seen = last_seen

        # if pid-row has no name but name-row does, you can copy it (optional)
        if (not row_pid.player_name) and row_name.player_name:
            row_pid.player_name = row_name.player_name

        db.delete(row_name)
        db.add(row_pid)
        return row_pid

    # -------------------------
    # 4) If name-row exists, update it + upgrade NULL->pid
    # -------------------------
    if row_name:
        row_name.price = float(price)
        row_name.last_seen = last_seen

        if (row_name.player_id is None or int(row_name.player_id or 0) == 0) and player_id:
            row_name.player_id = int(player_id)

        db.add(row_name)
        return row_name

    # -------------------------
    # 5) Else create new row
    # -------------------------
    row = PlayerOdds(
        fixture_id=fixture_id,
        player_id=int(player_id) if player_id else None,
        player_name=player_name,
        market=market,
        line=float(line) if line is not None else None,
        bookmaker=bookmaker,
        price=float(price),
        last_seen=last_seen,
    )
    db.add(row)
    return row

# -----------------------------
# Main endpoint (existing)
# -----------------------------


@router.get("/")
def get_player_odds(
    fixture_id: int,
    player: Optional[str] = None,
    include_meta: bool = False,
    db: Session = Depends(get_db),
):
    """
    Returns stored player odds for this fixture, optionally filtered by player name.
    """
    q = db.query(PlayerOdds).filter(PlayerOdds.fixture_id == fixture_id)
    if player:
        q = q.filter(PlayerOdds.player_name.ilike(f"%{player}%"))

    rows = q.order_by(PlayerOdds.market.asc(), PlayerOdds.line.asc().nullsfirst()).all()

    out = []
    for r in rows:
        out.append(
            {
                "player_id": r.player_id,
                "player_name": r.player_name,
                "market": r.market,
                "line": r.line,
                "bookmaker": r.bookmaker,
                "price": r.price,
                "last_seen": r.last_seen.isoformat() if r.last_seen else None,
            }
        )

    payload = {
        "fixture_id": fixture_id,
        "count": len(out),
        "rows": out,
        "markets": sorted(list({r["market"] for r in out})),
        "bookmakers": sorted(list({r["bookmaker"] for r in out})),
    }

    if include_meta:
        payload["meta"] = {"filter_player": player}

    return payload


# ---------------------------------------------------------
# Ingest/refresh route
# ---------------------------------------------------------


@router.post("/ingest")
def ingest_player_odds_for_fixture(
    fixture_id: int,
    db: Session = Depends(get_db),
):
    """
    Ingest player odds for a fixture.
      - builds alias map from season players (firstname/lastname aware)
      - resolves player_id from odds string
      - upserts with NULL->id upgrade
    """
    from ..services.player_odds_provider import (  # your service
        fetch_raw_player_odds_for_fixture,
    )

    alias_map = _build_fixture_player_index(db, fixture_id)
    raw = fetch_raw_player_odds_for_fixture(fixture_id) or {}

    # Support both shapes:
    #   provider: { data: { response: [...] } }
    #   wrapped:  { raw: { data: { response: [...] } } }
    resp = (
        (((raw.get("raw") or {}).get("data") or {}).get("response"))
        or (((raw.get("data") or {}).get("response")) or [])
    )
    if not isinstance(resp, list):
        resp = []

    now = datetime.now(timezone.utc)

    processed = 0
    for block in resp:
        for bm in (block.get("bookmakers") or []):
            bookmaker = bm.get("name") or bm.get("bookmaker") or "Unknown"
            for bet in (bm.get("bets") or []):
                bet_id = int(bet.get("id") or 0)
                bet_name = bet.get("name") or ""

                market = _map_bet_to_market(bet_id, bet_name)
                if not market:
                    continue

                for v in (bet.get("values") or []):
                    value_str = v.get("value") or ""
                    odd_str = v.get("odd") or v.get("price") or None
                    if not value_str or not odd_str:
                        continue

                    # parse line + player from "Name - 1.5"
                    name_part, line = _strip_line_suffix(value_str)

                    try:
                        price = float(odd_str)
                    except Exception:
                        continue

                    # resolve player id by alias
                    # resolve player id by alias
                    pid = None
                    key = _norm_name(name_part)

                    # 1) direct match (works if alias_map has full name)
                    pid = alias_map.get(key)

                    if not pid:
                        f, l = _split_first_last(name_part)  # from odds: "Alessandro Bastoni" -> ("alessandro","bastoni")

                        if l:
                            # 2) try "A Bastoni" / "A. Bastoni" (works when season roster stores initials)
                            if f:
                                fi = f[:1].lower()
                                for cand in (f"{fi} {l}", f"{fi}. {l}"):
                                    pid = alias_map.get(_norm_name(cand))
                                    if pid:
                                        break

                            # 3) try unique last name (only if unique in fixture, built in alias map)
                            if not pid:
                                pid = alias_map.get(_norm_name(l))

                    _upsert_player_odds(
                        db,
                        fixture_id=fixture_id,
                        player_id=pid,
                        player_name=name_part.strip(),
                        market=market,
                        line=line,
                        bookmaker=bookmaker,
                        price=price,
                        last_seen=now,
                    )
                    processed += 1

    db.commit()

    return {
        "fixture_id": fixture_id,
        "rows_processed": processed,
        "alias_map_size": len(alias_map),
    }


# -----------------------------
# Raw passthrough (optional helper)
# -----------------------------


@router.get("/raw")
def raw_player_odds(
    fixture_id: int,
    db: Session = Depends(get_db),
):
    """
    If you already have this route elsewhere, ignore.
    Keeping here as a handy debug passthrough.
    """
    from ..services.player_odds_provider import fetch_raw_player_odds_for_fixture

    raw = fetch_raw_player_odds_for_fixture(fixture_id) or {}
    return {"fixture_id": fixture_id, "raw": raw}


# -----------------------------
# Market mapping
# -----------------------------


def _map_bet_to_market(bet_id: int, bet_name: str) -> Optional[str]:
    """
    Map provider bet identifiers to your internal market slugs.
    API-Football bet ids seen:
      92 Anytime Goal Scorer
      93 First Goal Scorer
      94 Last Goal Scorer
      212 Player Assists
      213 Player Triples
      215 Player Singles
    """
    if bet_id == 92:
        return "anytime_goalscorer"
    if bet_id == 93:
        return "first_goalscorer"
    if bet_id == 94:
        return "last_goalscorer"
    if bet_id == 212:
        return "assists"
    if bet_id == 213:
        return "player_triples"
    if bet_id == 215:
        return "player_singles"

    n = (bet_name or "").lower()
    if "anytime" in n and "scorer" in n:
        return "anytime_goalscorer"
    if "first goal" in n:
        return "first_goalscorer"
    if "last goal" in n:
        return "last_goalscorer"
    if "assist" in n:
        return "assists"
    if "player triples" in n:
        return "player_triples"
    if "player singles" in n:
        return "player_singles"

    return None