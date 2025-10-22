# api/app/services/fixtures_gridiron.py
from __future__ import annotations

import re
from time import sleep
from typing import Iterable, Optional, Dict, List, Union
from datetime import date, datetime, timezone

from sqlalchemy.orm import Session

from ..models import Fixture, Odds
from .api_gridiron import fetch_fixtures as api_fetch_fixtures, fetch_odds_for_game

# ------------ small utils ------------

def _norm(s: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _kickoff_from_payload(p: dict) -> datetime:
    """
    Expected API shape (API-Sports Gridiron):
      p["game"]["date"] = {
        "date": "YYYY-MM-DD",
        "time": "HH:MM",
        "timezone": "UTC" or "+/-HH:MM",
        "timestamp": <unix seconds>  # sometimes present
      }
    Some feeds provide ISO at p["date"] or p["game"]["date"]["full"] — we handle fallbacks.
    """
    g = (p.get("game") or {})
    dd = g.get("date") or {}

    # 1) If a UNIX timestamp is present, trust it.
    ts = dd.get("timestamp")
    if isinstance(ts, (int, float)) and ts > 0:
        try:
            return datetime.fromtimestamp(int(ts), tz=timezone.utc)
        except Exception:
            pass

    # 2) Handle a direct ISO string
    if isinstance(dd, str):
        try:
            dt = datetime.fromisoformat(dd.replace("Z", "+00:00"))
            return _to_utc(dt)
        except Exception:
            pass

    # 3) Build from components
    d = dd.get("date") or dd.get("day") or ""
    t = dd.get("time") or "00:00"
    tz_raw = (dd.get("timezone") or "").strip().upper()

    # Accept "UTC" explicitly; otherwise only accept "+/-HH:MM" shapes
    if tz_raw == "UTC":
        tz_part = "+00:00"
    else:
        tz_part = tz_raw if (tz_raw.startswith(("+", "-")) and len(tz_raw) >= 3) else ""

    iso = f"{d}T{t}{tz_part}"
    try:
        dt = datetime.fromisoformat(iso)
    except Exception:
        # last resort: now (UTC)
        return datetime.now(timezone.utc)
    return _to_utc(dt)

# capture full decimals like 45.5, 45.25, etc.
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")

def _extract_line_from_val(v: dict) -> Optional[float]:
    for k in ("total", "points", "point", "line", "handicap"):
        if v.get(k) is not None:
            try:
                return float(str(v[k]).replace(",", "."))
            except Exception:
                pass
    txt = (v.get("name") or v.get("label") or v.get("value") or "")
    m = _NUM_RE.search(str(txt))
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

def _is_main_gridiron_total(x: float) -> bool:
    # CFB totals can be higher than NFL; keep a wider but sane window.
    return 30.0 <= x <= 80.0

def _pick_main_total_line(lines: List[float]) -> Optional[float]:
    sane = [l for l in lines if _is_main_gridiron_total(l)]
    if not sane:
        return None
    sane.sort()
    mid = len(sane) // 2
    return sane[mid] if len(sane) % 2 == 1 else (sane[mid - 1] + sane[mid]) / 2.0

# ---- optional bookmaker filters (mirror soccer style) ----
BOOK_BLACKLIST = {"1xbet"}
BOOK_WHITELIST = {
    "bet365", "williamhill", "betfair", "unibet",
    "skybet", "ladbrokes", "coral", "betvictor", "pinnacle", "betmgm", "draftkings", "fanduel"
}

def _is_blacklisted_book(name: Optional[str]) -> bool:
    return _norm(name) in BOOK_BLACKLIST

def _is_whitelisted_book(name: Optional[str]) -> bool:
    n = _norm(name)
    # keep open by default; switch to `return (n in BOOK_WHITELIST)` when you want to lock down
    return True  # or: (n in BOOK_WHITELIST)

# ------------ fixtures upsert ------------

def upsert_fixture_from_payload(db: Session, p: dict) -> Fixture:
    gid = str(p.get("id") or (p.get("game") or {}).get("id") or "")
    lg = (p.get("league") or {})
    tm = (p.get("teams") or {})

    home_name = (tm.get("home") or {}).get("name") or "Home"
    away_name = (tm.get("away") or {}).get("name") or "Away"
    kickoff = _kickoff_from_payload(p)

    comp = (lg.get("name") or "NFL/CFB")
    league_name_lc = comp.lower()
    # infer sport
    if any(k in league_name_lc for k in ("college", "cfb", "ncaa")):
        sport = "cfb"
    else:
        sport = "nfl"

    f = db.query(Fixture).filter(Fixture.provider_fixture_id == gid).one_or_none()
    if f:
        f.comp = comp
        f.home_team = home_name
        f.away_team = away_name
        f.kickoff_utc = kickoff
        f.sport = sport
        # store provider IDs if model has them
        if hasattr(Fixture, "provider_league_id"):
            try:
                f.provider_league_id = int(lg.get("id")) if lg.get("id") is not None else None
            except Exception:
                f.provider_league_id = None
        if hasattr(Fixture, "provider_home_team_id"):
            try:
                f.provider_home_team_id = int((tm.get("home") or {}).get("id") or 0) or None
            except Exception:
                f.provider_home_team_id = None
        if hasattr(Fixture, "provider_away_team_id"):
            try:
                f.provider_away_team_id = int((tm.get("away") or {}).get("id") or 0) or None
            except Exception:
                f.provider_away_team_id = None
        db.flush()
        return f

    f = Fixture(
        provider_fixture_id=gid,
        comp=comp,
        home_team=home_name,
        away_team=away_name,
        kickoff_utc=kickoff,
        sport=sport,
    )
    if hasattr(Fixture, "provider_league_id"):
        try:
            f.provider_league_id = int(lg.get("id")) if lg.get("id") is not None else None
        except Exception:
            pass
    if hasattr(Fixture, "provider_home_team_id"):
        try:
            f.provider_home_team_id = int((tm.get("home") or {}).get("id") or 0) or None
        except Exception:
            pass
    if hasattr(Fixture, "provider_away_team_id"):
        try:
            f.provider_away_team_id = int((tm.get("away") or {}).get("id") or 0) or None
        except Exception:
            pass
    db.add(f)
    db.flush()
    return f

# ------------ odds upsert ------------

def _upsert_odds(db: Session, fixture_id: int, bookmaker: str, market: str, price: float):
    if not (price and 1.01 < price < 51.0):
        return
    row = (
        db.query(Odds)
        .filter(Odds.fixture_id == fixture_id, Odds.bookmaker == bookmaker, Odds.market == market)
        .one_or_none()
    )
    now = datetime.now(timezone.utc)
    if row:
        row.price = price
        row.last_seen = now
    else:
        db.add(Odds(fixture_id=fixture_id, bookmaker=bookmaker, market=market, price=price, last_seen=now))
        db.flush()

# ------------ market parsing ------------

def _parse_and_write_markets(
    db: Session,
    fixture_id: int,
    odds_rows: List[dict],
    *,
    prefer_book: Optional[str] = None,
) -> int:
    """
    Normalize NFL/CFB markets per bookmaker:
      • Skip any 3-way markets.
      • Moneyline → HOME_WIN / AWAY_WIN.
      • Spread/Handicap → SPREAD_HOME / SPREAD_AWAY (odds only).
      • Totals → choose ONE 'main' line per book (median in 30–80), write O{line}/U{line}.
      • Works with both shapes: a flat list of bookmaker rows OR a wrapper with 'bookmakers'.
    """
    if not odds_rows:
        return 0

    written = 0
    preferred = _norm(prefer_book) if prefer_book else ""

    def iter_books(rows: List[dict]):
        """Yield bookmaker dicts regardless of shape (flat row vs wrapper)."""
        for row in rows or []:
            # shape A: row itself is a bookmaker record (common for API-Sports odds)
            if row.get("bets") or row.get("markets"):
                yield row
            # shape B: wrapper row containing a 'bookmakers'/'books' list
            for b in (row.get("bookmakers") or row.get("books") or []):
                yield b

    def handle_book(b: dict):
        nonlocal written
        bname_raw = (b.get("name") or b.get("bookmaker") or "")
        if _is_blacklisted_book(bname_raw) or not _is_whitelisted_book(bname_raw):
            return
        bname = _norm(bname_raw)

        bets = b.get("bets") or b.get("markets") or []

        totals_by_line: Dict[float, Dict[str, float]] = {}

        for mk in bets:
            name_raw = (mk.get("name") or mk.get("market") or "")
            name = _norm(name_raw)
            vals = mk.get("values") or mk.get("outcomes") or []

            # --- 3-way skip ---
            if name in ("3wayresult", "3way", "1x2", "matchresult3way"):
                continue

            # --- Moneyline ---
            if name in ("homeaway", "moneyline", "winner", "ml", "money", "matchwinner"):
                home = next(
                    (v for v in vals if _norm(v.get("name") or v.get("label") or v.get("value")) in ("home", "1", "h", "yes")),
                    None
                )
                away = next(
                    (v for v in vals if _norm(v.get("name") or v.get("label") or v.get("value")) in ("away", "2", "a", "no")),
                    None
                )
                if home and home.get("odd"):
                    try:
                        _upsert_odds(db, fixture_id, bname, "HOME_WIN", float(home["odd"]))
                        written += 1
                    except Exception:
                        pass
                if away and away.get("odd"):
                    try:
                        _upsert_odds(db, fixture_id, bname, "AWAY_WIN", float(away["odd"]))
                        written += 1
                    except Exception:
                        pass

            # --- Spreads (odds only) ---
            if ("spread" in name) or ("handicap" in name) or ("line" in name and "total" not in name):
                h = next((v for v in vals if _norm(v.get("name") or v.get("label") or v.get("value")) in ("home", "1", "h")), None)
                a = next((v for v in vals if _norm(v.get("name") or v.get("label") or v.get("value")) in ("away", "2", "a")), None)
                if h and h.get("odd"):
                    try:
                        _upsert_odds(db, fixture_id, bname, "SPREAD_HOME", float(h["odd"]))
                        written += 1
                    except Exception:
                        pass
                if a and a.get("odd"):
                    try:
                        _upsert_odds(db, fixture_id, bname, "SPREAD_AWAY", float(a["odd"]))
                        written += 1
                    except Exception:
                        pass

            # --- Totals ---
            if ("total" in name) or ("overunder" in name) or ("points" in name) or ("over" in name and "under" in name):
                for v in vals:
                    lab = _norm(v.get("name") or v.get("label") or v.get("value"))
                    ln = _extract_line_from_val(v)
                    if ln is None:
                        continue
                    side = "O" if lab.startswith("over") else ("U" if lab.startswith("under") else None)
                    if side is None:
                        continue
                    try:
                        odd = float(v["odd"])
                    except Exception:
                        continue
                    d = totals_by_line.setdefault(ln, {})
                    d[side] = odd

        # write one "main" total line per book
        if totals_by_line:
            lines = list(totals_by_line.keys())
            main_ln = _pick_main_total_line(lines)
            if main_ln is not None and main_ln in totals_by_line:
                pair = totals_by_line[main_ln]
                o_price = pair.get("O")
                u_price = pair.get("U")
                if o_price and u_price and o_price > 1.01 and u_price > 1.01:
                    tag = f"{main_ln:.1f}".rstrip("0").rstrip(".")
                    _upsert_odds(db, fixture_id, bname, f"O{tag}", float(o_price)); written += 1
                    _upsert_odds(db, fixture_id, bname, f"U{tag}", float(u_price)); written += 1

    # Scan preferred book(s) first (if provided), then others
    if preferred:
        for b in iter_books(odds_rows):
            if _norm(b.get("name") or b.get("bookmaker") or "") == preferred:
                handle_book(b)
        for b in iter_books(odds_rows):
            if _norm(b.get("name") or b.get("bookmaker") or "") != preferred:
                handle_book(b)
    else:
        for b in iter_books(odds_rows):
            handle_book(b)

    return written

# ------------ public ingest ------------

def _normalize_leagues_arg(leagues: Union[str, Iterable[str], None]) -> List[str]:
    """
    Turn 'NCAA' or 'NFL,NCAA' or ['NFL','NCAA'] into ['NFL','NCAA'].
    """
    if leagues is None:
        return []
    if isinstance(leagues, str):
        return [s.strip() for s in leagues.split(",") if s.strip()]
    try:
        return [str(s).strip() for s in leagues if str(s).strip()]
    except TypeError:
        return [str(leagues).strip()]

def ingest_gridiron_and_odds(
    db: Session,
    day: date,
    leagues: Union[str, Iterable[str]],
    prefer_book: Optional[str] = None,
    max_games: Optional[int] = None,
    odds_delay_sec: float = 0.35,
    max_fixtures: Optional[int] = None,      # kept for backward compatibility
    require_uk_books_for_cfb: bool = False,  # placeholder (not enforced yet)
) -> dict:
    if max_games is None:
        max_games = max_fixtures

    leagues_norm = _normalize_leagues_arg(leagues)

    # DO NOT wrap with list(); pass through — the API layer already handles lists/strings.
    raw = api_fetch_fixtures(day, leagues_norm) or []
    if isinstance(max_games, int) and max_games > 0:
        raw = raw[:max_games]

    fixtures_written = 0
    odds_written = 0

    for payload in raw:
        f = upsert_fixture_from_payload(db, payload)
        fixtures_written += 1

        try:
            gid = int(f.provider_fixture_id)
        except Exception:
            continue

        try:
            odds_rows = fetch_odds_for_game(gid) or []
        except Exception:
            odds_rows = []

        odds_written += _parse_and_write_markets(
            db,
            f.id,
            odds_rows,
            prefer_book=prefer_book,
        )

        if odds_delay_sec:
            sleep(odds_delay_sec)

    db.commit()
    return {
        "date": str(day),
        "leagues": leagues_norm,
        "fixtures_upserted": fixtures_written,
        "odds_rows_written": odds_written,
    }