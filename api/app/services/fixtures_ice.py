# api/app/services/fixtures_ice.py
from __future__ import annotations
import re
from time import sleep
from typing import Iterable, Optional, Dict, List
from datetime import date, datetime, timezone
from sqlalchemy.orm import Session
from ..models import Fixture, Odds
from .api_ice import fetch_fixtures as api_fetch_fixtures, fetch_odds_for_game

def _norm(s) -> str:
    """Normalize any value (string/int) to a lowercase alphanumeric string."""
    if s is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def _to_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)

def _kickoff_from_payload(p: dict) -> datetime:
    g = (p.get("game") or {})
    dd = (g.get("date") or {})
    ts = dd.get("timestamp")
    if isinstance(ts, (int, float)) and ts > 0:
        try: return datetime.fromtimestamp(int(ts), tz=timezone.utc)
        except: pass
    d = dd.get("date") or ""
    t = dd.get("time") or "00:00"
    tz = (dd.get("timezone") or "").strip().upper()
    tz_part = "+00:00" if tz == "UTC" else (tz if (tz.startswith(("+","-")) and len(tz)>=3) else "")
    try:  return _to_utc(datetime.fromisoformat(f"{d}T{t}{tz_part}"))
    except: return datetime.now(timezone.utc)

BOOK_BLACKLIST = {"1xbet"}
def _is_blacklisted_book(name: Optional[str]) -> bool:
    return _norm(name) in BOOK_BLACKLIST
def _is_whitelisted_book(name: Optional[str]) -> bool:
    return True

def upsert_fixture_from_payload(db: Session, p: dict) -> Fixture:
    gid = str(p.get("id") or (p.get("game") or {}).get("id") or "")
    lg = (p.get("league") or {})
    tm = (p.get("teams") or {})
    home_name = (tm.get("home") or {}).get("name") or "Home"
    away_name = (tm.get("away") or {}).get("name") or "Away"
    kickoff = _kickoff_from_payload(p)
    comp = (lg.get("name") or "NHL")
    sport = "nhl"

    f = db.query(Fixture).filter(Fixture.provider_fixture_id == gid).one_or_none()
    if f:
        f.comp = comp; f.home_team = home_name; f.away_team = away_name; f.kickoff_utc = kickoff; f.sport = sport
        db.flush(); return f

    f = Fixture(provider_fixture_id=gid, comp=comp, home_team=home_name, away_team=away_name, kickoff_utc=kickoff, sport=sport)
    db.add(f); db.flush()
    return f

def _upsert_odds(db: Session, fixture_id: int, bookmaker: str, market: str, price: float):
    if not (price and 1.01 < price < 51.0): return
    row = db.query(Odds).filter(Odds.fixture_id==fixture_id, Odds.bookmaker==bookmaker, Odds.market==market).one_or_none()
    now = datetime.now(timezone.utc)
    if row: row.price = price; row.last_seen = now
    else:   db.add(Odds(fixture_id=fixture_id, bookmaker=bookmaker, market=market, price=price, last_seen=now)); db.flush()

def _parse_and_write_markets(db: Session, fixture_id: int, odds_rows: List[dict], *, prefer_book: Optional[str]=None) -> int:
    if not odds_rows: return 0
    written = 0; preferred = _norm(prefer_book) if prefer_book else ""
    def iter_books(rows: List[dict]):
        for row in rows or []:
            if row.get("bets") or row.get("markets"): yield row
            for b in (row.get("bookmakers") or row.get("books") or []): yield b
    def handle_book(b: dict):
        nonlocal written
        bname_raw = (b.get("name") or b.get("bookmaker") or "")
        if _is_blacklisted_book(bname_raw) or not _is_whitelisted_book(bname_raw): return
        bname = _norm(bname_raw)
        bets = b.get("bets") or b.get("markets") or []
        totals_by_line: Dict[float, Dict[str,float]] = {}
        for mk in bets:
            name = _norm(mk.get("name") or mk.get("market") or "")
            vals = mk.get("values") or mk.get("outcomes") or []

            # Moneyline (2-way) — HOME_WIN / AWAY_WIN
            if name in ("moneyline","ml","matchwinner","homeaway","winner"):
                home = next((v for v in vals if _norm(v.get("name") or v.get("label") or v.get("value")) in ("home","1","h","yes")), None)
                away = next((v for v in vals if _norm(v.get("name") or v.get("label") or v.get("value")) in ("away","2","a","no")), None)
                if home and home.get("odd"):
                    try: _upsert_odds(db, fixture_id, bname, "HOME_WIN", float(home["odd"])); written += 1
                    except: pass
                if away and away.get("odd"):
                    try: _upsert_odds(db, fixture_id, bname, "AWAY_WIN", float(away["odd"])); written += 1
                    except: pass

            # 3-way (Regulation) — ignore or map if you want later (DRAW)
            if name in ("3wayresult","1x2","regulationtime","matchresult3way"):
                # optional: map to HOME_WIN/DRAW/AWAY_WIN_reg if you later model it
                continue

            # Puckline spreads (±1.5 typically) — store odds only (no line keying)
            if ("puck" in name) or ("spread" in name) or ("handicap" in name and "total" not in name):
                h = next((v for v in vals if "home" in _norm(v.get("name") or v.get("label") or v.get("value"))), None)
                a = next((v for v in vals if "away" in _norm(v.get("name") or v.get("label") or v.get("value"))), None)
                if h and h.get("odd"):
                    try: _upsert_odds(db, fixture_id, bname, "SPREAD_HOME", float(h["odd"])); written += 1
                    except: pass
                if a and a.get("odd"):
                    try: _upsert_odds(db, fixture_id, bname, "SPREAD_AWAY", float(a["odd"])); written += 1
                    except: pass

            # Totals — build O{line}/U{line}, typical NHL window ~4.5–7.5
            if ("total" in name) or ("overunder" in name) or ("goals" in name) or ("over" in name and "under" in name):
                for v in vals:
                    lab = _norm(v.get("name") or v.get("label") or v.get("value"))
                    line = None
                    for k in ("total","points","point","line","handicap"):
                        if v.get(k) is not None:
                            try: line = float(str(v[k]).replace(",", ".")); break
                            except: pass
                    if line is None:  # try regex from label
                        import re
                        m = re.search(r"(\d+(?:\.\d+)?)", str(v.get("name") or v.get("label") or v.get("value") or ""))
                        if m:
                            try: line = float(m.group(1))
                            except: pass
                    if line is None: continue
                    side = "O" if lab.startswith("over") else ("U" if lab.startswith("under") else None)
                    if side is None or not v.get("odd"): continue
                    d = totals_by_line.setdefault(line, {}); d[side] = float(v["odd"])
        # pick a median-ish line
        if totals_by_line:
            lines = sorted([ln for ln in totals_by_line.keys() if 3.5 <= ln <= 9.5])
            if lines:
                mid = lines[len(lines)//2]
                pair = totals_by_line.get(mid) or {}
                o, u = pair.get("O"), pair.get("U")
                if o and u and o>1.01 and u>1.01:
                    tag = f"{mid:.1f}".rstrip("0").rstrip(".")
                    _upsert_odds(db, fixture_id, bname, f"O{tag}", o); written += 1
                    _upsert_odds(db, fixture_id, bname, f"U{tag}", u); written += 1

    if preferred:
        for b in iter_books(odds_rows):
            if _norm(b.get("name") or b.get("bookmaker") or "") == preferred: handle_book(b)
        for b in iter_books(odds_rows):
            if _norm(b.get("name") or b.get("bookmaker") or "") != preferred: handle_book(b)
    else:
        for b in iter_books(odds_rows): handle_book(b)
    return written

def ingest_ice_and_odds(
    db: Session,
    day: date,
    leagues: Iterable[str],
    prefer_book: Optional[str] = None,
    max_games: Optional[int] = None,
    odds_delay_sec: float = 0.35,
) -> dict:
    raw = api_fetch_fixtures(day, list(leagues)) or []
    if isinstance(max_games, int) and max_games > 0: raw = raw[:max_games]
    fixtures_written = 0; odds_written = 0
    for payload in raw:
        f = upsert_fixture_from_payload(db, payload); fixtures_written += 1
        try: gid = int(f.provider_fixture_id)
        except: continue
        try: odds_rows = fetch_odds_for_game(gid) or []
        except: odds_rows = []
        odds_written += _parse_and_write_markets(db, f.id, odds_rows, prefer_book=prefer_book)
        if odds_delay_sec: sleep(odds_delay_sec)
    db.commit()
    return {"date": str(day), "leagues": list(leagues), "fixtures_upserted": fixtures_written, "odds_rows_written": odds_written}