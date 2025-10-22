# api/app/services/fixtures_nba.py
from __future__ import annotations
import re
from time import sleep
from typing import Iterable, Optional, Dict, List, Any
from datetime import date, datetime, timezone
from sqlalchemy.orm import Session

from ..models import Fixture, Odds
from .api_nba import fetch_fixtures as api_fetch_fixtures, fetch_odds_for_game


def _norm(s) -> str:
    if s is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _to_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _parse_iso_or_now(x: str | None) -> datetime:
    if not x:
        return datetime.now(timezone.utc)
    try:
        # API uses Zulu timestamps: 2025-10-22T02:00:00.000Z
        if x.endswith("Z"):
            x = x[:-1] + "+00:00"
        return _to_utc(datetime.fromisoformat(x))
    except Exception:
        return datetime.now(timezone.utc)


def _kickoff_from_payload(p: dict) -> datetime:
    """
    NBA response gives: p["date"]["start"] as ISO string.
    Fallbacks included for robustness.
    """
    if not isinstance(p, dict):
        return datetime.now(timezone.utc)

    dt_start = None
    try:
        dt_start = ((p.get("date") or {}).get("start"))
    except Exception:
        dt_start = None

    if dt_start:
        return _parse_iso_or_now(str(dt_start))

    # Fallbacks
    ts = (p.get("timestamp") or (p.get("game") or {}).get("timestamp"))
    if isinstance(ts, (int, float)) and ts > 0:
        try:
            return datetime.fromtimestamp(int(ts), tz=timezone.utc)
        except Exception:
            pass

    return datetime.now(timezone.utc)


def _league_name(p: dict) -> str:
    lg = p.get("league")
    if isinstance(lg, dict):
        return str(lg.get("name") or lg.get("key") or lg.get("slug") or "NBA").strip().upper()
    return str(lg or "NBA").strip().upper()


BOOK_BLACKLIST = {"1xbet"}


def _is_blacklisted_book(name: Optional[str]) -> bool:
    return _norm(name) in BOOK_BLACKLIST


def _is_whitelisted_book(name: Optional[str]) -> bool:
    return True


def _team_name(d: Any) -> str:
    """Return best-available team name from dict OR string; never blank."""
    if isinstance(d, dict):
        for k in ("name", "nickname", "code"):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        t = d.get("team")
        if isinstance(t, dict):
            for k in ("name", "nickname", "code"):
                v = t.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    elif isinstance(d, str):
        s = d.strip()
        if s:
            return s
    return "TBD"


def upsert_fixture_from_payload(db: Session, p: dict) -> Fixture:
    if not isinstance(p, dict):
        f = Fixture(
            provider_fixture_id=str(p),
            comp="NBA",
            home_team="TBD",
            away_team="TBD",
            kickoff_utc=datetime.now(timezone.utc),
            sport="nba",
        )
        db.add(f)
        db.flush()
        return f

    gid = str(p.get("id") or (p.get("game") or {}).get("id") or "")

    tm = p.get("teams") or {}
    if not isinstance(tm, dict):
        tm = {}

    # NBA API: "visitors" (away) and "home"
    visitors = tm.get("visitors") or tm.get("away") or {}
    home = tm.get("home") or tm.get("host") or {}

    away_name = _team_name(visitors)
    home_name = _team_name(home)
    kickoff = _kickoff_from_payload(p)
    comp = _league_name(p)
    if comp.lower() == "standard":
        comp = "NBA"
    sport = "nba"

    if gid:
        f = db.query(Fixture).filter(Fixture.provider_fixture_id == gid).one_or_none()
        if f:
            f.comp = comp
            f.home_team = home_name
            f.away_team = away_name
            f.kickoff_utc = kickoff
            f.sport = sport
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
        db.add(f)
        db.flush()
        return f

    # Fallback identity if no provider id
    f = (
        db.query(Fixture)
        .filter(
            Fixture.sport == sport,
            Fixture.comp == comp,
            Fixture.home_team == home_name,
            Fixture.away_team == away_name,
            Fixture.kickoff_utc == kickoff,
        )
        .one_or_none()
    )
    if f:
        return f

    f = Fixture(
        provider_fixture_id=None,
        comp=comp,
        home_team=home_name,
        away_team=away_name,
        kickoff_utc=kickoff,
        sport=sport,
    )
    db.add(f)
    db.flush()
    return f


def _upsert_odds(db: Session, fixture_id: int, bookmaker: str, market: str, price: float):
    if not (price and 1.01 < price < 101.0):
        return
    now = datetime.now(timezone.utc)
    row = (
        db.query(Odds)
        .filter(Odds.fixture_id == fixture_id, Odds.bookmaker == bookmaker, Odds.market == market)
        .one_or_none()
    )
    if row:
        row.price = price
        row.last_seen = now
    else:
        db.add(Odds(fixture_id=fixture_id, bookmaker=bookmaker, market=market, price=price, last_seen=now))
        db.flush()


def _parse_and_write_markets(
    db: Session, fixture_id: int, odds_rows: List[Any], *, prefer_book: Optional[str] = None
) -> int:
    """
    Moneyline: HOME_WIN / AWAY_WIN
    Spread:    SPREAD_HOME / SPREAD_AWAY
    Totals:    O{line} / U{line} (e.g., O221.5 / U221.5)
    """
    if not odds_rows:
        return 0
    written = 0
    preferred = _norm(prefer_book) if prefer_book else ""

    def iter_books(rows: List[Any]):
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            if row.get("bets") or row.get("markets"):
                yield row
            for b in (row.get("bookmakers") or row.get("books") or []):
                if isinstance(b, dict):
                    yield b

    def handle_book(b: dict):
        nonlocal written
        if not isinstance(b, dict):
            return
        bname_raw = (b.get("name") or b.get("bookmaker") or "")
        if _is_blacklisted_book(bname_raw) or not _is_whitelisted_book(bname_raw):
            return
        bname = _norm(bname_raw)

        bets = b.get("bets") or b.get("markets") or []
        if not isinstance(bets, list):
            return

        totals_by_line: Dict[float, Dict[str, float]] = {}

        for mk in bets:
            if not isinstance(mk, dict):
                continue
            name = _norm(mk.get("name") or mk.get("market") or "")
            vals = mk.get("values") or mk.get("outcomes") or []
            if not isinstance(vals, list):
                continue

            # Moneyline
            if name in ("moneyline", "ml", "winner", "matchwinner", "homeaway"):
                home = next(
                    (v for v in vals if _norm(v.get("name") or v.get("label") or v.get("value")) in ("home", "1", "h")),
                    None,
                )
                away = next(
                    (v for v in vals if _norm(v.get("name") or v.get("label") or v.get("value")) in ("away", "2", "a")),
                    None,
                )
                if isinstance(home, dict) and home.get("odd"):
                    try:
                        _upsert_odds(db, fixture_id, bname, "HOME_WIN", float(str(home["odd"]).replace(",", ".")))
                        written += 1
                    except:
                        pass
                if isinstance(away, dict) and away.get("odd"):
                    try:
                        _upsert_odds(db, fixture_id, bname, "AWAY_WIN", float(str(away["odd"]).replace(",", ".")))
                        written += 1
                    except:
                        pass

            # Spread (handicap)
            if ("spread" in name) or ("handicap" in name and "total" not in name) or ("line" in name and "total" not in name):
                h = next((v for v in vals if "home" in _norm(v.get("name") or v.get("label") or v.get("value"))), None)
                a = next((v for v in vals if "away" in _norm(v.get("name") or v.get("label") or v.get("value"))), None)
                if isinstance(h, dict) and h.get("odd"):
                    try:
                        _upsert_odds(db, fixture_id, bname, "SPREAD_HOME", float(str(h["odd"]).replace(",", ".")))
                        written += 1
                    except:
                        pass
                if isinstance(a, dict) and a.get("odd"):
                    try:
                        _upsert_odds(db, fixture_id, bname, "SPREAD_AWAY", float(str(a["odd"]).replace(",", ".")))
                        written += 1
                    except:
                        pass

            # Totals (game points)
            if ("total" in name) or ("overunder" in name) or ("points" in name) or ("totals" in name):
                for v in vals:
                    if not isinstance(v, dict):
                        continue
                    lab = _norm(v.get("name") or v.get("label") or v.get("value"))
                    line = None
                    for k in ("total", "points", "point", "line", "handicap"):
                        if v.get(k) is not None:
                            try:
                                line = float(str(v[k]).replace(",", "."))
                                break
                            except:
                                pass
                    if line is None:
                        m = re.search(r"(\d+(?:\.\d+)?)", str(v.get("name") or v.get("label") or v.get("value") or ""))
                        if m:
                            try:
                                line = float(m.group(1))
                            except:
                                pass
                    if line is None or not v.get("odd"):
                        continue
                    side = "O" if lab.startswith("over") else ("U" if lab.startswith("under") else None)
                    if side is None:
                        continue
                    try:
                        price = float(str(v["odd"]).replace(",", "."))
                    except Exception:
                        continue
                    d = totals_by_line.setdefault(line, {})
                    d[side] = price

        # pick a representative totals line (NBA typical range)
        if totals_by_line:
            lines = sorted([ln for ln in totals_by_line.keys() if 150.5 <= ln <= 300.5])
            if lines:
                mid = lines[len(lines) // 2]
                pair = totals_by_line.get(mid) or {}
                o, u = pair.get("O"), pair.get("U")
                if o and u and o > 1.01 and u > 1.01:
                    tag = f"{mid:.1f}".rstrip("0").rstrip(".")
                    _upsert_odds(db, fixture_id, bname, f"O{tag}", o)
                    _upsert_odds(db, fixture_id, bname, f"U{tag}", u)
                    written += 2

    if preferred:
        for b in iter_books(odds_rows):
            name = _norm(
                (b.get("name") if isinstance(b, dict) else "")
                or (isinstance(b, dict) and b.get("bookmaker") or "")
            )
            if name == preferred:
                handle_book(b)
        for b in iter_books(odds_rows):
            name = _norm(
                (b.get("name") if isinstance(b, dict) else "")
                or (isinstance(b, dict) and b.get("bookmaker") or "")
            )
            if name != preferred:
                handle_book(b)
    else:
        for b in iter_books(odds_rows):
            handle_book(b)
    return written


def ingest_nba_and_odds(
    db: Session,
    day: date,
    leagues: Iterable[str] | None,
    *,
    prefer_book: Optional[str] = None,
    max_games: Optional[int] = None,
    odds_delay_sec: float = 0.35,
) -> dict:
    # leagues are ignored for NBA (API filters by date), but keep shape consistent
    league_list: List[str] = []
    if leagues is None:
        league_list = ["NBA"]
    else:
        if isinstance(leagues, str):
            league_list = [s.strip() for s in leagues.split(",") if s.strip()]
        else:
            league_list = [str(s).strip() for s in leagues if str(s).strip()]
        if not league_list:
            league_list = ["NBA"]

    raw = api_fetch_fixtures(day, league_list) or []
    raw = [r for r in raw if isinstance(r, dict)]

    if isinstance(max_games, int) and max_games > 0:
        raw = raw[:max_games]

    fixtures_written = 0
    odds_written = 0

    for payload in raw:
        try:
            f = upsert_fixture_from_payload(db, payload)
            fixtures_written += 1
        except Exception:
            continue

        pid = f.provider_fixture_id
        if not pid:
            continue
        try:
            gid = int(str(pid))
        except Exception:
            continue

        try:
            odds_rows = fetch_odds_for_game(gid) or []
        except Exception:
            odds_rows = []

        safe_rows = [r for r in odds_rows if isinstance(r, dict)]
        try:
            odds_written += _parse_and_write_markets(db, f.id, safe_rows, prefer_book=prefer_book)
        except Exception:
            pass

        if odds_delay_sec:
            sleep(odds_delay_sec)

    db.commit()
    return {
        "date": str(day),
        "leagues": league_list,
        "fixtures_upserted": fixtures_written,
        "odds_rows_written": odds_written,
    }