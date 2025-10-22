# api/app/services/fixtures.py
from __future__ import annotations

import re
from time import sleep
from typing import Iterable
from datetime import date, datetime, timezone

from sqlalchemy.orm import Session

from ..models import Fixture, Odds
from .apifootball import fetch_fixtures as api_fetch_fixtures, fetch_odds_for_fixture, canonicalize_comp

# ---------- basic helpers ----------

def _norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def _latest_iso(dt: str) -> datetime:
    return datetime.fromisoformat(dt.replace("Z", "+00:00"))

def _as_dot(x: str | None) -> str:
    return (x or "").replace(",", ".").strip()

def _val_label(v: dict) -> str:
    # SAFE: value may be int/float/str/missing
    return str(v.get("value") or "").strip().lower()

def _val_odd(v: dict) -> float | None:
    try:
        o = v.get("odd")
        return float(o) if o is not None else None
    except Exception:
        return None

def _val_line_ish(v: dict) -> str:
    for k in ("handicap", "line", "total", "points", "point", "goal", "goals"):
        if v.get(k) is not None:
            return _as_dot(str(v[k]))
    lab = _val_label(v)
    # fallback: extract like "Over 2.5"
    m = re.search(r"(\d+(?:[.,]\d)?)", lab)
    return _as_dot(m.group(1)) if m else ""

def _values_sorted(values: list[dict]) -> list[dict]:
    def _ts(v):
        return v.get("last_update") or v.get("lastUpdate") or ""
    return sorted(values or [], key=_ts, reverse=True)

# ---- market recognizers (soccer) ----

_TOT_PATTERNS = (
    "over/under", "under/over", "total", "goals over/under", "total goals",
    "goal line", "goals line", "match goals", "goals"
)

def _is_totals_market(name: str) -> bool:
    n = (name or "").strip().lower()
    return any(pat in n for pat in _TOT_PATTERNS)

def _is_btts_market(name: str) -> bool:
    n = (name or "").strip().lower()
    return n in ("btts",) or ("both teams to score" in n) or ("both teams score" in n)

def _is_moneyline_market(name: str) -> bool:
    n = (name or "").strip().lower()
    # Treat these as "result" markets; we’ll decide 3-way vs 2-way from outcomes.
    return n in (
        "winner", "match winner", "matchwinner",
        "1x2", "3way", "3 way", "3-way", "3-way result",
        "result", "match result", "full time result", "fulltime result", "ft result"
    )

def _norm_val_label(v: dict) -> str:
    return re.sub(r"[^a-z0-9]+", "", (_val_label(v) or "").lower())

# ---------- bookmaker filters ----------

BOOK_BLACKLIST = {"1xbet"}

BOOK_WHITELIST = {
    "bet365", "williamhill", "betfair", "unibet",
    "skybet", "ladbrokes", "coral", "betvictor"
}

def _is_blacklisted_book(name: str | None) -> bool:
    return _norm(name) in BOOK_BLACKLIST

def _is_whitelisted_book(name: str | None) -> bool:
    return _norm(name) in BOOK_WHITELIST

# ---------- DB upserts ----------
def upsert_fixture_from_payload(db: Session, payload: dict) -> Fixture:
    fx = payload["fixture"]
    lg = payload.get("league", {})
    tm = payload["teams"]

    ext_id = str(fx["id"])
    kickoff = _latest_iso(fx["date"])
    home = tm["home"]["name"]
    away = tm["away"]["name"]

    # ✅ Canonicalize competition name
    comp = canonicalize_comp(lg)  # pass full league dict (id, name, country)
    if not comp:  # fallback if canonicalize_comp returns None
        comp = lg.get("name") or lg.get("round") or "FOOTBALL"

    league_name = (lg.get("name") or "").lower()
    sport = "nfl" if "nfl" in league_name or "american football" in league_name else "football"

    f = (
        db.query(Fixture)
        .filter(Fixture.provider_fixture_id == ext_id)
        .one_or_none()
    )
    if f:
        f.comp = comp
        f.home_team = home
        f.away_team = away
        f.kickoff_utc = kickoff
        f.sport = sport

        f.provider_league_id = lg.get("id")
        f.provider_home_team_id = tm["home"].get("id")
        f.provider_away_team_id = tm["away"].get("id")

        db.flush()
        return f

    f = Fixture(
        provider_fixture_id=ext_id,
        comp=comp,
        home_team=home,
        away_team=away,
        kickoff_utc=kickoff,
        sport=sport,
        provider_league_id=lg.get("id"),
        provider_home_team_id=tm["home"].get("id"),
        provider_away_team_id=tm["away"].get("id"),
    )
    db.add(f); db.flush()
    return f

def _upsert_odds(db: Session, fixture_id: int, bookmaker: str, market: str, price: float):
    row = (
        db.query(Odds)
        .filter(
            Odds.fixture_id == fixture_id,
            Odds.bookmaker == bookmaker,
            Odds.market == market,
        )
        .one_or_none()
    )
    now = datetime.now(timezone.utc)
    if row:
        row.price = price
        row.last_seen = now
    else:
        db.add(Odds(
            fixture_id=fixture_id,
            bookmaker=bookmaker,
            market=market,
            price=price,
            last_seen=now
        ))
        db.flush()

# ---------- soccer odds parsing (restrict totals to 1.5/2.5) ----------

SOCCER_TOTAL_LINES = {"2.5"}  # add/remove here if you want

def _write_first_over_under_for_line(
    db: Session,
    fixture_id: int,
    book_label: str,
    values: list[dict],
    line_str: str,
) -> int:
    """
    Writes O{line}/U{line} once per bookmaker if found in values.
    Returns how many rows written (0..2).
    """
    written = 0
    # Over X.X
    for v in values:
        if "over" in _val_label(v) and _val_line_ish(v) == line_str:
            odd = _val_odd(v)
            if odd and odd > 1.01:
                _upsert_odds(db, fixture_id, book_label, f"O{line_str}", odd)
                written += 1
                break
    # Under X.X
    for v in values:
        if "under" in _val_label(v) and _val_line_ish(v) == line_str:
            odd = _val_odd(v)
            if odd and odd > 1.01:
                _upsert_odds(db, fixture_id, book_label, f"U{line_str}", odd)
                written += 1
                break
    return written

def _parse_and_write_markets(
    db: Session,
    fixture_id: int,
    odds_rows: list[dict] | None,
    prefer_book: str | None = None,
    allow_fallback: bool = True,
) -> int:
    odds_rows = odds_rows or []
    written = 0
    pref_norm = _norm(prefer_book) if prefer_book else ""
    written_keys: set[tuple[str, str]] = set()

    def scan(prefer_only: bool):
        nonlocal written
        for row in odds_rows:
            for book in row.get("bookmakers") or []:
                bname_raw = (book.get("name") or book.get("id") or "")
                bname_norm = _norm(bname_raw)

                # ✅ enforce whitelist
                if not _is_whitelisted_book(bname_raw):
                    continue
                if _is_blacklisted_book(bname_raw):
                    continue
                if prefer_only and pref_norm and pref_norm not in bname_norm:
                    continue

                book_label = bname_norm
                ...
                for mk in book.get("bets") or []:
                    key = (mk.get("name") or "").strip().lower()
                    vals = _values_sorted(mk.get("values", []))
                    print(f"\n--- Market: {mk.get('name')} ---")
                    for v in vals:
                        print(f"  Label: {_val_label(v)} → Norm: {_norm_val_label(v)}")

                    # ---- Soccer Over/Under: ONLY 1.5 and 2.5 ----
                    if _is_totals_market(key):
                        for ln in SOCCER_TOTAL_LINES:
                            if (book_label, f"O{ln}") not in written_keys:
                                w = _write_first_over_under_for_line(db, fixture_id, book_label, vals, ln)
                                if w:
                                    if any("over" in _val_label(v) and _val_line_ish(v) == ln for v in vals):
                                        written_keys.add((book_label, f"O{ln}"))
                                    if any("under" in _val_label(v) and _val_line_ish(v) == ln for v in vals):
                                        written_keys.add((book_label, f"U{ln}"))
                                    written += w

                    # ---- BTTS ----
                    if _is_btts_market(key):
                        if (book_label, "BTTS_Y") not in written_keys:
                            for v in vals:
                                if _norm_val_label(v) in ("yes", "y"):
                                    odd = _val_odd(v)
                                    if odd and odd > 1.01:
                                        _upsert_odds(db, fixture_id, book_label, "BTTS_Y", odd)
                                        written_keys.add((book_label, "BTTS_Y")); written += 1; break
                        if (book_label, "BTTS_N") not in written_keys:
                            for v in vals:
                                if _norm_val_label(v) in ("no", "n"):
                                    odd = _val_odd(v)
                                    if odd and odd > 1.01:
                                        _upsert_odds(db, fixture_id, book_label, "BTTS_N", odd)
                                        written_keys.add((book_label, "BTTS_N")); written += 1; break

                    # ---- Result markets (robust 3-way vs 2-way) ----
                    if _is_moneyline_market(key):
                        labels = [_norm_val_label(v) for v in vals]
                        has_draw = any(l in ("x", "draw", "tie", "3") for l in labels)

                        if has_draw:
                            # 3-way: HOME_WIN / DRAW / AWAY_WIN
                            if (book_label, "HOME_WIN") not in written_keys:
                                for v in vals:
                                    if _norm_val_label(v) in ("1", "home", "team1"):
                                        odd = _val_odd(v)
                                        if odd and odd > 1.01:
                                            _upsert_odds(db, fixture_id, book_label, "HOME_WIN", odd)
                                            written_keys.add((book_label, "HOME_WIN")); written += 1; break
                            if (book_label, "DRAW") not in written_keys:
                                for v in vals:
                                    if _norm_val_label(v) in ("x", "draw", "tie", "3"):
                                        odd = _val_odd(v)
                                        if odd and odd > 1.01:
                                            _upsert_odds(db, fixture_id, book_label, "DRAW", odd)
                                            written_keys.add((book_label, "DRAW")); written += 1; break
                            if (book_label, "AWAY_WIN") not in written_keys:
                                for v in vals:
                                    if _norm_val_label(v) in ("2", "away", "team2"):
                                        odd = _val_odd(v)
                                        if odd and odd > 1.01:
                                            _upsert_odds(db, fixture_id, book_label, "AWAY_WIN", odd)
                                            written_keys.add((book_label, "AWAY_WIN")); written += 1; break
                        else:
                            # True 2-way: write ML_HOME / ML_AWAY (kept out of soccer 1X2 model)
                            if (book_label, "ML_HOME") not in written_keys:
                                for v in vals:
                                    if _norm_val_label(v) in ("1", "home", "team1", "yes"):
                                        odd = _val_odd(v)
                                        if odd and odd > 1.01:
                                            _upsert_odds(db, fixture_id, book_label, "ML_HOME", odd)
                                            written_keys.add((book_label, "ML_HOME")); written += 1; break
                            if (book_label, "ML_AWAY") not in written_keys:
                                for v in vals:
                                    if _norm_val_label(v) in ("2", "away", "team2", "no"):
                                        odd = _val_odd(v)
                                        if odd and odd > 1.01:
                                            _upsert_odds(db, fixture_id, book_label, "ML_AWAY", odd)
                                            written_keys.add((book_label, "ML_AWAY")); written += 1; break

                    # ---- Double Chance (1X, 12, X2) ----
                    if "double chance" in key or key.strip().lower() in {"doublechance"}:
                        print(f"--- Market: {key} ---")
                        if (book_label, "1X") not in written_keys:
                            for v in vals:
                                if _norm_val_label(v) in ("1x", "homeorx", "1orx", "homedraw"):
                                    odd = _val_odd(v)
                                    if odd and odd > 1.01:
                                        _upsert_odds(db, fixture_id, book_label, "1X", odd)
                                        written_keys.add((book_label, "1X")); written += 1; break
                        if (book_label, "12") not in written_keys:
                            for v in vals:
                                if _norm_val_label(v) in ("12", "1or2", "homeoraway", "homeaway"):
                                    odd = _val_odd(v)
                                    if odd and odd > 1.01:
                                        _upsert_odds(db, fixture_id, book_label, "12", odd)
                                        written_keys.add((book_label, "12")); written += 1; break
                        if (book_label, "X2") not in written_keys:
                            for v in vals:
                                if _norm_val_label(v) in ("x2", "awayorx", "2orx", "drawaway"):
                                    odd = _val_odd(v)
                                    if odd and odd > 1.01:
                                        _upsert_odds(db, fixture_id, book_label, "X2", odd)
                                        written_keys.add((book_label, "X2")); written += 1; break
    if not prefer_book:
        scan(prefer_only=False)
    else:
        scan(prefer_only=True)
        if allow_fallback:
            scan(prefer_only=False)

    return written

# ---------- public ingest functions ----------

def ingest_fixtures_and_odds(
    db: Session,
    day: date,
    leagues: Iterable[str],
    prefer_book: str | None = None,
    max_fixtures: int | None = None,
    odds_delay_sec: float = 0.35,
) -> dict:
    raw = api_fetch_fixtures(day, list(leagues)) or []

    def _ko_dt(item: dict) -> datetime:
        try:
            return datetime.fromisoformat(item["fixture"]["date"].replace("Z", "+00:00"))
        except Exception:
            return datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)

    raw.sort(key=_ko_dt)
    if isinstance(max_fixtures, int) and max_fixtures > 0:
        raw = raw[:max_fixtures]

    fixtures_written = 0
    odds_written = 0

    for payload in raw:
        f = upsert_fixture_from_payload(db, payload)
        fixtures_written += 1
        try:
            odds_rows = fetch_odds_for_fixture(int(payload["fixture"]["id"])) or []
        except Exception:
            odds_rows = []
        odds_written += _parse_and_write_markets(
            db, f.id, odds_rows, prefer_book=prefer_book, allow_fallback=True
        )
        sleep(odds_delay_sec)

    db.commit()
    return {
        "date": str(day),
        "leagues": list(leagues),
        "fixtures_upserted": fixtures_written,
        "odds_rows_written": odds_written,
    }

def ingest_odds_for_fixture_id(
    db: Session,
    provider_fixture_id: str | int,
    prefer_book: str | None = None,
    allow_fallback: bool = True,
) -> dict:
    f = (
        db.query(Fixture)
        .filter(Fixture.provider_fixture_id == str(provider_fixture_id))
        .one_or_none()
    )
    if not f:
        return {"fixture_found": False, "odds_rows_written": 0}
    try:
        odds_rows = fetch_odds_for_fixture(int(provider_fixture_id))
    except Exception:
        odds_rows = []
    written = _parse_and_write_markets(
        db, f.id, odds_rows, prefer_book=prefer_book, allow_fallback=allow_fallback
    )
    db.commit()
    return {
        "fixture_found": True,
        "fixture_id": f.id,
        "odds_rows_written": written,
        "book_preferred": prefer_book or "ALL",
    }