# api/app/routers/fixtures.py
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from collections import defaultdict

from ..db import get_db
from ..models import Fixture, Edge, Odds, LeagueStanding
from ..services.apifootball import get_standings_for_league
from ..services.apifootball import LEAGUE_MAP as SOCCER_LEAGUE_MAP

router = APIRouter(prefix="/api/fixtures", tags=["fixtures"])


# -------------------------------------------------------------------
# Generic fixtures endpoints
# -------------------------------------------------------------------

@router.get("/all")
def get_all_fixtures(db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc)
    fixtures = (
        db.query(Fixture)
        .filter(Fixture.kickoff_utc >= now)
        .order_by(Fixture.kickoff_utc.asc())
        .all()
    )
    return [
        {
            "id": f.id,
            "home_team": f.home_team,
            "away_team": f.away_team,
            "comp": f.comp,
            "kickoff_utc": f.kickoff_utc.isoformat(),
        }
        for f in fixtures
    ]


@router.get("/league")
def get_fixtures_by_league(league: str, db: Session = Depends(get_db)):
    fixtures = (
        db.query(Fixture)
        .filter(Fixture.comp == league)
        .order_by(Fixture.kickoff_utc.asc())
        .all()
    )
    return [
        {
            "id": f.id,
            "home_team": f.home_team,
            "away_team": f.away_team,
            "comp": f.comp,
            "kickoff_utc": f.kickoff_utc.isoformat(),
        } for f in fixtures
    ]


@router.get("/id/{fixture_id}/json")
def fixture_detail_json(fixture_id: int, db: Session = Depends(get_db)):
    f = db.query(Fixture).filter(Fixture.id == fixture_id).one_or_none()
    if not f:
        return JSONResponse({"error": "Fixture not found"}, status_code=404)

    sub = (
        db.query(Edge.market, func.max(Edge.edge).label("max_edge"))
        .filter(Edge.fixture_id == fixture_id)
        .group_by(Edge.market)
        .subquery()
    )

    best = (
        db.query(Edge)
        .join(sub, and_(Edge.market == sub.c.market, Edge.edge == sub.c.max_edge))
        .filter(Edge.fixture_id == fixture_id)
        .order_by(Edge.edge.desc())
        .all()
    )

    odds = (
        db.query(Odds)
        .filter(Odds.fixture_id == fixture_id)
        .order_by(Odds.market.asc(), Odds.last_seen.desc())
        .all()
    )

    return JSONResponse({
        "fixture": {
            "id": f.id,
            "home_team": f.home_team,
            "away_team": f.away_team,
            "comp": f.comp,
            "kickoff_utc": f.kickoff_utc.isoformat(),
        },
        "best_edges": [
            {
                "market": e.market,
                "price": float(e.price),
                "edge": float(e.edge),
                "bookmaker": e.bookmaker,
            } for e in best
        ],
        "odds": [
            {
                "market": o.market,
                "price": float(o.price),
                "bookmaker": o.bookmaker,
                "last_seen": o.last_seen.isoformat(),
            } for o in odds
        ],
    })


@router.get("/round")
def get_fixtures_same_round(fixture_id: int, db: Session = Depends(get_db)):
    """Return fixtures in the same competition and same kickoff date, excluding the current one."""
    f = db.query(Fixture).filter(Fixture.id == fixture_id).one_or_none()
    if not f:
        return JSONResponse({"error": "Fixture not found"}, status_code=404)

    kickoff_day = f.kickoff_utc.date()
    start = datetime.combine(kickoff_day, datetime.min.time(), tzinfo=f.kickoff_utc.tzinfo)
    end = datetime.combine(kickoff_day, datetime.max.time(), tzinfo=f.kickoff_utc.tzinfo)

    fixtures = (
        db.query(Fixture)
        .filter(Fixture.comp == f.comp)
        .filter(Fixture.kickoff_utc >= start, Fixture.kickoff_utc <= end)
        .filter(Fixture.id != fixture_id)
        .order_by(Fixture.kickoff_utc.asc())
        .all()
    )

    return [
        {
            "id": fx.id,
            "home_team": fx.home_team,
            "away_team": fx.away_team,
            "comp": fx.comp,
            "kickoff_utc": fx.kickoff_utc.isoformat(),
        }
        for fx in fixtures
    ]


# -------------------------------------------------------------------
# League table with caching (auto-refresh after 6 hours)
# -------------------------------------------------------------------

@router.get("/league/table")
def get_league_table(
    league: str = Query(...),
    refresh: int = Query(0, ge=0, le=1, description="Force refresh from provider if 1"),
    db: Session = Depends(get_db),
):
    """
    Return league table from DB if available (and less than 6 hours old).
    Auto-refresh if data stale or refresh=1.
    """
    league_id = SOCCER_LEAGUE_MAP.get(league)
    if not league_id:
        raise HTTPException(status_code=400, detail=f"Unknown league: {league}")

    now = datetime.utcnow()
    six_hours_ago = now - timedelta(hours=6)

    # Pull existing records
    rows = (
        db.query(LeagueStanding)
        .filter(LeagueStanding.league == league)
        .order_by(LeagueStanding.position.asc())
        .all()
    )

    # ✅ Return cached if recent and not forcing refresh
    if rows and not refresh:
        first = rows[0]
        last_updated = getattr(first, "updated_at", None)
        if not last_updated or last_updated > six_hours_ago:
            season = getattr(first, "season", None)
            return {
                "league": league,
                "league_name": league,
                "season": season,
                "table": [
                    {
                        "position": r.position,
                        "team": r.team,
                        "played": r.played,
                        "win": r.win,
                        "draw": r.draw,
                        "lose": r.lose,
                        "gf": r.gf,
                        "ga": r.ga,
                        "points": r.points,
                        "form": r.form,
                    }
                    for r in rows
                ],
            }

    # 🆕 Otherwise fetch fresh
    league_meta, standings = get_standings_for_league(league_id)

    db.query(LeagueStanding).filter(LeagueStanding.league == league).delete()
    to_add = []
    for row in standings:
        to_add.append(
            LeagueStanding(
                league=league,
                season=row.get("season") or league_meta.get("season"),
                team=row.get("team"),
                position=row.get("position"),
                played=row.get("played"),
                win=row.get("win"),
                draw=row.get("draw"),
                lose=row.get("lose"),
                gf=row.get("gf"),
                ga=row.get("ga"),
                points=row.get("points"),
                form=row.get("form"),
                updated_at=now,
            )
        )
    if to_add:
        db.add_all(to_add)
        db.commit()

    return {
        "league": league,
        "league_name": league_meta.get("name", league),
        "season": league_meta.get("season"),
        "table": standings,
    }


# -------------------------------------------------------------------
# Gridiron and Hockey fixtures
# -------------------------------------------------------------------

@router.get("/gridiron")
def get_gridiron_fixtures(
    day: str = Query(..., description="YYYY-MM-DD (UTC day)"),
    sport: str = Query("all", regex="^(all|nfl|cfb)$"),
    db: Session = Depends(get_db),
):
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    q = (
        db.query(Fixture)
        .filter(Fixture.kickoff_utc >= start, Fixture.kickoff_utc < end)
        .filter(or_(Fixture.sport == "nfl", Fixture.sport == "cfb"))
        .order_by(Fixture.kickoff_utc.asc())
    )
    if sport != "all":
        q = q.filter(Fixture.sport == sport)

    fixtures = q.all()
    if not fixtures:
        return {"day": day, "count": 0, "fixtures": []}

    fixture_ids = [f.id for f in fixtures]
    odds_rows = (
        db.query(Odds.fixture_id, Odds.bookmaker, Odds.market, Odds.price)
        .filter(Odds.fixture_id.in_(fixture_ids))
        .all()
    )

    by_fix = defaultdict(list)
    for fid, book, mkt, price in odds_rows:
        by_fix[fid].append({"book": book, "market": mkt, "price": float(price)})

    out = []
    for f in fixtures:
        odds_list = by_fix.get(f.id, [])
        best_home, best_away = None, None
        totals = defaultdict(dict)

        for o in odds_list:
            m = (o["market"] or "").upper()
            p = o["price"]
            if m == "HOME_WIN":
                best_home = max(best_home or 0.0, p)
            elif m == "AWAY_WIN":
                best_away = max(best_away or 0.0, p)
            elif m.startswith(("O", "U")):
                try:
                    line = float(m[1:].replace(" ", ""))
                except Exception:
                    continue
                side = "O" if m.startswith("O") else "U"
                totals[line][side] = max(totals[line].get(side, 0.0), p)

        main_total = None
        lines = [ln for ln, d in totals.items() if "O" in d and "U" in d]
        lines.sort()
        if lines:
            mid = len(lines) // 2
            main_total = lines[mid] if len(lines) % 2 == 1 else (lines[mid - 1] + lines[mid]) / 2.0

        total_summary = None
        if main_total is not None:
            pair = totals[main_total]
            total_summary = {"line": main_total, "O": pair.get("O"), "U": pair.get("U")}

        out.append({
            "id": f.id,
            "comp": f.comp,
            "sport": f.sport,
            "home_team": f.home_team,
            "away_team": f.away_team,
            "kickoff_utc": f.kickoff_utc.isoformat(),
            "odds": {
                "best_home": best_home,
                "best_away": best_away,
                "total": total_summary,
            },
        })

    return {"day": day, "count": len(out), "fixtures": out}


@router.get("/gridiron/upcoming")
def gridiron_upcoming(
    sport: str = Query(..., description="nfl or cfb"),
    days_ahead: int = Query(7, ge=1, le=31),
    db: Session = Depends(get_db),
):
    sport = (sport or "").strip().lower()
    if sport not in ("nfl", "cfb"):
        raise HTTPException(status_code=400, detail="sport must be 'nfl' or 'cfb'")

    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days_ahead)

    fixtures = (
        db.query(Fixture)
        .filter(Fixture.sport == sport)
        .filter(Fixture.kickoff_utc >= now, Fixture.kickoff_utc <= end)
        .order_by(Fixture.kickoff_utc.asc(), Fixture.id.asc())
        .all()
    )

    ids = [f.id for f in fixtures]
    best_home, best_away = {}, {}
    if ids:
        rows = (
            db.query(Odds)
            .filter(Odds.fixture_id.in_(ids))
            .filter(Odds.market.in_(["HOME_WIN", "AWAY_WIN"]))
            .order_by(Odds.last_seen.desc())
            .all()
        )
        for o in rows:
            if o.market == "HOME_WIN":
                cur = best_home.get(o.fixture_id)
                if not cur or (o.price or 0) > cur["price"]:
                    best_home[o.fixture_id] = {"price": float(o.price), "bookmaker": o.bookmaker}
            elif o.market == "AWAY_WIN":
                cur = best_away.get(o.fixture_id)
                if not cur or (o.price or 0) > cur["price"]:
                    best_away[o.fixture_id] = {"price": float(o.price), "bookmaker": o.bookmaker}

    def ser(f):
        return {
            "id": f.id,
            "provider_fixture_id": f.provider_fixture_id,
            "home_team": f.home_team,
            "away_team": f.away_team,
            "comp": f.comp,
            "kickoff_utc": f.kickoff_utc.isoformat(),
            "best_home": best_home.get(f.id),
            "best_away": best_away.get(f.id),
        }

    return {"sport": sport, "fixtures": [ser(f) for f in fixtures]}


@router.get("/ice/upcoming")
def ice_upcoming(
    sport: str = Query("nhl"),
    days_ahead: int = Query(7, ge=1, le=30),
    db: Session = Depends(get_db),
):
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days_ahead)
    rows = (
        db.query(Fixture)
        .filter(Fixture.sport == "nhl")
        .filter(Fixture.kickoff_utc >= now, Fixture.kickoff_utc <= end)
        .order_by(Fixture.kickoff_utc.asc())
        .all()
    )

    out = []
    for f in rows:
        best_home = (
            db.query(Odds.bookmaker, Odds.price)
            .filter(Odds.fixture_id == f.id, Odds.market == "HOME_WIN")
            .order_by(Odds.price.desc()).first()
        )
        best_away = (
            db.query(Odds.bookmaker, Odds.price)
            .filter(Odds.fixture_id == f.id, Odds.market == "AWAY_WIN")
            .order_by(Odds.price.desc()).first()
        )
        out.append({
            "id": f.id,
            "provider_fixture_id": f.provider_fixture_id,
            "home_team": f.home_team,
            "away_team": f.away_team,
            "comp": f.comp,
            "kickoff_utc": f.kickoff_utc.isoformat(),
            "best_home": ({"bookmaker": best_home[0], "price": float(best_home[1])} if best_home else None),
            "best_away": ({"bookmaker": best_away[0], "price": float(best_away[1])} if best_away else None),
        })
    return {"sport": "nhl", "fixtures": out}