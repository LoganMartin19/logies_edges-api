# api/app/services/form.py
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import or_, desc

from ..models import Fixture
from ..services.apifootball import canonicalize_comp  # map provider leagues -> internal keys (EPL, UCL, ...)

# ---------------------------------------------------------------------
# DB helpers (used as fallback)
# ---------------------------------------------------------------------

def _base_query(db: Session, team_name: str, before: datetime, comp: Optional[str]):
    q = (
        db.query(Fixture)
        .filter(
            Fixture.kickoff_utc < before,
            or_(Fixture.home_team == team_name, Fixture.away_team == team_name),
            Fixture.full_time_home.isnot(None),
            Fixture.full_time_away.isnot(None),
        )
    )
    if comp:
        q = q.filter(Fixture.comp == comp)
    return q


def get_recent_form(
    db: Session,
    team_name: str,
    before: datetime,
    n: int = 5,
    *,
    comp: Optional[str] = None,
) -> Dict[str, Any]:
    fixtures = (
        _base_query(db, team_name, before, comp)
        .order_by(desc(Fixture.kickoff_utc))
        .limit(n)
        .all()
    )

    played = len(fixtures)
    if not played:
        return {
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_diff": 0,
            "avg_goals_for": 0.0,
            "avg_goals_against": 0.0,
        }

    wins = draws = losses = 0
    gf_total = ga_total = 0
    for f in fixtures:
        is_home = (f.home_team == team_name)
        gf = f.full_time_home if is_home else f.full_time_away
        ga = f.full_time_away if is_home else f.full_time_home
        gf_total += gf
        ga_total += ga
        if gf > ga:
            wins += 1
        elif gf < ga:
            losses += 1
        else:
            draws += 1

    return {
        "played": played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": gf_total,
        "goals_against": ga_total,
        "goal_diff": gf_total - ga_total,
        "avg_goals_for": round(gf_total / played, 2),
        "avg_goals_against": round(ga_total / played, 2),
    }


def get_recent_fixtures(
    db: Session,
    team_name: str,
    before: datetime,
    n: int = 5,
    *,
    comp: Optional[str] = None,
) -> List[Dict[str, Any]]:
    fixtures = (
        _base_query(db, team_name, before, comp)
        .order_by(desc(Fixture.kickoff_utc))
        .limit(n)
        .all()
    )

    results: List[Dict[str, Any]] = []
    for f in fixtures:
        is_home = (f.home_team == team_name)
        opponent = f.away_team if is_home else f.home_team
        gf = f.full_time_home if is_home else f.full_time_away
        ga = f.full_time_away if is_home else f.full_time_home
        result = "win" if gf > ga else "loss" if gf < ga else "draw"
        results.append({
            "date": f.kickoff_utc.isoformat(),
            "team": team_name,
            "opponent": opponent,
            "is_home": is_home,
            "goals_for": gf,
            "goals_against": ga,
            "score": f"{gf}-{ga}",
            "result": result,
            "fixture_id": f.id,
            "comp": f.comp,     # internal key (DB)
            "comp_key": f.comp, # normalized alongside API results
        })
    return results


# ---------------------------------------------------------------------
# Football API-first logic
# ---------------------------------------------------------------------

def _infer_season(as_of: datetime) -> int:
    """July–June season inference (e.g., Sep 2025 -> 2025)."""
    return as_of.year if as_of.month >= 7 else as_of.year - 1


def _recent_from_api(
    team_provider_id: int,
    as_of: datetime,
    *,
    season: Optional[int],
    limit: int,
    comp: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Returns rows from the team's perspective:
      goals_for / goals_against reflect the team,
      score is rendered as 'GF-GA',
      result is derived from GF/GA,
      is_home indicates whether THIS match was at home for the team.
    """
    from ..services.apifootball import get_team_recent_results

    season = season or _infer_season(as_of)

    try:
        api_matches: List[dict] = get_team_recent_results(
            team_id=team_provider_id,
            season=season,
            limit=max(limit * 3, limit),
            before_iso=as_of.isoformat(),
        ) or []
    except TypeError:
        api_matches = get_team_recent_results(
            team_id=team_provider_id,
            season=season,
            n=max(limit * 3, limit),
        ) or []

    comp_upper = (comp or "").upper()
    recent: List[Dict[str, Any]] = []

    for m in api_matches:
        league_name = m.get("league_name") or (m.get("league", {}) or {}).get("name", "")
        league_obj = m.get("league") or {
            "id": m.get("league_id"),
            "name": league_name,
            "country": m.get("league_country"),
        }
        comp_key = (m.get("comp_key") or canonicalize_comp(league_obj) or "").upper()

        if comp_upper and comp_upper not in {comp_key, (league_name or "").strip().upper()}:
            continue

        date_iso = m.get("date")
        # is_home may arrive as True/False or "home"/"away"
        raw_home = m.get("is_home")
        is_home = raw_home if isinstance(raw_home, bool) else str(raw_home).lower().startswith("h")

        # parse score (provider home-away)
        score_str = m.get("score")
        if not score_str and "goals" in m:
            gh = int((m["goals"].get("home") or 0))
            ga = int((m["goals"].get("away") or 0))
            score_str = f"{gh}-{ga}"
        if not score_str:
            score_str = "0-0"

        try:
            home_g, away_g = [int(x) for x in score_str.split("-", 1)]
        except Exception:
            home_g, away_g = 0, 0

        # Team perspective
        gf, ga = (home_g, away_g) if is_home else (away_g, home_g)
        score_render = f"{gf}-{ga}"
        res = "win" if gf > ga else ("loss" if gf < ga else "draw")

        recent.append({
            "date": date_iso,
            "team": None,
            "opponent": m.get("opponent"),
            "is_home": bool(is_home),
            "goals_for": int(gf),
            "goals_against": int(ga),
            "score": score_render,      # perspective-safe
            "result": res,
            "fixture_id": m.get("fixture_id"),
            "comp": league_name,        # provider display name
            "comp_key": comp_key,       # internal key (EPL, UCL, etc.)
        })

    recent.sort(key=lambda x: x["date"] or "", reverse=True)
    return recent[:limit]


def get_recent_form_api_first(
    db: Session,
    team_name: str,
    team_provider_id: Optional[int],
    before: datetime,
    n: int = 5,
    *,
    comp: Optional[str] = None,
    season: Optional[int] = None,
) -> Dict[str, Any]:
    recent: List[Dict[str, Any]] = []
    if team_provider_id:
        recent = _recent_from_api(
            team_provider_id,
            before,
            season=season,
            limit=max(n * 3, 12),
            comp=comp,
        )

    if not recent:
        recent = get_recent_fixtures(db, team_name, before, n=n, comp=comp)

    # 🔒 Final consistency pass: ensure result matches goals_for/goals_against
    for r in recent:
        try:
            gf = int(r.get("goals_for", 0))
            ga = int(r.get("goals_against", 0))
            r["score"] = f"{gf}-{ga}"
            r["result"] = "win" if gf > ga else "loss" if gf < ga else "draw"
        except Exception:
            r["result"] = r.get("result") or "draw"

    recent.sort(key=lambda m: m["date"] or "", reverse=True)
    recent = recent[:n]

    wins = draws = losses = 0
    gf_total = ga_total = 0
    for m in recent:
        gf_total += int(m["goals_for"])
        ga_total += int(m["goals_against"])
        if m["goals_for"] > m["goals_against"]:
            wins += 1
        elif m["goals_for"] < m["goals_against"]:
            losses += 1
        else:
            draws += 1

    played = len(recent)
    summary = {
        "played": played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_for": gf_total,
        "goals_against": ga_total,
        "goal_diff": gf_total - ga_total,
        "avg_goals_for": round(gf_total / played, 2) if played else 0.0,
        "avg_goals_against": round(ga_total / played, 2) if played else 0.0,
    }

    return {"summary": summary, "recent": recent}


# ---------------------------------------------------------------------
# Hybrid logic (Football + Gridiron + Hockey)
# ---------------------------------------------------------------------

# Optional imports – keep this module decoupled
try:
    from .api_gridiron import fetch_team_games as _gridiron_fetch_team_games  # type: ignore
except Exception:
    _gridiron_fetch_team_games = None  # gracefully fallback to DB

try:
    from .api_ice import fetch_team_games as _hockey_fetch_team_games  # type: ignore
except Exception:
    _hockey_fetch_team_games = None  # gracefully fallback to DB


def get_hybrid_form_for_fixture(
    db: Session,
    fixture: Fixture,
    n: int = 5,
    comp_scope: bool = True,   # True = restrict to fixture.comp (internal key), False = all comps
) -> dict:
    """
    Returns:
      {
        "home": {"summary": {...}, "recent": [...]},
        "away": {"summary": {...}, "recent": [...]},
        "season": inferred_season_int
      }
    Uses API-first logic for each sport (football/hockey/gridiron), falling back to DB when needed.
    """
    as_of = fixture.kickoff_utc or datetime.now(timezone.utc)
    comp = fixture.comp if comp_scope else None  # internal key expected here
    comp_upper = (fixture.comp or "").upper()
    sport = (getattr(fixture, "sport", "") or "").lower()  # e.g., "nfl"/"cfb"/"soccer"/"nhl"

    # Provider team IDs from fixture row (if present)
    home_pid = getattr(fixture, "provider_home_team_id", None)
    away_pid = getattr(fixture, "provider_away_team_id", None)

    season = _infer_season(as_of)

    # --- Resolve missing provider team IDs (multi-sport) ---
    if not (home_pid and away_pid) and getattr(fixture, "provider_fixture_id", None):
        prov_id = int(fixture.provider_fixture_id)
        core = None
        try:
            # 🏒 prefer hockey if comp/sport indicates NHL
            if "NHL" in comp_upper or "HOCKEY" in comp_upper or "ICE" in comp_upper or sport == "nhl":
                from .api_ice import get_fixture as _api_ice_get_fixture
                fxh = _api_ice_get_fixture(prov_id)
                core = (fxh.get("response") or [None])[0]
            # 🏈 prefer gridiron if NFL/CFB
            elif any(k in comp_upper for k in ("NFL", "CFB", "NCAA", "AMERICAN")) or sport in {"nfl", "cfb", "ncaa", "american"}:
                from .api_gridiron import get_fixture as _api_gridiron_get_fixture
                fxg = _api_gridiron_get_fixture(prov_id)
                core = (fxg.get("response") or [None])[0]
            # ⚽️ default to football
            else:
                from .apifootball import get_fixture as _api_get_fixture
                fx = _api_get_fixture(prov_id)
                core = (fx.get("response") or [None])[0]
        except Exception:
            core = None

        if core:
            teams = core.get("teams") or {}
            home_pid = home_pid or (teams.get("home") or {}).get("id")
            away_pid = away_pid or (teams.get("away") or {}).get("id")
            # optional: capture season if present
            if (core.get("league") or {}).get("season"):
                season = core["league"]["season"]

            # ✅ Persist to DB so next calls don’t need resolver
            try:
                fixture.provider_home_team_id = home_pid
                fixture.provider_away_team_id = away_pid
                db.add(fixture)
                db.commit()
            except Exception:
                db.rollback()

    # SPORT ROUTING FLAGS
    is_hockey = ("NHL" in comp_upper) or ("HOCKEY" in comp_upper) or (sport == "nhl") or ("ICE" in comp_upper)
    is_gridiron = any(k in comp_upper for k in ("NFL", "CFB", "NCAA", "AMERICAN")) or sport in {"nfl", "cfb", "ncaa", "american"}

    # ---------- helpers ----------
    def _summarize(recent: List[Dict[str, Any]]) -> Dict[str, Any]:
        wins = draws = losses = gf = ga = 0
        for r in recent:
            gf += int(r.get("goals_for", 0))
            ga += int(r.get("goals_against", 0))
            if r.get("goals_for", 0) > r.get("goals_against", 0):
                wins += 1
            elif r.get("goals_for", 0) < r.get("goals_against", 0):
                losses += 1
            else:
                draws += 1
        p = len(recent)
        return {
            "played": p,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "goals_for": gf,
            "goals_against": ga,
            "goal_diff": gf - ga,
            "avg_goals_for": round(gf / p, 2) if p else 0.0,
            "avg_goals_against": round(ga / p, 2) if p else 0.0,
        }

    def _normalize_gridiron_games(rows: List[Dict[str, Any]], team_id: Optional[int], team_name: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for g in rows or []:
            teams = g.get("teams") or {}
            home_blk = teams.get("home") or g.get("homeTeam") or {}
            away_blk = teams.get("away") or g.get("awayTeam") or {}
            home_id = home_blk.get("id")
            away_id = away_blk.get("id")
            home_nm = home_blk.get("name") or home_blk.get("nickname") or home_blk.get("code") or "Home"
            away_nm = away_blk.get("name") or away_blk.get("nickname") or away_blk.get("code") or "Away"

            is_home = None
            if team_id is not None:
                is_home = (home_id == team_id)
            if is_home is None:
                is_home = (home_nm == team_name)

            scores = g.get("scores") or g.get("score") or {}
            def _score(side: str) -> int:
                v = scores.get(side)
                if isinstance(v, int):
                    return int(v)
                try:
                    return int((v or {}).get("total", 0))
                except Exception:
                    return 0

            ts = _score("home") if is_home else _score("away")
            os = _score("away") if is_home else _score("home")

            date_iso = g.get("date")
            if not isinstance(date_iso, str):
                date_iso = str(g.get("timestamp") or "")

            opp = (away_nm if is_home else home_nm)
            res = "win" if ts > os else ("loss" if ts < os else "draw")
            out.append({
                "date": date_iso,
                "team": team_name,
                "opponent": opp,
                "is_home": bool(is_home),
                "goals_for": int(ts or 0),        # “goals” naming is used app-wide; it’s points here
                "goals_against": int(os or 0),
                "score": f"{int(ts or 0)}-{int(os or 0)}",
                "result": res,
                "fixture_id": g.get("id") or (g.get("game") or {}).get("id"),
                "comp": (g.get("league") or {}).get("name") or "NFL/CFB",
                "comp_key": (g.get("league") or {}).get("name") or "NFL/CFB",
            })
        out.sort(key=lambda x: x["date"] or "", reverse=True)
        return out[:n]

    def _normalize_hockey_games(rows: List[Dict[str, Any]], team_id: Optional[int], team_name: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for g in rows or []:
            teams = g.get("teams") or {}
            home_blk = teams.get("home") or {}
            away_blk = teams.get("away") or {}
            home_id = home_blk.get("id")
            away_id = away_blk.get("id")
            home_nm = home_blk.get("name") or "Home"
            away_nm = away_blk.get("name") or "Away"

            is_home = None
            if team_id is not None:
                is_home = (home_id == team_id)
            if is_home is None:
                is_home = (home_nm == team_name)

            scores = g.get("scores") or {}
            def _score(side: str) -> int:
                v = scores.get(side)
                if isinstance(v, int):
                    return int(v)
                try:
                    return int((v or {}).get("total", 0))
                except Exception:
                    return 0

            tf = _score("home") if is_home else _score("away")
            ta = _score("away") if is_home else _score("home")

            date_iso = g.get("date") or str(g.get("timestamp") or "")
            opp = (away_nm if is_home else home_nm)
            res = "win" if tf > ta else ("loss" if tf < ta else "draw")
            out.append({
                "date": date_iso,
                "team": team_name,
                "opponent": opp,
                "is_home": bool(is_home),
                "goals_for": int(tf or 0),
                "goals_against": int(ta or 0),
                "score": f"{int(tf or 0)}-{int(ta or 0)}",
                "result": res,
                "fixture_id": g.get("id"),
                "comp": (g.get("league") or {}).get("name") or "NHL",
                "comp_key": (g.get("league") or {}).get("name") or "NHL",
            })
        out.sort(key=lambda x: x["date"] or "", reverse=True)
        return out[:n]

    # ---------- HOME ----------
    if is_gridiron and _gridiron_fetch_team_games and home_pid:
        home_api_rows = _gridiron_fetch_team_games(team_id=home_pid, season=None, league=None, status="FT", last_n=n) or []
        home_recent = _normalize_gridiron_games(home_api_rows, home_pid, fixture.home_team)
    elif is_hockey and _hockey_fetch_team_games and home_pid:
        home_api_rows = _hockey_fetch_team_games(team_id=home_pid, season=None, league="NHL", status="FT", last_n=n) or []
        home_recent = _normalize_hockey_games(home_api_rows, home_pid, fixture.home_team)
    else:
        home_recent = get_recent_form_api_first(
            db=db,
            team_name=fixture.home_team,
            team_provider_id=home_pid,
            before=as_of,
            n=n,
            comp=comp,
            season=season,
        )["recent"]

    # ---------- AWAY ----------
    if is_gridiron and _gridiron_fetch_team_games and away_pid:
        away_api_rows = _gridiron_fetch_team_games(team_id=away_pid, season=None, league=None, status="FT", last_n=n) or []
        away_recent = _normalize_gridiron_games(away_api_rows, away_pid, fixture.away_team)
    elif is_hockey and _hockey_fetch_team_games and away_pid:
        away_api_rows = _hockey_fetch_team_games(team_id=away_pid, season=None, league="NHL", status="FT", last_n=n) or []
        away_recent = _normalize_hockey_games(away_api_rows, away_pid, fixture.away_team)
    else:
        away_recent = get_recent_form_api_first(
            db=db,
            team_name=fixture.away_team,
            team_provider_id=away_pid,
            before=as_of,
            n=n,
            comp=comp,
            season=season,
        )["recent"]

    # ---------- SUMMARIES ----------
    home_summary = _summarize(home_recent)
    away_summary = _summarize(away_recent)

    return {
        "home": {"summary": home_summary, "recent": home_recent},
        "away": {"summary": away_summary, "recent": away_recent},
        "season": season,
    }

def get_fixture_form_summary(db: Session, fixture_id: int, n: int = 5) -> dict:
    """Convenience wrapper for expert predictions and analytics."""
    fixture = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fixture:
        return {}

    hybrid = get_hybrid_form_for_fixture(db, fixture, n=n)
    home = hybrid.get("home", {}).get("summary", {})
    away = hybrid.get("away", {}).get("summary", {})
    return {"home": home, "away": away}