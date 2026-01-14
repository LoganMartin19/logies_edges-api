# --- replace your import block header with this ---
from fastapi import APIRouter, Query, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from datetime import datetime, timezone, timedelta  # ✅ consolidated
import re
import html  # ✅ for unescaping player names

from ..db import get_db
from ..models import Fixture, PlayerOdds, ModelProb
from ..services.apifootball import (
    get_predictions,
    get_lineups,
    get_h2h,
    get_team_stats,
    get_top_scorers,
    get_players,
    get_injuries,
    get_events,
    get_fixture,
    get_player_stats,
    get_team_recent_results,
    get_fixture_statistics,
    get_team_shots_against_avgs,
    get_team_xg_avgs,
    get_team_fouls_from_statistics_avg,
    get_team_fouls_drawn_avg,
    get_team_fouls_committed_avg,
    get_fixture_players,
    BASE_URL, _get_all_pages,  # ✅ keep these here
)
from ..services.player_odds import ingest_player_odds
from ..services.player_model import (
    poisson_pmf,
    poisson_cdf,
    prob_over_xpoint5,
    prob_card,
    fair_odds,
    edge,
)
from ..services.player_cache import get_team_season_players_cached, get_fixture_players_cached
from ..auth_firebase import require_premium

router = APIRouter(prefix="/football", tags=["football"])

# at top with other imports
from ..services.apifootball import BASE_URL, _get_all_pages


def _fetch_team_season_players(team_id: int, season: int) -> list[dict]:
    """
    Fetch ALL pages from /players for a team+season (no league filter).
    This returns a flat list of 'response' rows (one per player),
    where each row.statistics is a list of competitions.
    """
    try:
        return _get_all_pages(
            f"{BASE_URL}/players", {"team": int(team_id), "season": int(season)}
        )
    except Exception:
        return []

# --- Team Stats caching -------------------------------------------------------
from sqlalchemy.orm import Session
from ..models import TeamSeasonStats

def _get_team_stats_cached(db: Session, team_id: int, league_id: int, season: int, refresh: bool = False):
    """
    Returns cached team stats from DB if available (<12h old), otherwise fetches from API.
    """
    row = (
        db.query(TeamSeasonStats)
        .filter_by(team_id=team_id, league_id=league_id, season=season)
        .first()
    )

    if row and not refresh:
        age = datetime.utcnow() - (row.updated_at or datetime.utcnow())
        if age < timedelta(hours=12):
            return row.stats_json

    # fetch fresh from API
    data = get_team_stats(team_id, league_id, season) or {}
    if not data:
        return row.stats_json if row else None

    if not row:
        row = TeamSeasonStats(
            team_id=team_id,
            league_id=league_id,
            season=season,
            stats_json=data,
            updated_at=datetime.utcnow(),
        )
        db.add(row)
    else:
        row.stats_json = data
        row.updated_at = datetime.utcnow()

    db.commit()
    return data
# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def resolve_provider_fixture_id(db: Session, fixture_id: int) -> int:
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx or not fx.provider_fixture_id:
        raise HTTPException(status_code=404, detail="Fixture not found")
    return int(fx.provider_fixture_id)


def _safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace("%", "")
        return float(s)
    except Exception:
        return default


def _blend_lambda(for_avg: float, opp_against_avg: float, w_for: float = 0.6) -> float:
    """Simple expected-goals proxy from GF avg and opponent GA avg."""
    w_opp = 1.0 - w_for
    return max(0.0, w_for * for_avg + w_opp * opp_against_avg)


def _match_1x2_from_poisson(lh: float, la: float, max_goals: int = 10) -> dict:
    """Sum independent goal PMFs up to max_goals to get 1X2 probabilities."""
    p_home = p_draw = p_away = 0.0
    ph = [poisson_pmf(h, lh) for h in range(max_goals + 1)]
    pa = [poisson_pmf(a, la) for a in range(max_goals + 1)]
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = ph[h] * pa[a]
            if h > a:
                p_home += p
            elif h == a:
                p_draw += p
            else:
                p_away += p
    s = max(p_home + p_draw + p_away, 1e-12)
    return {"home": p_home / s, "draw": p_draw / s, "away": p_away / s}


def _totals_btts_from_poisson(lh: float, la: float) -> dict:
    """Over 2.5 via total lambda; BTTS via complements."""
    lam_tot = lh + la
    p_le2 = sum(poisson_pmf(k, lam_tot) for k in range(0, 3))
    p_over25 = 1.0 - p_le2
    p_h0 = poisson_pmf(0, lh)
    p_a0 = poisson_pmf(0, la)
    p_btts = 1.0 - p_h0 - p_a0 + (p_h0 * p_a0)
    return {"over_2_5": p_over25, "btts_yes": p_btts}


# ---------------------------------------------------------------------------
# Player props data (league-season scoped), with normalization fallbacks
# ---------------------------------------------------------------------------

def _get_player_props_data(fixture_id: int, db: Session) -> dict:
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx or not fx.provider_fixture_id:
        raise HTTPException(status_code=404, detail="Fixture not found")

    pfx = int(fx.provider_fixture_id)
    fx_json = get_fixture(pfx)
    if not fx_json.get("response"):
        raise HTTPException(status_code=404, detail="Fixture detail not found")
    fr = fx_json["response"][0]

    league_id = fr["league"]["id"]
    season = int(fr["league"]["season"])
    home_id = fr["teams"]["home"]["id"]
    away_id = fr["teams"]["away"]["id"]
    league_name_lc = (fr["league"]["name"] or "").strip().lower()

    # -----------------------
    # choose correct stat-block
    # -----------------------
    def _pick_block(stat_blocks: list[dict]) -> dict | None:
        # prefer: exact league.id
        for s in stat_blocks or []:
            lg = (s.get("league", {}) or {})
            if int(lg.get("id") or 0) == int(league_id):
                return s
        # then: exact league name
        for s in stat_blocks or []:
            lg = (s.get("league", {}) or {})
            if (lg.get("name") or "").strip().lower() == league_name_lc:
                return s
        return None

    # -----------------------
    # flatten all players → include season totals
    # -----------------------
    def _flatten(items: list[dict]) -> list[dict]:
        out = []

        for row in items or []:
            player = row.get("player", {}) or {}
            stats_all = row.get("statistics") or []

            # ---- collect full-season totals across all competitions ----
            season_minutes = 0
            season_apps = 0

            for block in stats_all:
                lg = block.get("league") or {}
                if int(lg.get("season") or 0) != season:
                    continue

                gms = block.get("games") or {}
                mins = int(gms.get("minutes") or 0)
                apps = int(gms.get("appearences") or gms.get("appearances") or 0)

                season_minutes += mins
                season_apps += apps

            # ---- pick correct competition block for per-90 stats ----
            s = _pick_block(stats_all)
            if not s:
                continue

            games = s.get("games", {}) or {}
            shots = s.get("shots", {}) or {}
            cards = s.get("cards", {}) or {}
            fouls = s.get("fouls", {}) or {}

            comp_minutes = int(games.get("minutes") or 0)
            sh_total = int(shots.get("total") or 0)
            sh_on = int(shots.get("on") or 0)
            fouls_comm = int((fouls.get("committed") or 0) or 0)
            # ✅ NEW: fouls drawn
            fouls_drawn = int((fouls.get("drawn") or 0) or 0)

            per90 = (
                lambda v: round((v * 90.0) / comp_minutes, 2)
                if comp_minutes
                else 0.0
            )

            out.append(
                {
                    "id": player.get("id"),
                    "name": player.get("name"),
                    "photo": player.get("photo"),
                    "pos": games.get("position") or player.get("position") or "?",

                    # competition minutes (single league/cup)
                    "minutes": comp_minutes,

                    # 🔥 full season totals (used for projected minutes)
                    "season_stats": {
                        "apps": season_apps,
                        "minutes": season_minutes,
                    },

                    # raw stats
                    "shots": sh_total,
                    "shots_on": sh_on,
                    "yellow": int(cards.get("yellow") or 0),
                    "red": int(cards.get("red") or 0),
                    "fouls_committed": fouls_comm,
                    "fouls_drawn": fouls_drawn,          # ✅ NEW

                    # per-90 from competition
                    "shots_per90": per90(sh_total),
                    "fouls_committed_per90": per90(fouls_comm),
                    "fouls_drawn_per90": per90(fouls_drawn),  # ✅ NEW
                }
            )

        # sort by meaningful players (season minutes first)
        out.sort(
            key=lambda r: (r["season_stats"]["minutes"], r["shots"], r["yellow"]),
            reverse=True,
        )
        return out

        
        # -----------------------
    # normalization fallback (CACHE-ONLY)
    # -----------------------
    def normalize_team(team_id: int) -> list[dict]:
        """
        Build a normalized roster for this team using ONLY the cached
        team-season players. We never hit /v3/players directly here – that
        should be primed via /football/admin/prime-players.
        """

        # 1) current season, all competitions (cached)
        rows = get_team_season_players_cached(db, team_id, season) or []
        flat = _flatten(rows if isinstance(rows, list) else [])
        if any(r["minutes"] > 0 for r in flat):
            return flat

        # 2) previous season, all competitions (cached)
        prev_rows = get_team_season_players_cached(db, team_id, season - 1) or []
        flat_prev = _flatten(prev_rows if isinstance(prev_rows, list) else [])
        if any(r["minutes"] > 0 for r in flat_prev):
            return flat_prev

        # 3) last resort: return whatever we have (even if minutes == 0),
        # but do NOT call the live provider.
        return flat or flat_prev

    # -----------------------
    # return payload
    # -----------------------
    return {
        "league_id": league_id,
        "season": season,
        "home_team": fx.home_team,
        "away_team": fx.away_team,
        "home": normalize_team(home_id),
        "away": normalize_team(away_id),
    }
# ---------------------------------------------------------------------------
# Injury utils (dedupe & ranking)
# ---------------------------------------------------------------------------

import re
_NAME_WORDS = re.compile(r"[^a-z]")


def _norm_name(name: str | None) -> str:
    if not name:
        return ""
    s = name.strip()
    parts = [p for p in _NAME_WORDS.sub(" ", s.lower()).split() if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    first, last = parts[0], parts[-1]
    return f"{first[0]}-{last}"


_SEVERITY_RANK = {
    "out": 4,
    "missing fixture": 4,
    "injured": 3,
    "injury": 3,
    "suspended": 3,
    "doubtful": 2,
    "questionable": 2,
    "minor": 1,
}


def _rank_row(row: dict) -> tuple:
    p = (row or {}).get("player") or {}
    name = p.get("name") or ""
    reason = (p.get("reason") or "").strip()
    typ = (p.get("type") or "").strip()
    sev = _SEVERITY_RANK.get(typ.lower(), 0)
    has_reason = 1 if reason else 0
    name_len = len(name)
    has_photo = 1 if p.get("photo") else 0
    return (has_reason, sev, name_len, has_photo)


def _merge_rows(a: dict, b: dict) -> dict:
    if _rank_row(b) > _rank_row(a):
        a, b = b, a
    ap, bp = (a.get("player") or {}), (b.get("player") or {})
    if not ap.get("reason") and bp.get("reason"):
        ap["reason"] = bp["reason"]
    if not ap.get("type") and bp.get("type"):
        ap["type"] = bp["type"]
    if not ap.get("photo") and bp.get("photo"):
        ap["photo"] = bp["photo"]
    if isinstance(ap.get("type"), str):
        t = ap["type"].strip()
        ap["type"] = t[:1].upper() + t[1:].lower()
    if isinstance(ap.get("reason"), str):
        r = ap["reason"].strip()
        ap["reason"] = r[:1].upper() + r[1:]
    a["player"] = ap
    return a


def _dedupe_injuries(items):
    if not isinstance(items, list):
        items = []
    buckets: dict[str, dict] = {}
    for row in items:
        p = (row or {}).get("player") or {}
        pid = p.get("id")
        key = f"id:{pid}" if pid is not None else f"name:{_norm_name(p.get('name')) or 'unknown'}"
        if key in buckets:
            buckets[key] = _merge_rows(buckets[key], row)
        else:
            buckets[key] = {**row, "player": {**p}}
    out = list(buckets.values())
    out.sort(key=lambda x: ((x.get("player") or {}).get("name") or "").lower())
    return out


# ---------------------------------------------------------------------------
# Statistics helpers for /fixtures/statistics
# ---------------------------------------------------------------------------

def _read_stat_value(stats_list, *aliases) -> float:
    """Find numeric value in /fixtures/statistics by 'type' label; handles '52%' strings."""
    if not isinstance(stats_list, list):
        return 0.0
    lowers = [a.lower() for a in aliases]
    for item in stats_list:
        typ = (item.get("type") or "").lower()
        if any(a in typ for a in lowers):
            v = item.get("value")
            try:
                return float(str(v).strip("%"))
            except Exception:
                return 0.0
    return 0.0


def get_team_attack_avgs(
    team_id: int,
    *,
    season: int,
    league_id: int | None = None,
    lookback: int = 5,
) -> dict:
    """
    Rolling per-match averages for this team from /fixtures/statistics:
    total shots, shots on target, corners, cards (yellow+red), expected_goals.
    """
    recent = get_team_recent_results(
        team_id, season=season, limit=lookback, league_id=league_id
    ) or []
    used = 0
    agg = dict(shots_for=0.0, sot_for=0.0, corners=0.0, cards=0.0, xg=0.0)

    for m in recent:
        fid = m.get("fixture_id")
        if not fid:
            continue
        j = get_fixture_statistics(int(fid)) or {}
        resp = j.get("response") or []
        if not isinstance(resp, list) or not resp:
            continue

        me = next((r for r in resp if (r.get("team") or {}).get("id") == team_id), None)
        if not me:
            continue
        stats = me.get("statistics") or []

        shots_total = _read_stat_value(stats, "Total Shots", "Shots total")
        shots_on = _read_stat_value(stats, "Shots on Goal", "Shots on Target")
        corners = _read_stat_value(stats, "Corner Kicks", "Corners")
        yellow = _read_stat_value(stats, "Yellow Cards")
        red = _read_stat_value(stats, "Red Cards")
        xg = _read_stat_value(stats, "expected_goals", "xg")

        agg["shots_for"] += shots_total
        agg["sot_for"] += shots_on
        agg["corners"] += corners
        agg["cards"] += (yellow or 0.0) + (red or 0.0)
        agg["xg"] += xg
        used += 1

    d = max(1, used)
    return {
        "matches_counted": used,
        "shots_for": round(agg["shots_for"] / d, 3),
        "sot_for": round(agg["sot_for"] / d, 3),
        "corners": round(agg["corners"] / d, 3),
        "cards": round(agg["cards"] / d, 3),
        "xg": round(agg["xg"] / d, 3),
    }


# ---------------------------------------------------------------------------
# NEW: model-prob helpers for preview
# ---------------------------------------------------------------------------

def _latest_probs_for_fixture(
    db: Session, fixture_id: int, source: str = "team_form"
) -> dict[str, float]:
    """Fetch latest ModelProb rows for this fixture/source and return {market: prob}."""
    sub = (
        db.query(ModelProb.market, func.max(ModelProb.as_of).label("as_of"))
        .filter(and_(ModelProb.fixture_id == fixture_id, ModelProb.source == source))
        .group_by(ModelProb.market)
        .subquery()
    )
    rows = (
        db.query(ModelProb.market, ModelProb.prob)
        .join(sub, and_(ModelProb.market == sub.c.market, ModelProb.as_of == sub.c.as_of))
        .all()
    )
    return {m: float(p) for (m, p) in rows if p is not None}


def _fair(p: float | None) -> float | None:
    if p is None or p <= 0.0 or p >= 1.0:
        return None
    return 1.0 / p


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/lineups")
def lineups(fixture_id: int, db: Session = Depends(get_db)):
    pfx = resolve_provider_fixture_id(db, fixture_id)
    try:
        return get_lineups(pfx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/players")
def players(fixture_id: int, db: Session = Depends(get_db)):
    """
    Match-level player stats for this fixture, using the cached
    /fixtures/players endpoint (primed via /football/admin/prime-players).
    Shape matches the old route: players.home / players.away are arrays of
    { player: {...}, statistics: [...] } rows.
    """
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx or not fx.provider_fixture_id:
        raise HTTPException(status_code=404, detail="Fixture not found")

    try:
        pfx = int(fx.provider_fixture_id)

        # 1) Fixture details – to know which provider team ID is home/away
        fjson = get_fixture(pfx) or {}
        core = (fjson.get("response") or [None])[0] or {}
        teams = core.get("teams") or {}
        home_pid = int((teams.get("home") or {}).get("id") or 0)
        away_pid = int((teams.get("away") or {}).get("id") or 0)

        if not (home_pid and away_pid):
            raise HTTPException(
                status_code=502, detail="Missing provider team IDs for fixture"
            )

        # 2) Cached fixture players
        pj = get_fixture_players_cached(db, pfx) or {}
        resp = pj.get("response") or []
        if not isinstance(resp, list) or not resp:
            raise HTTPException(
                status_code=404, detail="No fixture players data available"
            )

        home_players: list[dict] = []
        away_players: list[dict] = []

        for side in resp:
            team_block = side.get("team") or {}
            tid = int(team_block.get("id") or 0)
            plist = side.get("players") or []

            if tid == home_pid:
                home_players.extend(plist)
            elif tid == away_pid:
                away_players.extend(plist)
            else:
                # if something weird happens, just ignore extra teams
                continue

        return {
            "source": "API-Football (cached /fixtures/players)",
            "fixture_id": fixture_id,
            "league_id": (core.get("league") or {}).get("id"),
            "season": (core.get("league") or {}).get("season"),
            "home_team": fx.home_team,
            "away_team": fx.away_team,
            "players": {
                "home": home_players,
                "away": away_players,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Players fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/season-players")
def season_players(fixture_id: int, db: Session = Depends(get_db)):
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).one_or_none()
    if not fx or not fx.provider_fixture_id:
        raise HTTPException(status_code=404, detail="Fixture not found")

    pfx = int(fx.provider_fixture_id)
    fx_resp = get_fixture(pfx) or {}
    fr = (fx_resp.get("response") or [None])[0]
    if not fr:
        raise HTTPException(status_code=404, detail="Fixture details not found")

    season = int(((fr.get("league") or {}).get("season")) or 0)
    home_id = int((((fr.get("teams") or {}).get("home") or {}).get("id")) or 0)
    away_id = int((((fr.get("teams") or {}).get("away") or {}).get("id")) or 0)

    if not (season and home_id and away_id):
        raise HTTPException(status_code=400, detail="Fixture missing season/team ids")

    # ✅ cached pulls (raw API-Football team-season players)
    home_rows = get_team_season_players_cached(db, home_id, season) or []
    away_rows = get_team_season_players_cached(db, away_id, season) or []

    def _build_player_view(rows: list[dict]) -> list[dict]:
        out: list[dict] = []

        def n(x):
            try:
                return int(x or 0)
            except Exception:
                return 0

        for row in rows:
            player = (row.get("player") or {}) or {}
            stats_all = row.get("statistics") or []

            comp_blocks: list[dict] = []
            apps = minutes = goals = assists = shots = shots_on = yellow = red = 0

            for s in stats_all:
                lg_s = (s.get("league") or {}) or {}
                if int(lg_s.get("season") or 0) != season:
                    continue

                games = (s.get("games") or {}) or {}
                goals_s = (s.get("goals") or {}) or {}
                shots_s = (s.get("shots") or {}) or {}
                cards_s = (s.get("cards") or {}) or {}

                comp_blocks.append(
                    {
                        "league_id": lg_s.get("id"),
                        "league": lg_s.get("name"),
                        "games": games,
                        "goals": goals_s,
                        "assists": (goals_s or {}).get("assists"),
                        "shots": shots_s,
                        "cards": cards_s,
                        "minutes": games.get("minutes") or 0,
                    }
                )

                apps      += n(games.get("appearences") or games.get("appearances"))
                minutes   += n(games.get("minutes"))
                goals     += n(goals_s.get("total"))
                assists   += n((goals_s or {}).get("assists"))
                shots     += n(shots_s.get("total"))
                shots_on  += n(shots_s.get("on"))
                yellow    += n(cards_s.get("yellow"))
                red       += n(cards_s.get("red"))

            totals = {
                "apps": apps,
                "minutes": minutes,
                "goals": goals,
                "assists": assists,
                "shots": shots,
                "shots_on": shots_on,
                "yellow": yellow,
                "red": red,
            }

            out.append(
                {
                    "player": player,
                    "total": totals,
                    "competitions": comp_blocks,
                }
            )

        return out

    return {
        "source": "API-Football (cached, season totals)",
        "fixture_id": fixture_id,
        "season": season,
        "home_team": fx.home_team,
        "away_team": fx.away_team,
        "players": {
            "home": _build_player_view(home_rows),
            "away": _build_player_view(away_rows),
        },
    }
@router.get("/player/summary")
def player_summary(
    fixture_id: int,
    player_id: int,
    lookback: int = Query(10, ge=3, le=20),
    db: Session = Depends(get_db),
):
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).one_or_none()
    if not fx or not fx.provider_fixture_id:
        raise HTTPException(status_code=404, detail="Fixture not found")

    pfx = int(fx.provider_fixture_id)
    fjson = get_fixture(pfx)
    core = (fjson.get("response") or [None])[0] or {}
    lg = core.get("league") or {}
    season = int(lg.get("season") or 0)
    league_id = int(lg.get("id") or 0)
    home_id = int(((core.get("teams") or {}).get("home") or {}).get("id") or 0)
    away_id = int(((core.get("teams") or {}).get("away") or {}).get("id") or 0)

    # ---------- pull ALL season rows for both teams (cached, all comps) ----------
    home_rows = get_team_season_players_cached(db, home_id, season) or []
    away_rows = get_team_season_players_cached(db, away_id, season) or []
    season_rows = home_rows + away_rows

    # ---------- locate this player across all competitions ----------
    player_rows = [
        r
        for r in season_rows
        if int((r.get("player") or {}).get("id") or 0) == int(player_id)
    ]

    # ---------- season competitions + totals for this player ----------
    comp_blocks: list[dict] = []
    for row in player_rows:
        for s in (row.get("statistics") or []):
            lg_s = (s.get("league") or {})
            if int(lg_s.get("season") or 0) != season:
                continue
            comp_blocks.append(
                {
                    "league_id": lg_s.get("id"),
                    "league": lg_s.get("name"),
                    "games": s.get("games") or {},
                    "goals": s.get("goals") or {},
                    "assists": (s.get("goals") or {}).get("assists"),
                    "shots": s.get("shots") or {},
                    "cards": s.get("cards") or {},
                    "minutes": (s.get("games") or {}).get("minutes") or 0,
                }
            )

    totals = None
    if comp_blocks:
        def n(x):
            try:
                return int(x or 0)
            except Exception:
                return 0

        totals = {
            "apps": sum(n((cb["games"] or {}).get("appearences")) for cb in comp_blocks),
            "minutes": sum(n(cb["minutes"]) for cb in comp_blocks),
            "goals": sum(n((cb["goals"] or {}).get("total")) for cb in comp_blocks),
            "assists": sum(n(cb.get("assists")) for cb in comp_blocks),
            "shots": sum(n((cb["shots"] or {}).get("total")) for cb in comp_blocks),
            "shots_on": sum(n((cb["shots"] or {}).get("on")) for cb in comp_blocks),
            "yellow": sum(n((cb["cards"] or {}).get("yellow")) for cb in comp_blocks),
            "red": sum(n((cb["cards"] or {}).get("red")) for cb in comp_blocks),
        }

    # ---------- helper: per-90 snapshot using season totals if needed ----------
    def _per90_from_stats(pl_row: dict, stats: dict) -> dict:
        """
        Produce stable per-90 values using:
          • this competition's stats when minutes >= 30
          • season totals fallback when minutes are tiny
        """
        games = stats.get("games") or {}
        shots = stats.get("shots") or {}
        fouls = stats.get("fouls") or {}
        cards = stats.get("cards") or {}

        mins = int(games.get("minutes") or 0)

        # position: prefer games.position, then player.position, then "?"
        pos = (
            games.get("position")
            or (pl_row.get("player") or {}).get("position")
            or "?"
        )

        # If comp minutes are very low but we have season totals → use those
        use_season = bool(totals) and mins < 30 and int(totals.get("minutes") or 0) > 0

        if use_season:
            mins_use = int(totals.get("minutes") or 0)
            sh_total = int(totals.get("shots") or 0)
            sh_on = int(totals.get("shots_on") or 0)
            fouls_c = 0  # we don't currently aggregate fouls in totals
            yc = int(totals.get("yellow") or 0)
        else:
            mins_use = mins
            sh_total = int(shots.get("total") or 0)
            sh_on = int(shots.get("on") or 0)
            fouls_c = int(fouls.get("committed") or 0)
            yc = int(cards.get("yellow") or 0)

        def p90(v: int) -> float:
            return round((v * 90.0) / mins_use, 2) if mins_use > 0 else 0.0

        return {
            "id": int((pl_row.get("player") or {}).get("id") or 0),
            "name": (pl_row.get("player") or {}).get("name"),
            "photo": (pl_row.get("player") or {}).get("photo"),
            "pos": pos,
            "minutes": mins_use,
            "shots_per90": p90(sh_total),
            "sot_per90": p90(sh_on),
            "fouls_per90": p90(fouls_c),
            "yc_per90": p90(yc),
        }

    # ---------- match-context block: prefer stats whose league.id matches fixture ----------
    match_block = None
    for row in player_rows:
        for s in (row.get("statistics") or []):
            lg_s = (s.get("league") or {})  # e.g. League One, UCL, cups etc.
            if (
                int(lg_s.get("id") or 0) == league_id
                and int(lg_s.get("season") or 0) == season
            ):
                match_block = _per90_from_stats(row, s)
                break
        if match_block:
            break

    # fallback: first available stats if no comp matches fixture league
    if not match_block and player_rows:
        first_stats = (player_rows[0].get("statistics") or [])
        if first_stats:
            match_block = _per90_from_stats(player_rows[0], first_stats[0])

    # ---------- Team label (best-effort) ----------
    team_name = None
    if player_rows:
        t = ((player_rows[0].get("statistics") or [None])[0] or {}).get("team") or {}
        team_name = t.get("name")

    return {
        "player": (
            {
                "id": match_block.get("id") if match_block else player_id,
                "name": match_block.get("name")
                if match_block
                else (
                    (player_rows[0].get("player") or {}).get("name")
                    if player_rows
                    else None
                ),
                "photo": match_block.get("photo")
                if match_block
                else (
                    (player_rows[0].get("player") or {}).get("photo")
                    if player_rows
                    else None
                ),
                "position": match_block.get("pos") if match_block else None,
            }
            if match_block or player_rows
            else {"id": player_id}
        ),
        "team": {
            "home_id": home_id,
            "away_id": away_id,
            "name": team_name or fx.home_team,  # fallback
        },
        "match_stats": match_block or {},
        "season_stats": {
            "competitions": comp_blocks,
            "totals": totals,
        },
    }

@router.get("/injuries")
def injuries(fixture_id: int, db: Session = Depends(get_db)):
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx or not fx.provider_fixture_id:
        raise HTTPException(status_code=404, detail="Fixture not found")

    try:
        pfx = int(fx.provider_fixture_id)
        fx_resp = get_fixture(pfx)
        resp_list = (fx_resp or {}).get("response") or []
        if not isinstance(resp_list, list) or not resp_list:
            raise HTTPException(status_code=404, detail="Fixture details not found")

        fr = resp_list[0]
        league_id = ((fr.get("league") or {}).get("id")) or None
        season = ((fr.get("league") or {}).get("season")) or 2025
        home_id = ((fr.get("teams") or {}).get("home") or {}).get("id")
        away_id = ((fr.get("teams") or {}).get("away") or {}).get("id")

        if not (home_id and away_id and league_id and season):
            raise HTTPException(
                status_code=400, detail="Fixture missing team/league/season ids"
            )

        raw_home = get_injuries(int(home_id), int(league_id), int(season)) or {}
        raw_away = get_injuries(int(away_id), int(league_id), int(season)) or {}

        home_list = _dedupe_injuries(
            (raw_home.get("response") or []) if isinstance(raw_home, dict) else []
        )
        away_list = _dedupe_injuries(
            (raw_away.get("response") or []) if isinstance(raw_away, dict) else []
        )

        return {
            "source": "API-Football",
            "fixture_id": fixture_id,
            "league_id": league_id,
            "season": season,
            "home_team": fx.home_team,
            "away_team": fx.away_team,
            "injuries": {
                "home": home_list,
                "away": away_list,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Injuries fetch failed: {e}")


@router.get("/team-fouls")
def team_fouls(
    team_id: int,
    season: int,
    league_id: int | None = Query(None),
    lookback: int = Query(5, ge=1, le=10),
    db: Session = Depends(get_db),
):
    """
    Per-team fouls profile (last N matches):

      • events-first averages from /fixtures/events
      • plus debug per-fixture breakdown
      • plus provider team-stats fallback (season totals) when league_id is given

    Returns:
      {
        "team_id": ...,
        "season": ...,
        "league_id": ...,
        "lookback": ...,
        "averages": {
          "drawn_per_match": float,
          "committed_per_match": float,
          "source": "events-first with team-stats fallback",
        },
        "events_seen_any": bool,
        "recent_fixtures": [ ... ],
        "stats_fallback": { ... } | null
      }
    """
    # primary averages from our events-based helpers
    drawn_avg = get_team_fouls_drawn_avg(
        team_id, season=season, league_id=league_id, lookback=lookback
    )
    committed_avg = get_team_fouls_committed_avg(
        team_id, season=season, league_id=league_id, lookback=lookback
    )

    # recent fixtures for debug
    recent = get_team_recent_results(
        team_id, season=season, limit=lookback, league_id=league_id
    ) or []

    fixtures_debug: list[dict] = []
    events_seen_any = False

    for m in recent:
        fid = m.get("fixture_id")
        if not fid:
            fixtures_debug.append(
                {
                    "fixture_id": None,
                    "date": m.get("date"),
                    "opponent": m.get("opponent"),
                    "is_home": m.get("is_home"),
                    "events_available": False,
                    "fouls_committed_by_team": None,
                    "fouls_committed_by_opp": None,
                }
            )
            continue

        ev = get_events(int(fid))
        ev_rows = ev.get("response", []) if isinstance(ev, dict) else []
        fouls_by_team = 0
        fouls_by_opp = 0

        if isinstance(ev_rows, list) and ev_rows:
            events_seen_any = True
            for e in ev_rows:
                if e.get("type") != "Foul":
                    continue
                t_id = (e.get("team") or {}).get("id")
                if t_id == team_id:
                    fouls_by_team += 1
                elif t_id is not None:
                    fouls_by_opp += 1

        fixtures_debug.append(
            {
                "fixture_id": fid,
                "date": m.get("date"),
                "opponent": m.get("opponent"),
                "is_home": m.get("is_home"),
                "events_available": bool(ev_rows),
                "fouls_committed_by_team": fouls_by_team if ev_rows else None,
                "fouls_committed_by_opp": fouls_by_opp if ev_rows else None,
            }
        )

    # provider team-stats fallback (season totals)
    stats_fallback = None
    if league_id is not None:
        try:
            s = _get_team_stats_cached(db, team_id, league_id, season) or {}
            r = s.get("response") or {}

            fouls = (r.get("fouls") or {}) or {}
            fixtures_total = int(
                ((r.get("fixtures") or {}).get("played") or {}).get("total") or 0
            )
            drawn_total = int((fouls.get("drawn") or {}).get("total") or 0)
            committed_total = int((fouls.get("committed") or {}).get("total") or 0)

            stats_fallback = {
                "fixtures_played": fixtures_total,
                "drawn_total": drawn_total,
                "committed_total": committed_total,
                "drawn_per_match": (
                    drawn_total / fixtures_total if fixtures_total else 0.0
                ),
                "committed_per_match": (
                    committed_total / fixtures_total if fixtures_total else 0.0
                ),
            }
        except Exception:
            stats_fallback = None

    return {
        "team_id": team_id,
        "season": season,
        "league_id": league_id,
        "lookback": lookback,
        "averages": {
            "drawn_per_match": round(drawn_avg, 3),
            "committed_per_match": round(committed_avg, 3),
            "source": "events-first with team-stats fallback",
        },
        "events_seen_any": events_seen_any,
        "recent_fixtures": fixtures_debug,
        "stats_fallback": stats_fallback,
    }


@router.get("/team-shots")
def team_shots(
    team_id: int,
    season: int,
    league_id: int | None = Query(None),
    lookback: int = Query(5, ge=1, le=10),
):
    data = get_team_shots_against_avgs(
        team_id=team_id,
        season=season,
        league_id=league_id,
        lookback=lookback,
    )
    return {
        "team_id": team_id,
        "season": season,
        "league_id": league_id,
        "lookback": lookback,
        **data,
    }
def _season_profile_all_comps(team_id: int, season: int, lookback: int = 60) -> dict:
    """
    Build a season profile for this team using ALL competitions in the given season.
    Uses get_team_recent_results with league_id=None and a big lookback (e.g. 60)
    and then pulls scores via /fixtures to get GF/GA + W/D/L.
    """
    recent = get_team_recent_results(
        team_id,
        season=season,
        limit=lookback,   # big enough to cover a full season in most leagues
        league_id=None,   # ✅ all competitions
    ) or []

    played = wins = draws = losses = 0
    gf = ga = 0

    for m in recent:
        fid = m.get("fixture_id")
        if not fid:
            continue

        try:
            fdet = get_fixture(int(fid)) or {}
            fr = (fdet.get("response") or [None])[0] or {}
            goals = fr.get("goals") or {}

            gh = int(goals.get("home") or 0)
            ga_ = int(goals.get("away") or 0)
            is_home = bool(m.get("is_home"))

            # orient score from this team's POV
            if is_home:
                gf_match, ga_match = gh, ga_
            else:
                gf_match, ga_match = ga_, gh

            played += 1
            gf += gf_match
            ga += ga_match

            if gf_match > ga_match:
                wins += 1
            elif gf_match == ga_match:
                draws += 1
            else:
                losses += 1

        except Exception:
            # if any fixture call fails, just skip that match
            continue

    if played <= 0:
        return {
            "played_total": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "gf": 0,
            "ga": 0,
            "avg_gf": 0.0,
            "avg_ga": 0.0,
        }

    return {
        "played_total": played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "gf": gf,
        "ga": ga,
        "avg_gf": round(gf / played, 3),
        "avg_ga": round(ga / played, 3),
    }

@router.get("/team-stats")
def team_stats(
    fixture_id: int = Query(..., description="Internal fixture.id"),
    refresh: bool = Query(False, description="Force refresh from provider"),
    db: Session = Depends(get_db),
):
    """
    Return API-Football team statistics for both sides of a fixture (with DB caching).

    Shape:
      {
        fixture_id, provider_fixture_id, league_id, season,
        home_team, away_team,
        home: { ...provider response... },          # league-only provider stats
        away: { ...provider response... },
        summary: { home: {...}, away: {...} },      # league-only subset
        season_all_comps: {                         # ✅ ALL competitions, this season
          home: { played_total, wins, draws, losses, gf, ga, avg_gf, avg_ga },
          away: { ... }
        }
      }
    """
    # resolve provider IDs
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx or not fx.provider_fixture_id:
        raise HTTPException(status_code=404, detail="Fixture not found")
    pfx = int(fx.provider_fixture_id)

    fjson = get_fixture(pfx) or {}
    core = (fjson.get("response") or [None])[0] or {}
    lg = core.get("league") or {}
    league_id = int(lg.get("id") or 0)
    season = int(lg.get("season") or 0)

    teams = core.get("teams") or {}
    home_pid = int(((teams.get("home") or {}).get("id")) or 0)
    away_pid = int(((teams.get("away") or {}).get("id")) or 0)

    if not (league_id and season and home_pid and away_pid):
        raise HTTPException(
            status_code=502,
            detail="Missing league/season/team ids from provider",
        )

    # fetch (cached) team-season stats for this league
    home_json = (
        _get_team_stats_cached(db, home_pid, league_id, season, refresh=refresh) or {}
    )
    away_json = (
        _get_team_stats_cached(db, away_pid, league_id, season, refresh=refresh) or {}
    )

    # provider puts payload under 'response' – keep that node for UI
    home = home_json.get("response") or {}
    away = away_json.get("response") or {}

    # tiny, safe summary block your UI can use if desired (league-only)
    def _safe(v, *path, default=0.0):
        cur = v
        try:
            for k in path:
                cur = cur.get(k) if isinstance(cur, dict) else {}
            x = cur
            if isinstance(x, (int, float)):
                return float(x)
            s = str(x).strip().replace("%", "")
            return float(s) if s else default
        except Exception:
            return default

    def _summ(r):
        return {
            "played_total": int(_safe(r, "fixtures", "played", "total", default=0)),
            "wins": int(_safe(r, "fixtures", "wins", "total", default=0)),
            "draws": int(_safe(r, "fixtures", "draws", "total", default=0)),
            "losses": int(_safe(r, "fixtures", "loses", "total", default=0)),
            "gf": int(_safe(r, "goals", "for", "total", "total", default=0)),
            "ga": int(_safe(r, "goals", "against", "total", "total", default=0)),
            "avg_gf": _safe(r, "goals", "for", "average", "total", default=0.0),
            "avg_ga": _safe(r, "goals", "against", "average", "total", default=0.0),
            "form": (r.get("form") or None),
        }

    # ✅ NEW: season profile across ALL competitions
    home_all = _season_profile_all_comps(home_pid, season)
    away_all = _season_profile_all_comps(away_pid, season)

    return {
        "fixture_id": fixture_id,
        "provider_fixture_id": pfx,
        "league_id": league_id,
        "season": season,
        "home_team": fx.home_team,
        "away_team": fx.away_team,
        "home": home,
        "away": away,
        "summary": {"home": _summ(home), "away": _summ(away)},  # league-only
        "season_all_comps": {                                   # all comps
            "home": home_all,
            "away": away_all,
        },
    }

@router.get("/opponent-pace")
def opponent_pace(
    fixture_id: int,
    lookback: int = Query(5, ge=1, le=10),
    db: Session = Depends(get_db),
):
    """
    Per-fixture context (last N matches in ALL competitions, not league-only):
      - Opponent defensive profile: shots/SoT conceded per match (last N, all comps)
      - Pace multipliers for shots/SoT (vs league baselines)
      - Team attacking avgs: Shots, SoT, Corners, Cards, xG (last N, all comps)
      - Team xG for/against (last N, all comps)
      - Team fouls committed/drawn (last N, all comps)
    """
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx or not fx.provider_fixture_id:
        raise HTTPException(status_code=404, detail="Fixture not found")

    pfx = int(fx.provider_fixture_id)
    fx_json = get_fixture(pfx)
    if not fx_json.get("response"):
        raise HTTPException(status_code=404, detail="Fixture detail not found")
    fr = fx_json["response"][0]

    league_id = int(fr["league"]["id"])
    season = int(fr["league"]["season"])
    home_id = int(fr["teams"]["home"]["id"])
    away_id = int(fr["teams"]["away"]["id"])

    # 🔁 Now using ALL competitions (league_id=None) for lookback stats

    # Opponent defensive profiles (concessions)
    away_conc = get_team_shots_against_avgs(
        away_id, season=season, league_id=None, lookback=lookback
    )
    home_conc = get_team_shots_against_avgs(
        home_id, season=season, league_id=None, lookback=lookback
    )

    # Our attacking profiles (using fixtures/statistics)
    home_att = get_team_attack_avgs(
        home_id, season=season, league_id=None, lookback=lookback
    )
    away_att = get_team_attack_avgs(
        away_id, season=season, league_id=None, lookback=lookback
    )

    # xG for/against
    home_xg = get_team_xg_avgs(
        home_id, season=season, league_id=None, lookback=lookback
    )
    away_xg = get_team_xg_avgs(
        away_id, season=season, league_id=None, lookback=lookback
    )

    # fouls committed/drawn
    home_fouls = get_team_fouls_from_statistics_avg(
        home_id, season=season, league_id=None, lookback=lookback
    )
    away_fouls = get_team_fouls_from_statistics_avg(
        away_id, season=season, league_id=None, lookback=lookback
    )

    LEAGUE_AVG_SHOTS = 12.5
    LEAGUE_AVG_SOT = 4.4
    clamp = lambda x, lo, hi: max(lo, min(hi, x))

    home_ctx = {
        "opp_team_id": away_id,
        "matches_counted": int(away_conc.get("matches_counted") or 0),
        "opp_shots_against_per_match": round(
            float(away_conc.get("shots_against_per_match") or 0.0), 3
        ),
        "opp_sot_against_per_match": round(
            float(away_conc.get("sot_against_per_match") or 0.0), 3
        ),
        "pace_factor_shots": clamp(
            (away_conc.get("shots_against_per_match") or 0.0) / LEAGUE_AVG_SHOTS,
            0.85,
            1.15,
        ),
        "pace_factor_sot": clamp(
            (away_conc.get("sot_against_per_match") or 0.0) / LEAGUE_AVG_SOT,
            0.85,
            1.20,
        ),
        "attack_avgs": home_att,
        "xg_avgs": {
            "xg_for_per_match": round(float(home_xg.get("xg_for_per_match") or 0.0), 3),
            "xg_against_per_match": round(
                float(home_xg.get("xg_against_per_match") or 0.0), 3
            ),
        },
        "fouls_avgs": {
            "fouls_committed_per_match": round(
                float(home_fouls.get("fouls_committed_per_match") or 0.0), 3
            ),
            "fouls_drawn_per_match": round(
                float(home_fouls.get("fouls_drawn_per_match") or 0.0), 3
            ),
        },
    }

    away_ctx = {
        "opp_team_id": home_id,
        "matches_counted": int(home_conc.get("matches_counted") or 0),
        "opp_shots_against_per_match": round(
            float(home_conc.get("shots_against_per_match") or 0.0), 3
        ),
        "opp_sot_against_per_match": round(
            float(home_conc.get("sot_against_per_match") or 0.0), 3
        ),
        "pace_factor_shots": clamp(
            (home_conc.get("shots_against_per_match") or 0.0) / LEAGUE_AVG_SHOTS,
            0.85,
            1.15,
        ),
        "pace_factor_sot": clamp(
            (home_conc.get("sot_against_per_match") or 0.0) / LEAGUE_AVG_SOT,
            0.85,
            1.20,
        ),
        "attack_avgs": away_att,
        "xg_avgs": {
            "xg_for_per_match": round(float(away_xg.get("xg_for_per_match") or 0.0), 3),
            "xg_against_per_match": round(
                float(away_xg.get("xg_against_per_match") or 0.0), 3
            ),
        },
        "fouls_avgs": {
            "fouls_committed_per_match": round(
                float(away_fouls.get("fouls_committed_per_match") or 0.0), 3
            ),
            "fouls_drawn_per_match": round(
                float(away_fouls.get("fouls_drawn_per_match") or 0.0), 3
            ),
        },
    }

    return {
        "fixture_id": fixture_id,
        "league_id": league_id,
        "season": season,
        "home_team": fx.home_team,
        "away_team": fx.away_team,
        "home_context": home_ctx,
        "away_context": away_ctx,
    }


@router.get("/player-props/fair")
def player_props_fair(
    fixture_id: int,
    team: str | None = Query(None, description="home|away"),
    markets: str | None = Query(None, description="CSV of markets"),
    min_prob: float = Query(0.0, ge=0.0, le=1.0),
    minutes: int | None = Query(None, ge=10, le=120),
    opponent_adj: bool = Query(True, description="apply opponent fouls/pace bumps"),
    ref_adj: bool = Query(True, description="apply referee cards bump (if known)"),
    ref_factor_override: float | None = Query(None, ge=0.5, le=1.5),
    lookback: int = Query(5, ge=2, le=10),
    db: Session = Depends(get_db),
    user=Depends(require_premium),
):
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx:
        raise HTTPException(status_code=404, detail="Fixture not found")

    try:
        pfx = int(fx.provider_fixture_id)
    except Exception:
        pfx = None

    league_id = season = home_pid = away_pid = None
    referee_name = (fx.referee or "").strip()

    if pfx:
        try:
            fjson = get_fixture(pfx)
            core = (fjson.get("response") or [None])[0] or {}
            lg = core.get("league") or {}
            league_id = lg.get("id")
            season = lg.get("season")
            teams = core.get("teams") or {}
            home_pid = (teams.get("home") or {}).get("id")
            away_pid = (teams.get("away") or {}).get("id")
            if not referee_name:
                referee_name = (core.get("fixture") or {}).get("referee") or ""
        except Exception:
            pass

    # events-first fouls context
    home_drawn90 = (
        get_team_fouls_drawn_avg(
            home_pid, season=season, league_id=league_id, lookback=lookback
        )
        if home_pid
        else 0.0
    )
    away_drawn90 = (
        get_team_fouls_drawn_avg(
            away_pid, season=season, league_id=league_id, lookback=lookback
        )
        if away_pid
        else 0.0
    )
    home_comm90 = (
        get_team_fouls_committed_avg(
            home_pid, season=season, league_id=league_id, lookback=lookback
        )
        if home_pid
        else 0.0
    )
    away_comm90 = (
        get_team_fouls_committed_avg(
            away_pid, season=season, league_id=league_id, lookback=lookback
        )
        if away_pid
        else 0.0
    )

    # shots/SoT conceded pace
    def _pace(team_id: int | None) -> tuple[float, float]:
        if not (team_id and league_id and season):
            return (0.0, 0.0)
        try:
            pa = (
                get_team_shots_against_avgs(
                    team_id, season=season, league_id=league_id, lookback=lookback
                )
                or {}
            )
            return float(pa.get("shots_against_per_match") or 0.0), float(
                pa.get("sot_against_per_match") or 0.0
            )
        except Exception:
            return (0.0, 0.0)

    away_opp_shotsA, away_opp_sotA = _pace(away_pid)  # affects HOME side
    home_opp_shotsA, home_opp_sotA = _pace(home_pid)  # affects AWAY side

    LEAGUE_AVG_SHOTS_AGAINST = 12.5
    LEAGUE_AVG_SOT_AGAINST = 4.4
    clamp = lambda x, lo, hi: max(lo, min(hi, x))

    def opponent_fouls_factor(opponent_drawn90: float) -> float:
        if not opponent_adj:
            return 1.0
        if opponent_drawn90 <= 0:
            return 1.0
        return clamp(opponent_drawn90 / 10.0, 0.80, 1.25)

    def pace_factor(opp_rate: float, league_avg: float, lo=0.80, hi=1.25) -> float:
        if not opponent_adj:
            return 1.0
        if opp_rate <= 0 or league_avg <= 0:
            return 1.0
        return clamp(opp_rate / league_avg, lo, hi)

    def referee_cards_factor() -> float:
        if not ref_adj:
            return 1.0
        if ref_factor_override is not None:
            return float(ref_factor_override)
        return 1.0

    stats_data = _get_player_props_data(fixture_id, db)

    # Best available odds (by player, market, line)
    stored_odds = (
        db.query(PlayerOdds).filter(PlayerOdds.fixture_id == fixture_id).all()
    )
    odds_map: dict[tuple[int, str, float], dict] = {}
    for o in stored_odds:
        key = (int(o.player_id), (o.market or "").lower(), float(o.line or 0.0))
        best = odds_map.get(key)
        if not best or float(o.price) > best["price"]:
            odds_map[key] = {"bookmaker": o.bookmaker, "price": float(o.price)}

    team_norm = (team or "").strip().lower()
    want_team = team_norm in {"home", "away"}
    market_set = set(
        m.strip().lower() for m in (markets or "").split(",") if m and m.strip()
    )

    out = {"fixture_id": fixture_id, "props": []}

    for side in ("home", "away"):
        if want_team and side != team_norm:
            continue

        roster = stats_data.get(side, []) or []

        if side == "home":
            fouls_ctx = opponent_fouls_factor(away_drawn90)  # for committed
            fouls_drawn_ctx = opponent_fouls_factor(away_comm90)  # for drawn
            shots_ctx = pace_factor(away_opp_shotsA, LEAGUE_AVG_SHOTS_AGAINST)
            sot_ctx = pace_factor(away_opp_sotA, LEAGUE_AVG_SOT_AGAINST)
            opp_drawn90 = away_drawn90
            opp_comm90 = away_comm90
            opp_shotsA, opp_sotA = away_opp_shotsA, away_opp_sotA
        else:
            fouls_ctx = opponent_fouls_factor(home_drawn90)
            fouls_drawn_ctx = opponent_fouls_factor(home_comm90)
            shots_ctx = pace_factor(home_opp_shotsA, LEAGUE_AVG_SHOTS_AGAINST)
            sot_ctx = pace_factor(home_opp_sotA, LEAGUE_AVG_SOT_AGAINST)
            opp_drawn90 = home_drawn90
            opp_comm90 = home_comm90
            opp_shotsA, opp_sotA = home_opp_shotsA, home_opp_sotA

        ref_ctx = referee_cards_factor()

        for pl in roster:
            # ---- season-aware projected minutes ----
            mins_played = int(pl.get("minutes") or 0)

            season_stats = pl.get("season_stats") or {}
            apps = (
                season_stats.get("apps")
                or pl.get("apps")
                or pl.get("appearances")
                or 0
            )
            mins_total = (
                season_stats.get("minutes")
                or pl.get("minutes_total")
                or 0
            )

            if apps and mins_total:
                avg_mins = mins_total / apps
            elif mins_played:
                # use roster minutes as a rough signal if season totals missing
                avg_mins = 80 if mins_played >= 600 else 60
            else:
                avg_mins = 75  # generic fallback

            # clamp to realistic football range
            avg_mins = max(50, min(avg_mins, 95))

            # allow explicit override via query param
            m_used = int(minutes or avg_mins)

            # ---- per-90 inputs ----
            shots_per90 = float(pl.get("shots_per90") or 0.0)
            fouls_comm90 = float(pl.get("fouls_committed_per90") or 0.0)
            fouls_drawn90 = float(pl.get("fouls_drawn_per90") or 0.0)

            if mins_played > 0:
                sot_per90 = (float(pl.get("shots_on") or 0.0) * 90.0) / mins_played
                cards_per90 = (float(pl.get("yellow") or 0.0) * 90.0) / mins_played
            else:
                sot_per90 = 0.0
                cards_per90 = 0.0

            # Apply context bumps
            p_shots15 = prob_over_xpoint5(
                shots_per90, m_used, 1.5, opponent_factor=shots_ctx
            )
            p_sot05 = prob_over_xpoint5(
                sot_per90, m_used, 0.5, opponent_factor=sot_ctx
            )
            p_fouls_comm05 = prob_over_xpoint5(
                fouls_comm90, m_used, 0.5, opponent_factor=fouls_ctx
            )
            p_fouls_drawn05 = prob_over_xpoint5(
                fouls_drawn90, m_used, 0.5, opponent_factor=fouls_drawn_ctx
            )
            p_card = prob_card(
                cards_per90,
                m_used,
                ref_factor=ref_ctx,
                opponent_factor=fouls_ctx,
            )

            markets_calc = [
                ("shots_over_1.5", 1.5, p_shots15, fair_odds(p_shots15)),
                ("sot_over_0.5", 0.5, p_sot05, fair_odds(p_sot05)),
                ("fouls_over_0.5", 0.5, p_fouls_comm05, fair_odds(p_fouls_comm05)),
                (
                    "fouls_drawn_over_0.5",
                    0.5,
                    p_fouls_drawn05,
                    fair_odds(p_fouls_drawn05),
                ),
                ("to_be_booked", 0.5, p_card, fair_odds(p_card)),
            ]

            for market, line, prob, fair in markets_calc:
                if prob < min_prob:
                    continue
                if market_set and market not in market_set:
                    continue

                key = (int(pl["id"]), market, float(line))
                bm = odds_map.get(key)

                out["props"].append(
                    {
                        "player_id": int(pl["id"]),
                        "player": html.unescape(pl.get("name") or ""),
                        "team_side": side,
                        "market": market,
                        "line": float(line),
                        "proj_minutes": m_used,
                        "prob": float(prob),
                        "fair_odds": float(fair) if fair else None,
                        "best_price": bm["price"] if bm else None,
                        "bookmaker": bm["bookmaker"] if bm else None,
                        "edge": edge(prob, bm["price"]) if bm and fair else None,
                        # context (for Why/preview UIs)
                        "per90_shots": round(shots_per90, 2),
                        "per90_fouls": round(fouls_comm90, 2),
                        "per90_fouls_drawn": round(fouls_drawn90, 2),
                        "per90_sot": round(sot_per90, 2),
                        "cards_per90": round(cards_per90, 2),
                        "opp_fouls_drawn_per90": round(opp_drawn90, 2),
                        "opp_fouls_committed_per90": round(opp_comm90, 2),
                        "opponent_factor": round(fouls_ctx, 3),
                        "ref_factor": round(ref_ctx, 3),
                        "opp_shots_against_per_match": round(opp_shotsA, 2),
                        "opp_sot_against_per_match": round(opp_sotA, 2),
                        "pace_factor_shots": round(shots_ctx, 3),
                        "pace_factor_sot": round(sot_ctx, 3),
                    }
                )

    out["props"].sort(
        key=lambda r: (float(r.get("edge") or 0.0), float(r["prob"])),
        reverse=True,
    )
    return out

@router.get("/preview")
def preview(
    fixture_id: int,
    top_n_per_team: int = Query(5, ge=1, le=10),
    minutes: int | None = Query(None, ge=10, le=120),
    opponent_adj: bool = Query(True),
    ref_adj: bool = Query(True),
    ref_factor_override: float | None = Query(None, ge=0.5, le=1.5),
    db: Session = Depends(get_db),
):
    """
    Self-consistent preview:
      - Model fairs for 1X2, Over/Under 2.5, BTTS (uses team_form if available; fallback Poisson)
      - Top-N player props (minutes-weighted) for each team
      - Narrative using opponent pace + rolling team stats (shots/SoT/corners/cards/xG + fouls)
    """
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx or not fx.provider_fixture_id:
        raise HTTPException(status_code=404, detail="Fixture not found")
    pfx = int(fx.provider_fixture_id)

    fjson = get_fixture(pfx)
    if not fjson.get("response"):
        raise HTTPException(status_code=404, detail="Fixture detail not found")
    core = fjson["response"][0]
    lg = core.get("league") or {}
    league_id = int(lg.get("id") or 0)
    season = int(lg.get("season") or 0)
    tinfo = core.get("teams") or {}
    home_pid = int((tinfo.get("home") or {}).get("id") or 0)
    away_pid = int((tinfo.get("away") or {}).get("id") or 0)
    referee_name = (core.get("fixture") or {}).get("referee") or (fx.referee or "")

    # --- Poisson backbone from team-season stats --------------------------
    home_stats = _get_team_stats_cached(db, home_pid, league_id, season) or {}
    away_stats = _get_team_stats_cached(db, away_pid, league_id, season) or {}
    hs = (home_stats.get("response") or {})
    as_ = (away_stats.get("response") or {})

    home_for_avg = _safe_float(
        (((hs.get("goals") or {}).get("for") or {}).get("average") or {}).get("total")
    )
    home_against = _safe_float(
        (((hs.get("goals") or {}).get("against") or {}).get("average") or {}).get(
            "total"
        )
    )
    away_for_avg = _safe_float(
        (((as_.get("goals") or {}).get("for") or {}).get("average") or {}).get(
            "total"
        )
    )
    away_against = _safe_float(
        (((as_.get("goals") or {}).get("against") or {}).get("average") or {}).get(
            "total"
        )
    )

    lam_home = _blend_lambda(home_for_avg, away_against, w_for=0.6)
    lam_away = _blend_lambda(away_for_avg, home_against, w_for=0.6)

    one_x_two = _match_1x2_from_poisson(lam_home, lam_away, max_goals=10)
    totals_btts = _totals_btts_from_poisson(lam_home, lam_away)
    p_home_poi, p_draw_poi, p_away_poi = (
        one_x_two["home"],
        one_x_two["draw"],
        one_x_two["away"],
    )
    ou_poi = totals_btts["over_2_5"]
    btts_poi = totals_btts["btts_yes"]

    # --- Pull latest team_form model probabilities and overlay ------------
    model = _latest_probs_for_fixture(db, fixture_id, source="team_form")
    p_home_use = model.get("HOME_WIN", p_home_poi)
    p_draw_use = model.get("DRAW", p_draw_poi)
    p_away_use = model.get("AWAY_WIN", p_away_poi)
    ou_use = model.get("O2.5", ou_poi)
    btts_use = model.get("BTTS_Y", btts_poi)
    model_source_used = "team_form" if model else "poisson"

    # --- Opponent pace (shots/SoT conceded) -------------------------------
    LOOKBACK = 5
    away_conc = get_team_shots_against_avgs(
        away_pid, season=season, league_id=league_id, lookback=LOOKBACK
    )
    home_conc = get_team_shots_against_avgs(
        home_pid, season=season, league_id=league_id, lookback=LOOKBACK
    )
    LEAGUE_AVG_SHOTS = 12.5
    LEAGUE_AVG_SOT = 4.4
    clamp = lambda x, lo, hi: max(lo, min(hi, x))

    home_ctx = {
        "opp_team_id": away_pid,
        "matches_counted": away_conc.get("matches_counted", 0),
        "opp_shots_against_per_match": away_conc.get("shots_against_per_match", 0.0),
        "opp_sot_against_per_match": away_conc.get("sot_against_per_match", 0.0),
        "pace_factor_shots": clamp(
            (away_conc.get("shots_against_per_match") or 0.0) / LEAGUE_AVG_SHOTS,
            0.85,
            1.15,
        ),
        "pace_factor_sot": clamp(
            (away_conc.get("sot_against_per_match") or 0.0) / LEAGUE_AVG_SOT,
            0.85,
            1.20,
        ),
    }
    away_ctx = {
        "opp_team_id": home_pid,
        "matches_counted": home_conc.get("matches_counted", 0),
        "opp_shots_against_per_match": home_conc.get("shots_against_per_match", 0.0),
        "opp_sot_against_per_match": home_conc.get("sot_against_per_match", 0.0),
        "pace_factor_shots": clamp(
            (home_conc.get("shots_against_per_match") or 0.0) / LEAGUE_AVG_SHOTS,
            0.85,
            1.15,
        ),
        "pace_factor_sot": clamp(
            (home_conc.get("sot_against_per_match") or 0.0) / LEAGUE_AVG_SOT,
            0.85,
            1.20,
        ),
    }

    # --- Rolling team stats for narrative (shots/xG/fouls etc.) ----------
    home_attack = get_team_attack_avgs(
        home_pid, season=season, league_id=league_id, lookback=LOOKBACK
    )
    away_attack = get_team_attack_avgs(
        away_pid, season=season, league_id=league_id, lookback=LOOKBACK
    )
    home_xg = get_team_xg_avgs(
        home_pid, season=season, league_id=league_id, lookback=LOOKBACK
    )
    away_xg = get_team_xg_avgs(
        away_pid, season=season, league_id=league_id, lookback=LOOKBACK
    )
    home_fouls = get_team_fouls_from_statistics_avg(
        home_pid, season=season, league_id=league_id, lookback=LOOKBACK
    )
    away_fouls = get_team_fouls_from_statistics_avg(
        away_pid, season=season, league_id=league_id, lookback=LOOKBACK
    )

    # --- Team fouls (committed & drawn) for context -----------------------
    home_drawn90 = float(home_fouls.get("fouls_drawn_per_match", 0.0) or 0.0)
    home_comm90 = float(home_fouls.get("fouls_committed_per_match", 0.0) or 0.0)
    away_drawn90 = float(away_fouls.get("fouls_drawn_per_match", 0.0) or 0.0)
    away_comm90 = float(away_fouls.get("fouls_committed_per_match", 0.0) or 0.0)

    def opponent_fouls_factor(opponent_drawn90: float) -> float:
        if not opponent_adj:
            return 1.0
        if opponent_drawn90 <= 0:
            return 1.0
        return clamp(opponent_drawn90 / 10.0, 0.80, 1.25)

    def referee_cards_factor() -> float:
        if not ref_adj:
            return 1.0
        if ref_factor_override is not None:
            return float(ref_factor_override)
        return 1.0

    # For committed fouls: scale by opponent fouls drawn
    # For fouls drawn: scale by opponent fouls committed
    fouls_comm_ctx_home = opponent_fouls_factor(away_drawn90)
    fouls_drawn_ctx_home = opponent_fouls_factor(away_comm90)
    fouls_comm_ctx_away = opponent_fouls_factor(home_drawn90)
    fouls_drawn_ctx_away = opponent_fouls_factor(home_comm90)
    ref_ctx = referee_cards_factor()

    # --- Player props preview (top-N by minutes) -------------------------
    pdata = _get_player_props_data(fixture_id, db)
    top_players = {"home": [], "away": []}

    for side in ("home", "away"):
        roster = sorted(
            pdata.get(side, []) or [],
            key=lambda r: r.get("minutes", 0),
            reverse=True,
        )[:top_n_per_team]

        if side == "home":
            fouls_comm_ctx = fouls_comm_ctx_home
            fouls_drawn_ctx = fouls_drawn_ctx_home
            pace_s = home_ctx["pace_factor_shots"]
            pace_t = home_ctx["pace_factor_sot"]
        else:
            fouls_comm_ctx = fouls_comm_ctx_away
            fouls_drawn_ctx = fouls_drawn_ctx_away
            pace_s = away_ctx["pace_factor_shots"]
            pace_t = away_ctx["pace_factor_sot"]

        block = []
        for pl in roster:
            mins_played = int(pl.get("minutes") or 0)

            # season-aware projected minutes (fallback 80/30 like before)
            season_stats = pl.get("season_stats") or {}
            apps = (
                season_stats.get("apps")
                or pl.get("apps")
                or pl.get("appearances")
                or 0
            )
            mins_total = (
                season_stats.get("minutes")
                or pl.get("minutes_total")
                or 0
            )

            if apps and mins_total:
                avg_mins = mins_total / apps
            elif mins_played:
                avg_mins = 80 if mins_played >= 600 else 60
            else:
                avg_mins = 75

            avg_mins = max(50, min(avg_mins, 95))
            m_used = int(minutes or avg_mins)

            shots_per90 = float(pl.get("shots_per90") or 0.0)
            fouls_comm90 = float(pl.get("fouls_committed_per90") or 0.0)
            fouls_drawn90 = float(pl.get("fouls_drawn_per90") or 0.0)
            shots_on_total = float(pl.get("shots_on") or 0.0)

            if mins_played > 0:
                sot_per90 = (shots_on_total * 90.0) / mins_played
                cards_per90 = (float(pl.get("yellow") or 0.0) * 90.0) / mins_played
            else:
                sot_per90 = 0.0
                cards_per90 = 0.0

            # Apply context bumps (pace + fouls)
            p_shots15 = prob_over_xpoint5(shots_per90 * pace_s, m_used, 1.5)
            p_sot05 = prob_over_xpoint5(sot_per90 * pace_t, m_used, 0.5)
            p_fouls_comm05 = prob_over_xpoint5(
                fouls_comm90, m_used, 0.5, opponent_factor=fouls_comm_ctx
            )
            p_fouls_drawn05 = prob_over_xpoint5(
                fouls_drawn90, m_used, 0.5, opponent_factor=fouls_drawn_ctx
            )
            p_card = prob_card(
                cards_per90,
                m_used,
                ref_factor=ref_ctx,
                opponent_factor=fouls_comm_ctx,
            )

            block.append(
                {
                    "player_id": pl.get("id"),
                    "player": html.unescape(pl.get("name") or ""),
                    "minutes": m_used,
                    "markets": {
                        "shots_over_1.5": {
                            "p": p_shots15,
                            "fair": fair_odds(p_shots15),
                        },
                        "sot_over_0.5": {
                            "p": p_sot05,
                            "fair": fair_odds(p_sot05),
                        },
                        "fouls_over_0.5": {
                            "p": p_fouls_comm05,
                            "fair": fair_odds(p_fouls_comm05),
                        },
                        "fouls_drawn_over_0.5": {
                            "p": p_fouls_drawn05,
                            "fair": fair_odds(p_fouls_drawn05),
                        },
                        "to_be_booked": {
                            "p": p_card,
                            "fair": fair_odds(p_card),
                        },
                    },
                }
            )
        top_players[side] = block

    # --- Narrative --------------------------------------------------------
    ht = fx.home_team
    at = fx.away_team
    fmt = lambda x: f"{x:.2f}"
    narrative = (
        f"{ht} vs {at}: model ({model_source_used}) favours "
        f"{ht if p_home_use > p_away_use else at} "
        f"({max(p_home_use, p_away_use) * 100:.1f}%), draw {p_draw_use * 100:.1f}%. "
        f"Over 2.5 {ou_use * 100:.1f}%, BTTS {btts_use * 100:.1f}%. "
        f"{ht} attack ~ shots {fmt(home_attack.get('shots_for', 0))}, "
        f"SoT {fmt(home_attack.get('sot_for', 0))}, corners {fmt(home_attack.get('corners', 0))}, "
        f"cards {fmt(home_attack.get('cards', 0))}, xG {fmt(home_xg.get('xg_for_per_match', 0))}; "
        f"{at} allow ~ shots {fmt(home_ctx['opp_shots_against_per_match'])} / "
        f"SoT {fmt(home_ctx['opp_sot_against_per_match'])}. "
        f"{at} attack ~ shots {fmt(away_attack.get('shots_for', 0))}, "
        f"SoT {fmt(away_attack.get('sot_for', 0))}, corners {fmt(away_attack.get('corners', 0))}, "
        f"cards {fmt(away_attack.get('cards', 0))}, xG {fmt(away_xg.get('xg_for_per_match', 0))}; "
        f"{ht} allow ~ shots {fmt(away_ctx['opp_shots_against_per_match'])} / "
        f"SoT {fmt(away_ctx['opp_sot_against_per_match'])}. Referee: {referee_name or 'tbc'}."
    )

    return {
        "fixture_id": fixture_id,
        "league_id": league_id,
        "season": season,
        "meta": {
            "home_team": ht,
            "away_team": at,
            "referee": referee_name or None,
            "lambda_home": round(lam_home, 3),
            "lambda_away": round(lam_away, 3),
            "model_source": model_source_used,
        },
        "markets": {
            "1x2": {
                "home": {"p": p_home_use, "fair": _fair(p_home_use)},
                "draw": {"p": p_draw_use, "fair": _fair(p_draw_use)},
                "away": {"p": p_away_use, "fair": _fair(p_away_use)},
            },
            "over_2_5": {"p": ou_use, "fair": _fair(ou_use)},
            "btts_yes": {"p": btts_use, "fair": _fair(btts_use)},
        },
        "top_players": top_players,
        "opponent_context": {
            "home": {
                **home_ctx,
                "attack_avgs": home_attack,
                "xg_avgs": home_xg,
                "fouls_avgs": home_fouls,
            },
            "away": {
                **away_ctx,
                "attack_avgs": away_attack,
                "xg_avgs": away_xg,
                "fouls_avgs": away_fouls,
            },
        },
        "summary": narrative,
    }
# add if it's not already there
# from ..services.apifootball import get_fixture_players

@router.get("/player/game-log")
def player_game_log(
    fixture_id: int,
    player_id: int,
    last: int = Query(5, ge=1, le=10),
    league_only: bool = Query(False, description="Limit to same league as the fixture"),
    db: Session = Depends(get_db),
):
    """
    Return last-N matches for this player (minutes, shots, SoT, goals, cards,
    fouls, rating, etc.).

    IMPORTANT: this route should NOT 404 just because fixture_players cache
    is missing for some past games. It falls back to live provider and then
    simply skips if still no data.
    """
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).one_or_none()
    if not fx or not fx.provider_fixture_id:
        raise HTTPException(status_code=404, detail="Fixture not found")

    # --- fixture context ---
    fcore = (get_fixture(int(fx.provider_fixture_id)).get("response") or [None])[0] or {}
    lg = fcore.get("league") or {}
    season = int(lg.get("season") or 0)
    league_id = int(lg.get("id") or 0)

    teams = fcore.get("teams") or {}
    home_pid = int((teams.get("home") or {}).get("id") or 0)
    away_pid = int((teams.get("away") or {}).get("id") or 0)

    # --- determine which team this player belongs to (home / away) ---
    home_rows = get_team_season_players_cached(db, home_pid, season) or []
    away_rows = get_team_season_players_cached(db, away_pid, season) or []

    belongs_home = any(
        int((r.get("player") or {}).get("id") or 0) == int(player_id)
        for r in home_rows
    )
    team_id = home_pid if belongs_home else away_pid

    # --- recent fixtures for this team ---
    recent = get_team_recent_results(
        team_id,
        season=season,
        limit=max(last * 2, last),  # ask for extra in case of DNPs
        league_id=(league_id if league_only else None),
    ) or []

    out: list[dict] = []

    for m in recent:
        fid = m.get("fixture_id")
        if not fid:
            continue

        # -------- fixture details (score, league name) --------
        score_str = None
        result = None
        league_name = None
        fresp = {}
        try:
            fdet = get_fixture(int(fid)) or {}
            fresp = (fdet.get("response") or [None])[0] or {}
            goals = (fresp.get("goals") or {})
            gh = int(goals.get("home") or 0)
            ga = int(goals.get("away") or 0)
            league_name = (fresp.get("league") or {}).get("name") or None

            if bool(m.get("is_home")):
                score_str = f"{gh}-{ga}"
                result = "W" if gh > ga else ("L" if gh < ga else "D")
            else:
                score_str = f"{ga}-{gh}"
                result = "W" if ga > gh else ("L" if ga < gh else "D")
        except Exception:
            score_str = None
            result = None
            league_name = None

        # -------- player stats from fixture: CACHE → LIVE FALLBACK --------
        pj = {}
        try:
            pj = get_fixture_players_cached(db, int(fid)) or {}
        except HTTPException as e:
            if e.status_code == 404:
                # cache miss → hit provider directly
                try:
                    pj = get_fixture_players(int(fid)) or {}
                except Exception:
                    pj = {}
            else:
                # any other HTTP error, bubble up
                raise

        resp = pj.get("response") or []
        if not isinstance(resp, list):
            resp = []

        found = False
        for side in resp:
            players_list = side.get("players") or []
            for pl in players_list:
                if int((pl.get("player") or {}).get("id") or 0) != int(player_id):
                    continue

                stats  = (pl.get("statistics") or [{}])[0]
                games  = stats.get("games")  or {}
                shots  = stats.get("shots")  or {}
                goals  = stats.get("goals")  or {}
                cards  = stats.get("cards")  or {}
                passes = stats.get("passes") or {}
                fouls  = stats.get("fouls")  or {}
                lg_s   = stats.get("league") or {}
                tm     = side.get("team")    or {}

                # opponent name (other team in response)
                opp_name = None
                for other in resp:
                    tid = (other.get("team") or {}).get("id")
                    if tid and tid != tm.get("id"):
                        opp_name = (other.get("team") or {}).get("name")
                        break

                comp_name = (
                    lg_s.get("name")
                    or league_name
                    or (fresp.get("league") or {}).get("name")
                    or "Unknown"
                )

                date_iso = m.get("date")

                out.append({
                    "fixture_id": int(fid),
                    "date": date_iso,
                    "team": html.unescape(tm.get("name") or ""),
                    "opponent": html.unescape(opp_name or ""),
                    "is_home": bool(m.get("is_home")),
                    "competition": comp_name,
                    "minutes": int(games.get("minutes") or 0),
                    "rating": games.get("rating"),
                    "shots": int(shots.get("total") or 0),
                    "sot": int(shots.get("on") or 0),
                    "goals": int(goals.get("total") or 0),
                    "assists": int((goals.get("assists") or 0) or 0),
                    "yellow": int(cards.get("yellow") or 0),
                    "red": int(cards.get("red") or 0),
                    "pass_acc": passes.get("accuracy"),
                    "fouls_committed": int((fouls.get("committed") or 0) or 0),
                    "fouls_drawn": int((fouls.get("drawn") or 0) or 0),
                    "score": score_str,
                    "result": result,
                })

                found = True
                break
            if found:
                break

        if len(out) >= last:
            break

    # newest first
    out.sort(key=lambda r: r.get("date") or "", reverse=True)

    return {
        "player_id": player_id,
        "team_id": team_id,
        "season": season,
        "games": out[:last],
    }

@router.post("/admin/prime-team-stats")
def prime_team_stats(
    day: str = Query(..., description="YYYY-MM-DD UTC start day"),
    days: int = Query(2, ge=1, le=14, description="How many days ahead to scan"),
    refresh: bool = Query(False, description="Force refresh from provider"),
    db: Session = Depends(get_db),
):
    """
    Prime team_season_stats for all SOCCER fixtures between [day, day+days).
    Uses _get_team_stats_cached() so it writes rows if missing/stale.
    """
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    end = start + timedelta(days=days)

    fixtures = (
        db.query(Fixture)
        .filter(Fixture.kickoff_utc >= start, Fixture.kickoff_utc < end)
        .filter((Fixture.sport == None) | (Fixture.sport.in_(["soccer", "football"])))
        .all()
    )

    primed, skipped = 0, 0
    for f in fixtures:
        if not f.provider_fixture_id:
            skipped += 1
            continue

        try:
            fx = get_fixture(int(f.provider_fixture_id)) or {}
            core = (fx.get("response") or [None])[0] or {}
            lg = core.get("league") or {}
            league_id = int(lg.get("id") or 0)
            season = int(lg.get("season") or 0)
            teams = core.get("teams") or {}
            home_id = int((teams.get("home") or {}).get("id") or 0)
            away_id = int((teams.get("away") or {}).get("id") or 0)
            if league_id and season and home_id:
                _get_team_stats_cached(db, home_id, league_id, season, refresh=refresh)
                primed += 1
            if league_id and season and away_id:
                _get_team_stats_cached(db, away_id, league_id, season, refresh=refresh)
                primed += 1
        except Exception:
            skipped += 1

    return {"day": day, "days": days, "primed": primed, "skipped": skipped}

# -----------------------------
# CLUB ROUTES
# -----------------------------

@router.get("/club/overview")
def club_overview(
    team_id: int = Query(..., description="API-Football team id"),
    season: int = Query(..., description="Season (e.g. 2025)"),
    league_id: int | None = Query(None, description="Optional league id for league-only stats"),
    lookback: int = Query(5, ge=3, le=15),
    db: Session = Depends(get_db),
):
    """
    Overview payload for ClubPage.
    - Basic team info from /players (team metadata is stable there)
    - Form summary from recent results
    - (Optional) league stats if league_id provided (cached via TeamSeasonStats)
    """

    # 1) basic team metadata via get_players (has team object + logo)
    team_meta = {}
    try:
        pj = get_players(team_id=team_id, season=season, page=1) or {}
        resp = pj.get("response") or []
        if resp:
            # API-Football shape often includes: response[i].team and response[i].players
            maybe_team = (resp[0] or {}).get("team") or {}
            if maybe_team:
                team_meta = {
                    "id": maybe_team.get("id"),
                    "name": maybe_team.get("name"),
                    "logo": maybe_team.get("logo"),
                }
    except Exception:
        team_meta = {"id": team_id}

    # 2) recent form (W/D/L + GF/GA) using your existing helper-style logic
    recent = get_team_recent_results(team_id, season=season, limit=lookback, league_id=None) or []
    form = []
    wins = draws = losses = 0
    gf = ga = 0

    for m in recent:
        fid = m.get("fixture_id")
        if not fid:
            continue
        try:
            fdet = get_fixture(int(fid)) or {}
            fr = (fdet.get("response") or [None])[0] or {}
            goals = fr.get("goals") or {}

            gh = int(goals.get("home") or 0)
            ga_ = int(goals.get("away") or 0)
            is_home = bool(m.get("is_home"))

            gf_match = gh if is_home else ga_
            ga_match = ga_ if is_home else gh

            gf += gf_match
            ga += ga_match

            if gf_match > ga_match:
                res = "W"
                wins += 1
            elif gf_match == ga_match:
                res = "D"
                draws += 1
            else:
                res = "L"
                losses += 1

            form.append({
                "fixture_id": int(fid),
                "date": m.get("date"),
                "opponent": m.get("opponent"),
                "is_home": is_home,
                "score_for": gf_match,
                "score_against": ga_match,
                "result": res,
                "competition": (fr.get("league") or {}).get("name"),
            })
        except Exception:
            continue

    form_summary = {
        "lookback": lookback,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "gf": gf,
        "ga": ga,
        "avg_gf": round(gf / max(1, (wins + draws + losses)), 3),
        "avg_ga": round(ga / max(1, (wins + draws + losses)), 3),
    }

    # 3) league-only cached stats (optional)
    league_stats = None
    if league_id is not None:
        try:
            league_stats = _get_team_stats_cached(db, team_id, int(league_id), int(season)) or None
        except Exception:
            league_stats = None

    return {
        "team": team_meta or {"id": team_id},
        "season": season,
        "league_id": league_id,
        "form_summary": form_summary,
        "recent_form": form[:lookback],
        "league_stats": league_stats,
    }


@router.get("/club/fixtures")
def club_fixtures(
    team_id: int = Query(..., description="API-Football team id"),
    season: int = Query(...),
    last: int = Query(10, ge=1, le=25),
    next: int = Query(5, ge=0, le=25),
    league_only: bool = Query(False),
    league_id: int | None = Query(None, description="If league_only=true, provide league_id when possible"),
):
    """
    Club fixtures list for the club page:
    - last N matches (played)
    - next N matches (upcoming)
    Uses get_team_recent_results for played.
    For upcoming, we can’t rely on get_team_recent_results, so we do a lightweight
    approach: call provider fixtures endpoint if you have it, OR just omit upcoming.
    (If you already have a provider helper for upcoming fixtures, wire it in.)
    """

    # Played (last N) – you already have this helper
    played = get_team_recent_results(
        team_id, season=season, limit=last, league_id=(league_id if league_only else None)
    ) or []

    # Normalize played into UI-friendly list
    played_out = []
    for m in played:
        fid = m.get("fixture_id")
        if not fid:
            continue
        try:
            fdet = get_fixture(int(fid)) or {}
            fr = (fdet.get("response") or [None])[0] or {}
            goals = fr.get("goals") or {}
            teams = fr.get("teams") or {}
            lg = fr.get("league") or {}
            fixture = fr.get("fixture") or {}

            played_out.append({
                "fixture_id": int(fid),
                "date": fixture.get("date"),
                "league": lg.get("name"),
                "league_id": lg.get("id"),
                "home": (teams.get("home") or {}).get("name"),
                "away": (teams.get("away") or {}).get("name"),
                "home_goals": goals.get("home"),
                "away_goals": goals.get("away"),
                "status": ((fixture.get("status") or {}) or {}).get("short"),
            })
        except Exception:
            continue

    # Upcoming – if you don’t have a helper, return empty for now
    # (If you *do* have a get_team_upcoming_fixtures helper, plug it here)
    upcoming_out = []

    return {
        "team_id": team_id,
        "season": season,
        "league_only": league_only,
        "league_id": league_id,
        "played": played_out,
        "upcoming": upcoming_out,
    }


@router.get("/club/squad")
def club_squad(
    team_id: int = Query(..., description="API-Football team id"),
    season: int = Query(...),
):
    """
    Squad list for ClubPage.
    Uses provider /players; returns a flat list (name, id, photo, position, age, etc.)
    """
    try:
        rows = []
        # get_players likely paginates; if your service supports page param, you can loop.
        # We'll just pull page 1 for now; if you want full paging, tell me the shape.
        pj = get_players(team_id=team_id, season=season, page=1) or {}
        resp = pj.get("response") or []

        for r in resp:
            pl = (r.get("player") or {}) or {}
            st = (r.get("statistics") or [None])[0] or {}
            g = (st.get("games") or {}) if isinstance(st, dict) else {}

            rows.append({
                "id": pl.get("id"),
                "name": pl.get("name"),
                "photo": pl.get("photo"),
                "age": pl.get("age"),
                "nationality": pl.get("nationality"),
                "position": g.get("position") or pl.get("position"),
                "appearances": g.get("appearences") or g.get("appearances"),
                "minutes": g.get("minutes"),
                "rating": g.get("rating"),
            })

        # sort: most minutes / apps first
        def _n(x):
            try:
                return float(x or 0)
            except Exception:
                return 0.0

        rows.sort(key=lambda r: (_n(r.get("minutes")), _n(r.get("appearances"))), reverse=True)

        return {
            "team_id": team_id,
            "season": season,
            "squad": rows,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Club squad fetch failed: {e}")


@router.get("/club/stats")
def club_stats(
    team_id: int = Query(..., description="API-Football team id"),
    league_id: int = Query(..., description="API-Football league id"),
    season: int = Query(...),
    refresh: bool = Query(False),
    db: Session = Depends(get_db),
):
    """
    League-only team stats (cached with TeamSeasonStats table).
    This powers your ClubPage 'Season Stats' section.
    """
    try:
        data = _get_team_stats_cached(db, team_id, league_id, season, refresh=refresh) or {}
        return {
            "team_id": team_id,
            "league_id": league_id,
            "season": season,
            "stats": data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Club stats fetch failed: {e}")


@router.get("/club/topscorers")
def club_top_scorers(
    league_id: int = Query(...),
    season: int = Query(...),
):
    """
    League top scorers (for club page context / league panel).
    """
    try:
        return get_top_scorers(league_id, season)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Top scorers fetch failed: {e}")

@router.get("/club/{team_id}")
def club_page(
    team_id: int,
    lookback_recent: int = Query(10, ge=1, le=25),
    lookahead_upcoming: int = Query(10, ge=1, le=25),
    db: Session = Depends(get_db),
):
    """
    Club page data from fixtures table.

    Returns:
      {
        team_id,
        team_name,
        stats: {played,wins,draws,losses,gf,ga,avg_gf,avg_ga},
        recent: [...],
        upcoming: [...]
      }
    """
    now = datetime.utcnow()

    # --- recent finished fixtures (kickoff <= now) ---
    recent_fx = (
        db.query(Fixture)
        .filter(
            Fixture.sport.in_(["football", "soccer"]),
            or_(
                Fixture.provider_home_team_id == team_id,
                Fixture.provider_away_team_id == team_id,
            ),
            Fixture.kickoff_utc != None,
            Fixture.kickoff_utc <= now,
        )
        .order_by(Fixture.kickoff_utc.desc())
        .limit(lookback_recent)
        .all()
    )

    # --- upcoming fixtures (kickoff > now) ---
    upcoming_fx = (
        db.query(Fixture)
        .filter(
            Fixture.sport.in_(["football", "soccer"]),
            or_(
                Fixture.provider_home_team_id == team_id,
                Fixture.provider_away_team_id == team_id,
            ),
            Fixture.kickoff_utc != None,
            Fixture.kickoff_utc > now,
        )
        .order_by(Fixture.kickoff_utc.asc())
        .limit(lookahead_upcoming)
        .all()
    )

    # best-effort team_name from any fixture row
    team_name = None
    sample = (upcoming_fx[:1] or recent_fx[:1])
    if sample:
        s = sample[0]
        if s.provider_home_team_id == team_id:
            team_name = s.home_team
        elif s.provider_away_team_id == team_id:
            team_name = s.away_team

    # --- compute simple W/D/L + GF/GA from recent settled scores only ---
    played = wins = draws = losses = gf = ga = 0
    for f in recent_fx:
        if f.full_time_home is None or f.full_time_away is None:
            continue

        is_home = f.provider_home_team_id == team_id
        gf_m = f.full_time_home if is_home else f.full_time_away
        ga_m = f.full_time_away if is_home else f.full_time_home

        played += 1
        gf += int(gf_m or 0)
        ga += int(ga_m or 0)

        if gf_m > ga_m:
            wins += 1
        elif gf_m < ga_m:
            losses += 1
        else:
            draws += 1

    stats = {
        "played": played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "gf": gf,
        "ga": ga,
        "avg_gf": round(gf / played, 3) if played else 0.0,
        "avg_ga": round(ga / played, 3) if played else 0.0,
    }

    def _row(f: Fixture):
        return {
            "fixture_id": int(f.id),
            "provider_fixture_id": f.provider_fixture_id,
            "comp": f.comp,
            "kickoff_utc": f.kickoff_utc.isoformat() if f.kickoff_utc else None,
            "home_team": f.home_team,
            "away_team": f.away_team,
            "full_time_home": f.full_time_home,
            "full_time_away": f.full_time_away,
        }

    return {
        "team_id": team_id,
        "team_name": team_name or str(team_id),
        "stats": stats,
        "recent": [_row(x) for x in recent_fx],
        "upcoming": [_row(x) for x in upcoming_fx],
    }