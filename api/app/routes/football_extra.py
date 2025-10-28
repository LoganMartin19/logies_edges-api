# --- replace your import block header with this ---
from fastapi import APIRouter, Query, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
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

    def _pick_block(stat_blocks: list[dict]) -> dict | None:
        # Prefer exact league id/season match, then name match
        for s in stat_blocks or []:
            lg = (s.get("league", {}) or {})
            if int(lg.get("id") or 0) == int(league_id):
                return s
        for s in stat_blocks or []:
            lg = (s.get("league", {}) or {})
            if (lg.get("name") or "").strip().lower() == league_name_lc:
                return s
        return None

    def _flatten(items: list[dict]) -> list[dict]:
        out = []
        for row in items or []:
            player = row.get("player", {}) or {}
            stats = row.get("statistics") or []
            s = _pick_block(stats)
            if not s:
                continue

            games = s.get("games", {}) or {}
            shots = s.get("shots", {}) or {}
            cards = s.get("cards", {}) or {}
            fouls = s.get("fouls", {}) or {}

            name = player.get("name") or "—"
            mins = int(games.get("minutes") or 0)

            sh_total = int(shots.get("total") or 0)
            sh_on = int(shots.get("on") or 0)
            sot_pct = round((sh_on / sh_total * 100.0), 1) if sh_total else 0.0

            fouls_comm = int((fouls.get("committed") or 0) or 0)

            per90 = (lambda v: round((v * 90.0) / mins, 2) if mins else 0.0)

            out.append({
                "id": player.get("id"),
                "name": name,
                "photo": player.get("photo"),
                "pos": games.get("position") or player.get("position") or "?",
                "minutes": mins,
                "shots": sh_total,
                "shots_on": sh_on,
                "sot_pct": sot_pct,
                "yellow": int(cards.get("yellow") or 0),
                "red": int(cards.get("red") or 0),
                "fouls_committed": fouls_comm,
                "shots_per90": per90(sh_total),
                "fouls_committed_per90": per90(fouls_comm),
            })
        out.sort(key=lambda r: (r["minutes"], r["shots"], r["yellow"]), reverse=True)
        return out

    def normalize_team(team_id: int) -> list[dict]:
        rows = get_player_stats(team_id, league_id, season)
        flat = _flatten(rows if isinstance(rows, list) else [])
        if any(r["minutes"] or r["shots"] or r["yellow"] for r in flat):
            return flat

        prev = get_player_stats(team_id, league_id, season - 1)
        flat_prev = _flatten(prev if isinstance(prev, list) else [])
        if any(r["minutes"] or r["shots"] or r["yellow"] for r in flat_prev):
            return flat_prev

        # inside normalize_team(team_id: int)
        all_comp = get_team_season_players_cached(db, team_id, season)
        return _flatten(all_comp)

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
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).first()
    if not fx or not fx.provider_fixture_id:
        raise HTTPException(status_code=404, detail="Fixture not found")

    try:
        pfx = int(fx.provider_fixture_id)
        fx_resp = get_fixture(pfx) or {}
        resp_list = fx_resp.get("response") or []
        if not isinstance(resp_list, list) or not resp_list:
            raise HTTPException(status_code=404, detail="Fixture details not found")
        fr = resp_list[0]

        league_id = fr["league"]["id"]
        season = fr["league"]["season"]
        home_id = fr["teams"]["home"]["id"]
        away_id = fr["teams"]["away"]["id"]

        raw_home = get_players(home_id, league_id, season) or {}
        raw_away = get_players(away_id, league_id, season) or {}
        home_players = raw_home.get("response", []) if isinstance(raw_home, dict) else []
        away_players = raw_away.get("response", []) if isinstance(raw_away, dict) else []

        return {
            "source": "API-Football",
            "fixture_id": fixture_id,
            "league_id": league_id,
            "season": season,
            "home_team": fx.home_team,
            "away_team": fx.away_team,
            "players": {"home": home_players, "away": away_players},
        }
    except HTTPException:
        raise
    except Exception as e:
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

    # ✅ cached pulls
    home_rows = get_team_season_players_cached(db, home_id, season) or []
    away_rows = get_team_season_players_cached(db, away_id, season) or []

    return {
        "source": "API-Football (cached)",
        "fixture_id": fixture_id,
        "season": season,
        "home_team": fx.home_team,
        "away_team": fx.away_team,
        "players": {"home": home_rows, "away": away_rows},
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

    # ---------- helper: build a per-90 snapshot from a single stats row
    def _per90_from_stats(pl_row: dict, stats: dict) -> dict:
        games = stats.get("games") or {}
        shots = stats.get("shots") or {}
        fouls = stats.get("fouls") or {}
        cards = stats.get("cards") or {}
        mins = int(games.get("minutes") or 0)
        p90 = (lambda v: round((v * 90.0) / mins, 2) if mins else 0.0)
        pos = games.get("position")
        return {
            "id": int((pl_row.get("player") or {}).get("id") or 0),
            "name": (pl_row.get("player") or {}).get("name"),
            "photo": (pl_row.get("player") or {}).get("photo"),
            "pos": pos or (pl_row.get("player") or {}).get("position") or "?",
            "minutes": mins,
            "shots_per90": p90(int(shots.get("total") or 0)),
            "sot_per90": p90(int(shots.get("on") or 0)),
            "fouls_per90": p90(int(fouls.get("committed") or 0)),
            "yc_per90": p90(int(cards.get("yellow") or 0)),
        }

    # ---------- pull ALL season rows for both teams (correct endpoint; all pages)
    home_rows = get_team_season_players_cached(db, home_id, season)
    away_rows = get_team_season_players_cached(db, away_id, season)
    season_rows = home_rows + away_rows

    # ---------- locate this player
    player_rows = [
        r
        for r in season_rows
        if int((r.get("player") or {}).get("id") or 0) == int(player_id)
    ]

    # Match-context block: prefer a statistics entry whose league.id == fixture league
    match_block = None
    for row in player_rows:
        for s in (row.get("statistics") or []):
            lg_s = (s.get("league") or {})  # e.g. League One, EFL Trophy, etc.
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

    # ---------- season competitions + totals for this player
    comp_blocks = []
    for row in player_rows:
        for s in (row.get("statistics") or []):
            lg_s = (s.get("league") or {})
            if int(lg_s.get("season") or 0) != season:
                continue
            comp_blocks.append({
                "league_id": lg_s.get("id"),
                "league": lg_s.get("name"),
                "games": s.get("games") or {},
                "goals": s.get("goals") or {},
                "assists": (s.get("goals") or {}).get("assists"),
                "shots": s.get("shots") or {},
                "cards": s.get("cards") or {},
                "minutes": (s.get("games") or {}).get("minutes") or 0,
            })

    totals = None
    if comp_blocks:
        def n(x):
            try:
                return int(x or 0)
            except:
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

    # Team label (best-effort)
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
                    player_rows[0].get("player").get("name") if player_rows else None
                ),
                "photo": match_block.get("photo")
                if match_block
                else (
                    player_rows[0].get("player").get("photo") if player_rows else None
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
):
    drawn_avg = get_team_fouls_drawn_avg(
        team_id, season=season, league_id=league_id, lookback=lookback
    )
    committed_avg = get_team_fouls_committed_avg(
        team_id, season=season, league_id=league_id, lookback=lookback
    )

    recent = get_team_recent_results(
        team_id, season=season, limit=lookback, league_id=league_id
    ) or []

    fixtures_debug = []
    events_seen_any = False
    for m in recent:
        fid = m.get("fixture_id")
        if not fid:
            fixtures_debug.append({
                "fixture_id": None,
                "date": m.get("date"),
                "opponent": m.get("opponent"),
                "is_home": m.get("is_home"),
                "events_available": False,
                "fouls_committed_by_team": None,
                "fouls_committed_by_opp": None,
            })
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

        fixtures_debug.append({
            "fixture_id": fid,
            "date": m.get("date"),
            "opponent": m.get("opponent"),
            "is_home": m.get("is_home"),
            "events_available": bool(ev_rows),
            "fouls_committed_by_team": fouls_by_team if ev_rows else None,
            "fouls_committed_by_opp": fouls_by_opp if ev_rows else None,
        })

    stats_fallback = None
    if league_id is not None:
        try:
            s = _get_team_stats_cached(team_id, league_id, season) or {}
            r = s.get("response") or {}
            fouls = (r.get("fouls") or {})
            fixtures_total = int(((r.get("fixtures") or {}).get("played") or {}).get("total") or 0)
            drawn_total = int(((fouls.get("drawn") or {}).get("total")) or 0)
            committed_total = int(((fouls.get("committed") or {}).get("total")) or 0)
            stats_fallback = {
                "fixtures_played": fixtures_total,
                "drawn_total": drawn_total,
                "committed_total": committed_total,
                "drawn_per_match": (drawn_total / fixtures_total) if fixtures_total else 0.0,
                "committed_per_match": (committed_total / fixtures_total) if fixtures_total else 0.0,
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
        home: { ...provider response... },
        away: { ...provider response... },
        summary: { home: {...}, away: {...} }   # handy subset
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
    season    = int(lg.get("season") or 0)

    teams = core.get("teams") or {}
    home_pid = int(((teams.get("home") or {}).get("id")) or 0)
    away_pid = int(((teams.get("away") or {}).get("id")) or 0)

    if not (league_id and season and home_pid and away_pid):
        raise HTTPException(status_code=502, detail="Missing league/season/team ids from provider")

    # fetch (cached) team-season stats for this league
    home_json = _get_team_stats_cached(db, home_pid, league_id, season, refresh=refresh) or {}
    away_json = _get_team_stats_cached(db, away_pid, league_id, season, refresh=refresh) or {}

    # provider puts payload under 'response' – keep that node for UI
    home = home_json.get("response") or {}
    away = away_json.get("response") or {}

    # tiny, safe summary block your UI can use if desired
    def _safe(v, *path, default=0.0):
        cur = v
        try:
            for k in path:
                cur = cur.get(k) if isinstance(cur, dict) else {}
            x = cur
            if isinstance(x, (int, float)): return float(x)
            s = str(x).strip().replace("%","")
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

    return {
        "fixture_id": fixture_id,
        "provider_fixture_id": pfx,
        "league_id": league_id,
        "season": season,
        "home_team": fx.home_team,
        "away_team": fx.away_team,
        "home": home,
        "away": away,
        "summary": {"home": _summ(home), "away": _summ(away)},
    }

@router.get("/opponent-pace")
def opponent_pace(
    fixture_id: int,
    lookback: int = Query(5, ge=1, le=10),
    db: Session = Depends(get_db),
):
    """
    Per-fixture context:
      - Opponent defensive profile: shots/SoT conceded per match (last N)
      - Pace multipliers for shots/SoT (vs league baselines)
      - Team attacking avgs: Shots, SoT, Corners, Cards, xG (last N)
      - Team xG for/against (last N)
      - Team fouls committed/drawn (last N)
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

    # Opponent defensive profiles (concessions)
    away_conc = get_team_shots_against_avgs(
        away_id, season=season, league_id=league_id, lookback=lookback
    )
    home_conc = get_team_shots_against_avgs(
        home_id, season=season, league_id=league_id, lookback=lookback
    )

    # Our attacking profiles (using fixtures/statistics)
    home_att = get_team_attack_avgs(
        home_id, season=season, league_id=league_id, lookback=lookback
    )
    away_att = get_team_attack_avgs(
        away_id, season=season, league_id=league_id, lookback=lookback
    )

    # xG for/against
    home_xg = get_team_xg_avgs(
        home_id, season=season, league_id=league_id, lookback=lookback
    )
    away_xg = get_team_xg_avgs(
        away_id, season=season, league_id=league_id, lookback=lookback
    )

    # fouls committed/drawn
    home_fouls = get_team_fouls_from_statistics_avg(
        home_id, season=season, league_id=league_id, lookback=lookback
    )
    away_fouls = get_team_fouls_from_statistics_avg(
        away_id, season=season, league_id=league_id, lookback=lookback
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
    home_drawn90 = get_team_fouls_drawn_avg(
        home_pid, season=season, league_id=league_id, lookback=lookback
    ) if home_pid else 0.0
    away_drawn90 = get_team_fouls_drawn_avg(
        away_pid, season=season, league_id=league_id, lookback=lookback
    ) if away_pid else 0.0
    home_comm90 = get_team_fouls_committed_avg(
        home_pid, season=season, league_id=league_id, lookback=lookback
    ) if home_pid else 0.0
    away_comm90 = get_team_fouls_committed_avg(
        away_pid, season=season, league_id=league_id, lookback=lookback
    ) if away_pid else 0.0

    # shots/SoT conceded pace
    def _pace(team_id: int | None) -> tuple[float, float]:
        if not (team_id and league_id and season):
            return (0.0, 0.0)
        try:
            pa = get_team_shots_against_avgs(
                team_id, season=season, league_id=league_id, lookback=lookback
            ) or {}
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
    stored_odds = db.query(PlayerOdds).filter(PlayerOdds.fixture_id == fixture_id).all()
    odds_map: dict[tuple[int, str, float], dict] = {}
    for o in stored_odds:
        key = (int(o.player_id), (o.market or "").lower(), float(o.line or 0.0))
        best = odds_map.get(key)
        if not best or float(o.price) > best["price"]:
            odds_map[key] = {"bookmaker": o.bookmaker, "price": float(o.price)}

    team_norm = (team or "").strip().lower()
    want_team = team_norm in {"home", "away"}
    market_set = set(m.strip().lower() for m in (markets or "").split(",") if m and m.strip())

    out = {"fixture_id": fixture_id, "props": []}

    for side in ("home", "away"):
        if want_team and side != team_norm:
            continue

        roster = stats_data.get(side, []) or []

        if side == "home":
            fouls_ctx = opponent_fouls_factor(away_drawn90)
            shots_ctx = pace_factor(away_opp_shotsA, LEAGUE_AVG_SHOTS_AGAINST)
            sot_ctx = pace_factor(away_opp_sotA, LEAGUE_AVG_SOT_AGAINST)
            opp_drawn90 = away_drawn90
            opp_comm90 = away_comm90
            opp_shotsA, opp_sotA = away_opp_shotsA, away_opp_sotA
        else:
            fouls_ctx = opponent_fouls_factor(home_drawn90)
            shots_ctx = pace_factor(home_opp_shotsA, LEAGUE_AVG_SHOTS_AGAINST)
            sot_ctx = pace_factor(home_opp_sotA, LEAGUE_AVG_SOT_AGAINST)
            opp_drawn90 = home_drawn90
            opp_comm90 = home_comm90
            opp_shotsA, opp_sotA = home_opp_shotsA, home_opp_sotA

        ref_ctx = referee_cards_factor()

        for pl in roster:
            mins_played = int(pl.get("minutes") or 0)
            m_used = minutes or (80 if mins_played >= 600 else 30)

            shots_per90 = float(pl.get("shots_per90") or 0.0)
            fouls90 = float(pl.get("fouls_committed_per90") or 0.0)

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
            p_fouls05 = prob_over_xpoint5(
                fouls90, m_used, 0.5, opponent_factor=fouls_ctx
            )
            p_card = prob_card(
                cards_per90, m_used, ref_factor=ref_ctx, opponent_factor=fouls_ctx
            )

            markets_calc = [
                ("shots_over_1.5", 1.5, p_shots15, fair_odds(p_shots15)),
                ("sot_over_0.5", 0.5, p_sot05, fair_odds(p_sot05)),
                ("fouls_over_0.5", 0.5, p_fouls05, fair_odds(p_fouls05)),
                ("to_be_booked", 0.5, p_card, fair_odds(p_card)),
            ]

            for market, line, prob, fair in markets_calc:
                if prob < min_prob:
                    continue
                if market_set and market not in market_set:
                    continue

                key = (int(pl["id"]), market, float(line))
                bm = odds_map.get(key)

                out["props"].append({
                    "player_id": int(pl["id"]),
                    "player": html.unescape(pl.get("name") or ""),  # ✅ unescape
                    "team_side": side,
                    "market": market,
                    "line": float(line),
                    "proj_minutes": int(m_used),
                    "prob": float(prob),
                    "fair_odds": float(fair) if fair else None,
                    "best_price": bm["price"] if bm else None,
                    "bookmaker": bm["bookmaker"] if bm else None,
                    "edge": edge(prob, bm["price"]) if bm and fair else None,
                    # context (for Why/preview UIs)
                    "per90_shots": round(shots_per90, 2),
                    "per90_sot": round(sot_per90, 2),
                    "per90_fouls": round(fouls90, 2),
                    "cards_per90": round(cards_per90, 2),
                    "opp_fouls_drawn_per90": round(opp_drawn90, 2),
                    "opp_fouls_committed_per90": round(opp_comm90, 2),
                    "opponent_factor": round(fouls_ctx, 3),
                    "ref_factor": round(ref_ctx, 3),
                    "opp_shots_against_per_match": round(opp_shotsA, 2),
                    "opp_sot_against_per_match": round(opp_sotA, 2),
                    "pace_factor_shots": round(shots_ctx, 3),
                    "pace_factor_sot": round(sot_ctx, 3),
                })

    out["props"].sort(
        key=lambda r: (float(r.get("edge") or 0.0), float(r["prob"])), reverse=True
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

    # --- Poisson backbone
    home_stats = _get_team_stats_cached(home_pid, league_id, season) or {}
    away_stats = _get_team_stats_cached(away_pid, league_id, season) or {}
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

    # --- Pull latest team_form model probabilities and overlay
    model = _latest_probs_for_fixture(db, fixture_id, source="team_form")
    p_home_use = model.get("HOME_WIN", p_home_poi)
    p_draw_use = model.get("DRAW", p_draw_poi)
    p_away_use = model.get("AWAY_WIN", p_away_poi)
    ou_use = model.get("O2.5", ou_poi)
    btts_use = model.get("BTTS_Y", btts_poi)
    model_source_used = "team_form" if model else "poisson"

    # Opponent pace
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

    # Rolling team stats for narrative
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

    # Player props preview (top-N by minutes) with fouls/pace context
    home_drawn90 = home_fouls.get("fouls_drawn_per_match", 0.0)
    away_drawn90 = away_fouls.get("fouls_drawn_per_match", 0.0)

    def opponent_fouls_factor(opponent_drawn90: float) -> float:
        if not opponent_adj:
            return 1.0
        if opponent_drawn90 <= 0:
            return 1.0
        return clamp(opponent_drawn90 / 10.0, 0.80, 1.25)

    def referee_cards_factor() -> float:
        if not ref_adj:
            return 1.0
        if ref_factor_override:
            return ref_factor_override
        return 1.0

    fouls_ctx_home = opponent_fouls_factor(away_drawn90)
    fouls_ctx_away = opponent_fouls_factor(home_drawn90)
    ref_ctx = referee_cards_factor()

    pdata = _get_player_props_data(fixture_id, db)
    top_players = {"home": [], "away": []}
    for side in ("home", "away"):
        roster = sorted(
            pdata.get(side, []) or [], key=lambda r: r.get("minutes", 0), reverse=True
        )[:top_n_per_team]
        opp_fouls_ctx = fouls_ctx_home if side == "home" else fouls_ctx_away
        pace_s = home_ctx["pace_factor_shots"] if side == "home" else away_ctx["pace_factor_shots"]
        pace_t = home_ctx["pace_factor_sot"] if side == "home" else away_ctx["pace_factor_sot"]
        block = []
        for pl in roster:
            mins_played = int(pl.get("minutes") or 0)
            m_used = minutes or (80 if mins_played >= 600 else 30)
            shots_per90 = float(pl.get("shots_per90") or 0.0)
            fouls90 = float(pl.get("fouls_committed_per90") or 0.0)
            shots_on = float(pl.get("shots_on") or 0.0)

            if mins_played > 0:
                sot_per90 = (shots_on * 90.0) / mins_played
                cards_per90 = (float(pl.get("yellow") or 0.0) * 90.0) / mins_played
            else:
                sot_per90 = 0.0
                cards_per90 = 0.0

            p_shots15 = prob_over_xpoint5(shots_per90 * pace_s, m_used, 1.5)
            p_sot05 = prob_over_xpoint5(sot_per90 * pace_t, m_used, 0.5)
            p_fouls05 = prob_over_xpoint5(
                fouls90, m_used, 0.5, opponent_factor=opp_fouls_ctx
            )
            p_card = prob_card(
                cards_per90, m_used, ref_factor=ref_ctx, opponent_factor=opp_fouls_ctx
            )

            block.append({
                "player_id": pl.get("id"),
                "player": html.unescape(pl.get("name") or ""),
                "minutes": m_used,
                "markets": {
                    "shots_over_1.5": {"p": p_shots15, "fair": fair_odds(p_shots15)},
                    "sot_over_0.5": {"p": p_sot05, "fair": fair_odds(p_sot05)},
                    "fouls_over_0.5": {"p": p_fouls05, "fair": fair_odds(p_fouls05)},
                    "to_be_booked": {"p": p_card, "fair": fair_odds(p_card)},
                },
            })
        top_players[side] = block

    # Narrative
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
    Return last-N matches for this player (minutes, shots, SoT, goals, cards, fouls, rating, etc.)
    Data source: /fixtures (to find recent team games) + /fixtures/players per fixture.
    """
    fx = db.query(Fixture).filter(Fixture.id == fixture_id).one_or_none()
    if not fx or not fx.provider_fixture_id:
        raise HTTPException(status_code=404, detail="Fixture not found")

    # fixture context
    fcore = (get_fixture(int(fx.provider_fixture_id)).get("response") or [None])[0] or {}
    lg = fcore.get("league") or {}
    season = int(lg.get("season") or 0)
    league_id = int(lg.get("id") or 0)
    teams = fcore.get("teams") or {}
    home_pid = int((teams.get("home") or {}).get("id") or 0)
    away_pid = int((teams.get("away") or {}).get("id") or 0)

    # decide which team this player belongs to
    home_rows = get_team_season_players_cached(db, home_pid, season)
    away_rows = get_team_season_players_cached(db, away_pid, season)
    belongs_home = any(int((r.get("player") or {}).get("id") or 0) == int(player_id) for r in home_rows)
    team_id = home_pid if belongs_home else away_pid

    # pull recent fixtures for that team
    recent = get_team_recent_results(
        team_id,
        season=season,
        limit=max(last * 2, last),  # ask a few extra in case of DNPs
        league_id=(league_id if league_only else None),
    ) or []

    out = []
    for m in recent:
        fid = m.get("fixture_id")
        if not fid:
            continue

        # fetch fixture detail for score + fallback competition
        score_str = None
        result = None
        league_name = None
        try:
            fdet = get_fixture(int(fid)) or {}
            fresp = (fdet.get("response") or [None])[0] or {}
            goals = (fresp.get("goals") or {})
            gh = int(goals.get("home") or 0)
            ga = int(goals.get("away") or 0)
            league_name = ((fresp.get("league") or {}).get("name")) or None

            # Orient score from player's-team perspective
            if bool(m.get("is_home")):
                score_str = f"{gh}-{ga}"
                if gh > ga:   result = "W"
                elif gh < ga: result = "L"
                else:         result = "D"
            else:
                score_str = f"{ga}-{gh}"
                if ga > gh:   result = "W"
                elif ga < gh: result = "L"
                else:         result = "D"
        except Exception:
            score_str = None
            result = None
            league_name = None

        pj = get_fixture_players_cached(db, fid) or {}
        resp = pj.get("response") or []
        if not isinstance(resp, list):
            resp = []

        # scan both sides for the player
        for side in resp:
            players_list = (side.get("players") or [])
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

                # opponent name = the other team in response
                opp_name = None
                for other in resp:
                    tid = (other.get("team") or {}).get("id")
                    if tid and tid != (tm.get("id")):
                        opp_name = (other.get("team") or {}).get("name")
                        break

                out.append({
                    "fixture_id": int(fid),
                    "date": m.get("date"),
                    "team": html.unescape(tm.get("name") or ""),
                    "opponent": html.unescape(opp_name or ""),
                    "is_home": bool(m.get("is_home")),
                    "competition": (lg_s.get("name") or league_name),
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
                    "result": result,  # W/D/L from player's team perspective
                })
                break  # found the player; stop inner loop
        if len(out) >= last:
            break

    out.sort(key=lambda r: r.get("date") or "", reverse=True)
    return {"player_id": player_id, "team_id": team_id, "season": season, "games": out[:last]}

from fastapi import Query

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