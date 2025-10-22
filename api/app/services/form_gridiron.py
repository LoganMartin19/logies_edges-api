# services/form_gridiron.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from app.services.api_gridiron import fetch_team_games

@dataclass
class GameForm:
    date: str
    home: bool
    opponent: str
    team_score: int
    opp_score: int
    result: str  # "W" or "L"
    venue: Optional[str] = None

@dataclass
class FormSummary:
    games: int
    w: int
    l: int
    pf: int
    pa: int
    avg_pf: float
    avg_pa: float
    attack_score: float   # 0..1
    defense_score: float  # 0..1 (higher is better defense)
    overall_score: float  # blended

@dataclass
class TeamFormGridiron:
    summary: FormSummary
    recent: List[GameForm]

def _normalize_str(x: Optional[str]) -> str:
    return (x or "").strip()

def _get_team_block(g: dict, side: str) -> Dict[str, Any]:
    # Prefer API-Sports "teams.home"/"teams.away"; fallback to common alternates
    t = (g.get("teams") or {})
    if side == "home":
        return t.get("home") or g.get("homeTeam") or {}
    return t.get("away") or g.get("awayTeam") or {}

def _get_score(g: dict, side: str) -> int:
    s = g.get("scores") or g.get("score") or {}
    v = s.get(side)
    if isinstance(v, int):
        return int(v)
    # nested like {"total": 27}
    try:
        return int((v or {}).get("total", 0))
    except Exception:
        return 0

def _get_date_iso(g: dict) -> str:
    # API has g["date"] or g["game"]["date"]["full"]/["timestamp"]
    if isinstance(g.get("date"), str):
        return g["date"]
    d = (g.get("game") or {}).get("date") or {}
    if isinstance(d, str):  # occasionally a flat ISO
        return d
    full = d.get("full")
    if isinstance(full, str):
        return full
    # last resort, stringify whatever timestamp is there
    ts = d.get("timestamp") or g.get("timestamp")
    return str(ts) if ts is not None else ""

def _result(ts: int, os: int) -> str:
    return "W" if ts > os else "L"

def _normalize(z: float, lo: float, hi: float) -> float:
    if hi <= lo: return 0.5
    v = (z - lo) / (hi - lo)
    return 0.0 if v < 0 else 1.0 if v > 1 else v

def get_recent_form_gridiron(
    team_id: int,
    season: int | str | None = None,
    last_n: int = 5,
    league: str | int | None = None,  # e.g., "NFL" or "NCAA"
) -> TeamFormGridiron:
    games = fetch_team_games(team_id=team_id, season=season, league=league, status="FT")
    if not games:
        return TeamFormGridiron(
            summary=FormSummary(
                games=0, w=0, l=0, pf=0, pa=0, avg_pf=0.0, avg_pa=0.0,
                attack_score=0.5, defense_score=0.5, overall_score=0.5
            ),
            recent=[]
        )

    # Sort by date/timestamp descending
    def _ts(g: dict) -> float:
        ts = g.get("timestamp") or (g.get("game") or {}).get("timestamp")
        if isinstance(ts, (int, float)):
            return float(ts)
        # fallback from ISO
        try:
            from datetime import datetime
            iso = _get_date_iso(g)
            return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.0

    games.sort(key=_ts, reverse=True)
    games = games[:max(1, int(last_n))]

    recent: List[GameForm] = []
    pf = pa = w = l = 0

    for g in games:
        home_blk = _get_team_block(g, "home")
        away_blk = _get_team_block(g, "away")
        home_id = home_blk.get("id")
        away_id = away_blk.get("id")

        is_home = (home_id == team_id)
        ts = _get_score(g, "home") if is_home else _get_score(g, "away")
        os = _get_score(g, "away") if is_home else _get_score(g, "home")

        opp_name = away_blk.get("name") if is_home else home_blk.get("name")
        opp_name = _normalize_str(opp_name) or "Opponent"

        recent.append(GameForm(
            date=_get_date_iso(g),
            home=is_home,
            opponent=opp_name,
            team_score=int(ts or 0),
            opp_score=int(os or 0),
            result=_result(ts or 0, os or 0),
            venue=((g.get("venue") or {}).get("name")),
        ))

        pf += int(ts or 0)
        pa += int(os or 0)
        if ts and os and ts > os: w += 1
        else: l += 1

    n = max(1, len(recent))
    avg_pf = pf / n
    avg_pa = pa / n

    # Simple league-agnostic baselines (tune later per league)
    attack_score  = _normalize(avg_pf, lo=14.0, hi=34.0)
    defense_score = _normalize(34.0 - avg_pa, lo=0.0, hi=20.0)  # fewer allowed -> better
    overall_score = 0.6 * attack_score + 0.4 * defense_score

    return TeamFormGridiron(
        summary=FormSummary(
            games=len(recent), w=w, l=l, pf=pf, pa=pa, avg_pf=avg_pf, avg_pa=avg_pa,
            attack_score=attack_score, defense_score=defense_score, overall_score=overall_score
        ),
        recent=recent
    )