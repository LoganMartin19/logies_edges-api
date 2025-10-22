# api/app/schemas.py
from datetime import datetime
from pydantic import BaseModel

# ---- Edge responses ----
class EdgeOut(BaseModel):
    fixture_id: int
    comp: str
    home_team: str
    away_team: str
    kickoff_utc: datetime
    market: str
    bookmaker: str
    price: float
    prob: float
    edge: float
    model_source: str | None = None

    class Config:
        from_attributes = True  # pydantic v2

# ---- Bets I/O ----
class BetIn(BaseModel):
    fixture_id: int
    market: str          # e.g. "O2.5", "BTTS_Y", "1X2_HOME"
    bookmaker: str       # e.g. "bet365"
    price: float         # decimal odds
    stake: float

class BetSettle(BaseModel):
    result: str          # "WON" | "LOST" | "VOID" | "PENDING"
    ret: float           # total returned to bankroll

class BetOut(BaseModel):
    id: int
    fixture_id: int
    market: str
    bookmaker: str
    price: float
    stake: float
    placed_at: datetime | None = None
    result: str | None = None
    ret: float | None = None
    pnl: float | None = None

    class Config:
        from_attributes = True  # pydantic v2