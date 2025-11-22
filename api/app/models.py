from sqlalchemy import (
    Column, BigInteger, Integer, String, Numeric, DateTime, ForeignKey,
    UniqueConstraint, Index, Boolean, Float, Date, func, JSON
)
from sqlalchemy.orm import relationship
from datetime import datetime
from .db import Base


class Fixture(Base):
    __tablename__ = "fixtures"

    id = Column(BigInteger, primary_key=True)
    provider_fixture_id = Column(String, unique=True, index=True)
    comp = Column(String, index=True)
    home_team = Column(String, index=True)
    provider_home_team_id = Column(Integer, index=True)
    away_team = Column(String, index=True)
    provider_away_team_id = Column(Integer, index=True)
    provider_league_id = Column(Integer, index=True)
    kickoff_utc = Column(DateTime, index=True)
    country = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    referee = Column(String, nullable=True)
    venue = Column(String, nullable=True)
    city = Column(String, nullable=True)
    sport = Column(String, default="football", index=True)  # "football","nfl","cfb","nhl","nba",...

    # Final score / settlement
    full_time_home = Column(Integer, nullable=True)
    full_time_away = Column(Integer, nullable=True)
    result_settled = Column(Boolean, default=False, index=True)

    # Relationships
    odds = relationship("Odds", back_populates="fixture", cascade="all, delete-orphan")
    probs = relationship("ModelProb", back_populates="fixture", cascade="all, delete-orphan")
    edges = relationship("Edge", back_populates="fixture", cascade="all, delete-orphan")
    closing_odds = relationship("ClosingOdds", back_populates="fixture", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="fixture", cascade="all, delete-orphan")
    player_odds = relationship("PlayerOdds", back_populates="fixture", cascade="all, delete-orphan")
    featured_picks = relationship("FeaturedPick", back_populates="fixture", cascade="all, delete-orphan")  # ✅

    def as_dict(self):
        return {
            "id": self.id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "comp": self.comp,
            "kickoff_utc": self.kickoff_utc.isoformat() if self.kickoff_utc else None,
        }

    def as_team_dict(self, team: str | None = None):
        is_home = team == self.home_team if team else True
        gf = self.full_time_home if is_home else self.full_time_away
        ga = self.full_time_away if is_home else self.full_time_home
        opponent = self.away_team if is_home else self.home_team
        return {
            "team": team or (self.home_team if is_home else self.away_team),
            "opponent": opponent,
            "is_home": is_home,
            "score": f"{gf}-{ga}" if gf is not None and ga is not None else None,
            "goals_for": gf,
            "goals_against": ga,
            "result": ("win" if gf > ga else "loss" if gf < ga else "draw")
                      if gf is not None and ga is not None else None,
            "date": self.kickoff_utc.isoformat() if self.kickoff_utc else None,
            "fixture_id": self.id,
        }


class Odds(Base):
    __tablename__ = "odds"

    id = Column(BigInteger, primary_key=True)
    fixture_id = Column(BigInteger, ForeignKey("fixtures.id", ondelete="CASCADE"), index=True)
    bookmaker = Column(String, index=True)
    market = Column(String, index=True)  # e.g. O2.5, BTTS_Y, HOME_WIN, etc.
    price = Column(Numeric)
    last_seen = Column(DateTime, default=datetime.utcnow, index=True)

    fixture = relationship("Fixture", back_populates="odds")

    __table_args__ = (
        UniqueConstraint("fixture_id", "bookmaker", "market", name="uq_odds_row"),
        Index("ix_odds_fx_mkt_book", "fixture_id", "market", "bookmaker"),
    )


class ModelProb(Base):
    __tablename__ = "model_probs"

    id = Column(BigInteger, primary_key=True)
    fixture_id = Column(BigInteger, ForeignKey("fixtures.id", ondelete="CASCADE"), index=True)
    source = Column(String)     # e.g. "consensus_v2", "team_form"
    market = Column(String)     # must match Odds.market naming
    prob = Column(Numeric)      # 0..1
    as_of = Column(DateTime, default=datetime.utcnow, index=True)

    fixture = relationship("Fixture", back_populates="probs")

    __table_args__ = (
        UniqueConstraint("fixture_id", "source", "market", name="uq_prob_row"),
        Index("ix_model_probs_fx_mkt", "fixture_id", "market"),
    )


class Edge(Base):
    __tablename__ = "edges"

    id = Column(BigInteger, primary_key=True)
    fixture_id = Column(BigInteger, ForeignKey("fixtures.id", ondelete="CASCADE"), index=True)
    market = Column(String, index=True)
    bookmaker = Column(String, index=True)
    price = Column(Numeric)
    model_source = Column(String, index=True)
    prob = Column(Numeric)
    edge = Column(Numeric)      # prob*price - 1
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    fixture = relationship("Fixture", back_populates="edges")

    __table_args__ = (
        Index("ix_edges_fx_mkt_book", "fixture_id", "market", "bookmaker"),
    )


class Bet(Base):
    __tablename__ = "bets"

    id = Column(BigInteger, primary_key=True)
    fixture_id = Column(BigInteger, ForeignKey("fixtures.id", ondelete="RESTRICT"), index=True)
    market = Column(String, index=True)
    bookmaker = Column(String, index=True)
    price = Column(Numeric)
    stake = Column(Numeric)
    placed_at = Column(DateTime, default=datetime.utcnow)
    result = Column(String, default="PENDING")  # WON/LOST/VOID/PENDING
    ret = Column(Numeric)                       # return
    pnl = Column(Numeric)
    duplicate_alert_hash = Column(String, nullable=True, index=True)


class Bankroll(Base):
    __tablename__ = "bankroll"

    id = Column(BigInteger, primary_key=True)
    as_of = Column(DateTime, default=datetime.utcnow, index=True)
    balance = Column(Numeric)


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    # match Bet.id type (BigInteger) ✅
    bet_id = Column(BigInteger, ForeignKey("bets.id"), unique=True, nullable=False)
    sent_at = Column(DateTime, default=datetime.utcnow)


class ClosingOdds(Base):
    __tablename__ = "closing_odds"

    id = Column(BigInteger, primary_key=True)
    fixture_id = Column(BigInteger, ForeignKey("fixtures.id", ondelete="CASCADE"), nullable=False, index=True)
    market = Column(String, nullable=False, index=True)       # e.g. "HOME_WIN", "DRAW", "O2.5"
    bookmaker = Column(String, nullable=False, index=True)
    price = Column(Float, nullable=False)
    captured_at = Column(DateTime, nullable=False, index=True)
    source = Column(String, nullable=False, default="closing_rule_30m")

    fixture = relationship("Fixture", back_populates="closing_odds")

    __table_args__ = (
        UniqueConstraint("fixture_id", "market", "bookmaker", name="uq_closingodds_fx_mkt_book"),
        Index("ix_closingodds_fx_mkt", "fixture_id", "market"),
    )


class Calibration(Base):
    __tablename__ = "calibration"

    id = Column(BigInteger, primary_key=True)
    market = Column(String, index=True, nullable=False)          # e.g. "U2.5", "BTTS_N"
    book = Column(String, index=True, nullable=False, default="bet365")
    scope = Column(String, index=True, nullable=False, default="global")  # e.g. "global" or a league name
    alpha = Column(Float, nullable=False)                         # intercept
    beta = Column(Float, nullable=False)                          # slope
    n_train = Column(Integer, nullable=False, default=0)
    logloss = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("market", "book", "scope", name="uq_calibration_key"),
    )


class TeamForm(Base):
    __tablename__ = "team_form"

    id = Column(Integer, primary_key=True)
    team = Column(String, index=True)
    comp = Column(String, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow)

    form_wins = Column(Integer, default=0)
    form_draws = Column(Integer, default=0)
    form_losses = Column(Integer, default=0)

    goals_for = Column(Integer, default=0)
    goals_against = Column(Integer, default=0)

    last_5_results = Column(String)           # e.g. "WDLWW"
    last_5_goals_for = Column(String)         # e.g. "2,0,1,3,2"
    last_5_goals_against = Column(String)     # e.g. "1,1,0,1,0"
    strength = Column(Float, nullable=True)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    fixture_id = Column(BigInteger, ForeignKey("fixtures.id", ondelete="CASCADE"), index=True)

    market = Column(String, index=True)          # e.g. "1X2", "BTTS", "O2.5"
    predicted_side = Column(String, index=True)  # e.g. "HOME_WIN", "BTTS_Y"
    prob = Column(Float)                         # probability of predicted side
    fair_price = Column(Float)                   # implied fair odds = 1/prob
    model_source = Column(String, index=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    settled = Column(Boolean, default=False, index=True)
    correct = Column(Boolean, nullable=True)
    confidence = Column(String, nullable=True)

    fixture = relationship("Fixture", back_populates="predictions")


class PollVote(Base):
    __tablename__ = "poll_votes"

    id = Column(Integer, primary_key=True, index=True)
    # match Fixture.id type (BigInteger) ✅
    fixture_id = Column(BigInteger, index=True)
    choice = Column(String)      # "home", "draw", "away"
    ip_hash = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class LeagueStanding(Base):
    __tablename__ = "league_standings"

    id = Column(Integer, primary_key=True, index=True)
    league = Column(String, index=True)      # e.g. "EPL", "UCL", etc.
    season = Column(String, index=True)      # e.g. "2025/2026"
    team = Column(String)
    position = Column(Integer)
    played = Column(Integer)
    win = Column(Integer)
    draw = Column(Integer)
    lose = Column(Integer)
    gf = Column(Integer)
    ga = Column(Integer)
    points = Column(Integer)
    form = Column(String)

    # ✅ Automatically set on insert and update
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PlayerOdds(Base):
    __tablename__ = "player_odds"

    id = Column(Integer, primary_key=True)
    fixture_id = Column(BigInteger, ForeignKey("fixtures.id", ondelete="CASCADE"), index=True)

    player_id = Column(Integer, index=True, nullable=True)   # provider player id if available
    player_name = Column(String, index=True, nullable=False)

    market = Column(String, index=True, nullable=False)      # e.g. "shots","sot","yellow"
    line = Column(Float, nullable=True)                      # e.g. 0.5, 1.5, 2.5

    bookmaker = Column(String, index=True, nullable=False)
    price = Column(Float, nullable=False)

    last_seen = Column(DateTime, default=datetime.utcnow, index=True)

    fixture = relationship("Fixture", back_populates="player_odds")

    __table_args__ = (
        UniqueConstraint("fixture_id", "player_id", "player_name", "market", "line", "bookmaker",
                         name="uq_player_odds_row"),
        Index("ix_player_odds_fx_market", "fixture_id", "market"),
        Index("ix_player_odds_player_market", "player_id", "market"),
    )

class FeaturedPick(Base):
    __tablename__ = "featured_picks"

    id = Column(BigInteger, primary_key=True)
    fixture_id = Column(
        BigInteger,
        ForeignKey("fixtures.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    # snapshot for public site
    day = Column(Date, index=True, nullable=False)
    sport = Column(String, index=True, nullable=False)
    comp = Column(String, index=True, nullable=True)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    kickoff_utc = Column(DateTime, nullable=False, index=True)

    # pick data
    market = Column(String, nullable=False)
    bookmaker = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    edge = Column(Float, nullable=True)
    note = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    stake = Column(Float, nullable=True)          # units risked
    result = Column(String, nullable=True)        # 'won' | 'lost' | 'void'
    settled_at = Column(DateTime, nullable=True)

    # 🔒 NEW: premium flag
    is_premium_only = Column(
        Boolean,
        nullable=False,
        server_default="false",  # existing rows => False
    )

    __table_args__ = (
        UniqueConstraint(
            "day", "fixture_id", "market", "bookmaker",
            name="uq_featuredpicks_day_fx_mkt_book",
        ),
        Index("ix_featuredpicks_day_sport", "day", "sport"),
    )

    # tie back to fixture
    fixture = relationship("Fixture", back_populates="featured_picks")

# --- AI-written match previews (cached) ---
class AIPreview(Base):
    __tablename__ = "ai_previews"

    id = Column(BigInteger, primary_key=True)
    fixture_id = Column(BigInteger, ForeignKey("fixtures.id", ondelete="CASCADE"), index=True, nullable=False)

    day = Column(Date, index=True, nullable=False)               # UTC calendar date for the card
    sport = Column(String, index=True, nullable=False, default="football")
    comp = Column(String, index=True, nullable=True)

    # The generated text
    preview = Column(String, nullable=False)                     # 4–6 lines, plain text/markdown
    model = Column(String, nullable=False, default="gpt-5")      # store which LLM you used
    tokens = Column(Integer, nullable=True)                      # optional accounting

    # housekeeping
    created_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("fixture_id", "day", name="uq_ai_preview_fixture_day"),
        Index("ix_ai_previews_day_sport", "day", "sport"),
    )

class TeamSeasonStats(Base):
    __tablename__ = "team_season_stats"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, index=True)
    league_id = Column(Integer, index=True)
    season = Column(Integer, index=True)
    stats_json = Column(JSON)           # store full API response
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TeamSeasonPlayers(Base):
    """
    Cache for /players?team=ID&season=YYYY (all pages, all comps).
    Stores the raw 'response' array flattened (one row per player).
    """
    __tablename__ = "team_season_players"
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, index=True, nullable=False)
    season = Column(Integer, index=True, nullable=False)
    players_json = Column(JSON, nullable=False)         # list[dict]  (flattened per-player objects you already build)
    updated_at = Column(DateTime, default=datetime.utcnow, index=True)
    __table_args__ = (UniqueConstraint("team_id", "season", name="uq_tsp_team_season"),)

class PlayerSeasonStats(Base):
    """
    Cache for /players (stats list) keyed by player+season (optional league_id filter).
    Useful when you need a player's comp breakdown quickly.
    """
    __tablename__ = "player_season_stats"
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, index=True, nullable=False)
    season = Column(Integer, index=True, nullable=False)
    league_id = Column(Integer, index=True, nullable=True)  # null = all comps
    stats_json = Column(JSON, nullable=False)               # full 'statistics' list for that player/season
    updated_at = Column(DateTime, default=datetime.utcnow, index=True)
    __table_args__ = (UniqueConstraint("player_id", "season", "league_id", name="uq_pss_player_season_league"),)

class FixturePlayersCache(Base):
    """
    Cache for /fixtures/players?fixture=ID (lineup+bench with stats blocks).
    """
    __tablename__ = "fixture_players_cache"
    id = Column(Integer, primary_key=True)
    fixture_provider_id = Column(Integer, index=True, nullable=False)  # API-Football fixture id
    payload = Column(JSON, nullable=False)                             # full response['response'] you use
    updated_at = Column(DateTime, default=datetime.utcnow, index=True)
    __table_args__ = (UniqueConstraint("fixture_provider_id", name="uq_fpc_fixture"),)

# api/app/models.py  (add near FeaturedPick)
# --- Accas (shared by model + tipsters) --------------------------------------

# --- AccaTicket / AccaLeg --------------------------------------------------

class AccaTicket(Base):
    __tablename__ = "acca_tickets"

    id = Column(BigInteger, primary_key=True)
    source = Column(String, nullable=False, default="tipster", index=True)
    tipster_id = Column(Integer, ForeignKey("tipsters.id", ondelete="CASCADE"), index=True, nullable=True)

    day = Column(Date, index=True, nullable=False)
    sport = Column(String, index=True, nullable=False, default="football")
    title = Column(String, nullable=True)
    note = Column(String, nullable=True)
    stake_units = Column(Float, nullable=False, default=1.0)
    is_public = Column(Boolean, nullable=False, default=False)

    combined_price = Column(Float, nullable=True)
    est_edge = Column(Float, nullable=True)

    result = Column(String, nullable=True)
    profit = Column(Float, nullable=True)
    settled_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # ✅ NEW: back-reference to Tipster
    tipster = relationship("Tipster", back_populates="accas")

    legs = relationship("AccaLeg", back_populates="ticket", cascade="all, delete-orphan")


class AccaLeg(Base):
    __tablename__ = "acca_legs"

    id = Column(BigInteger, primary_key=True)
    ticket_id = Column(BigInteger, ForeignKey("acca_tickets.id", ondelete="CASCADE"), index=True, nullable=False)

    fixture_id = Column(BigInteger, ForeignKey("fixtures.id", ondelete="SET NULL"))
    # ✅ denormalised names for fast UI (optional, nullable)
    home_name = Column(String, nullable=True)
    away_name = Column(String, nullable=True)

    market = Column(String, nullable=False)
    bookmaker = Column(String, nullable=True)
    price = Column(Float, nullable=False)
    note = Column(String, nullable=True)

    result = Column(String, nullable=True)

    ticket = relationship("AccaTicket", back_populates="legs")

class ExpertPrediction(Base):
    __tablename__ = "expert_predictions"

    id = Column(BigInteger, primary_key=True)
    fixture_id = Column(BigInteger, ForeignKey("fixtures.id", ondelete="CASCADE"), index=True, nullable=False)
    day = Column(Date, index=True, nullable=False)

    # full JSON payload we show on the UI
    payload = Column(JSON, nullable=False)

    # handy columns for quick filtering / future stats
    home_prob = Column(Numeric)     # 0..1
    draw_prob = Column(Numeric)     # 0..1
    away_prob = Column(Numeric)     # 0..1
    confidence = Column(String)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)

    fixture = relationship("Fixture")
    __table_args__ = (
        UniqueConstraint("fixture_id", "day", name="uq_expertpred_fixture_day"),
        Index("ix_expertpred_fixture_day", "fixture_id", "day"),
    )

# --- Tipster ---------------------------------------------------------------

class Tipster(Base):
    __tablename__ = "tipsters"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    bio = Column(String)
    avatar_url = Column(String)
    sport_focus = Column(String)
    social_links = Column(JSON, default=dict)
    join_date = Column(DateTime, default=datetime.utcnow)
    is_verified = Column(Boolean, default=False)

    roi_30d = Column(Float, default=0.0)
    winrate_30d = Column(Float, default=0.0)
    profit_30d = Column(Float, default=0.0)
    picks_30d = Column(Integer, default=0)

    # existing
    picks = relationship("TipsterPick", back_populates="tipster", cascade="all, delete-orphan")

    # ✅ NEW: link to acca tickets this tipster owns
    accas = relationship("AccaTicket", back_populates="tipster", cascade="all, delete-orphan")


class TipsterPick(Base):
    __tablename__ = "tipster_picks"

    id = Column(Integer, primary_key=True)

    tipster_id = Column(Integer, ForeignKey("tipsters.id"), index=True, nullable=False)
    fixture_id = Column(BigInteger, ForeignKey("fixtures.id", ondelete="SET NULL"), index=True, nullable=True)

    market = Column(String, index=True)         # "O2.5", "BTTS_Y", "HOME_WIN", etc
    bookmaker = Column(String)                  # optional
    price = Column(Float)                       # decimal odds
    stake = Column(Float, default=1.0)          # £ units
    created_at = Column(DateTime, default=datetime.utcnow)

    # ⭐️ NEW: premium gating
    is_premium_only = Column(Boolean, default=False, nullable=False)

    # settlement
    result = Column(String, nullable=True)      # "WIN","LOSE","PUSH", None (unsettled)
    profit = Column(Float, default=0.0)         # stake*(price-1) or -stake (push=0)

    tipster = relationship("Tipster", back_populates="picks")
    fixture = relationship("Fixture")

# --- Users, Follows, Bet Log, Subscriptions ---------------------------------

class User(Base):
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True)
    firebase_uid = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, index=True)
    display_name = Column(String)
    avatar_url = Column(String)
    is_admin = Column(Boolean, default=False)

    # optional: if this user is also a tipster account
    tipster_id = Column(Integer, ForeignKey("tipsters.id", ondelete="SET NULL"), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)

    # ⭐️ NEW: Stripe + premium
    stripe_customer_id = Column(String, nullable=True)
    is_premium = Column(Boolean, default=False)
    premium_activated_at = Column(DateTime, nullable=True)

    follows = relationship("UserFollow", back_populates="user", cascade="all, delete-orphan")
    subscriptions = relationship("TipsterSubscription", back_populates="user", cascade="all, delete-orphan")
    bets = relationship("UserBet", back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (Index("ix_users_uid_email", "firebase_uid", "email"),)


class UserFollow(Base):
    __tablename__ = "user_follows"

    id = Column(BigInteger, primary_key=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    tipster_id = Column(Integer, ForeignKey("tipsters.id", ondelete="CASCADE"), index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="follows")
    tipster = relationship("Tipster")

    __table_args__ = (
        UniqueConstraint("user_id", "tipster_id", name="uq_user_follow"),
        Index("ix_follow_user_tipster", "user_id", "tipster_id"),
    )


class UserBet(Base):
    __tablename__ = "user_bets"

    id = Column(BigInteger, primary_key=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    fixture_id = Column(BigInteger, ForeignKey("fixtures.id", ondelete="SET NULL"), index=True, nullable=True)
    source_tipster_id = Column(Integer, ForeignKey("tipsters.id", ondelete="SET NULL"), nullable=True)

    market = Column(String, index=True)     # "O2.5","BTTS_Y","HOME_WIN", etc
    bookmaker = Column(String, index=True)
    price = Column(Float, nullable=False)
    stake = Column(Float, nullable=False, default=1.0)
    placed_at = Column(DateTime, default=datetime.utcnow, index=True)

    result = Column(String, nullable=True)  # "WON","LOST","VOID" (null = unsettled)
    ret = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)

    user = relationship("User", back_populates="bets")
    fixture = relationship("Fixture")
    source_tipster = relationship("Tipster")

    __table_args__ = (Index("ix_userbets_user_time", "user_id", "placed_at"),)


class TipsterSubscription(Base):
    __tablename__ = "tipster_subscriptions"

    id = Column(BigInteger, primary_key=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    tipster_id = Column(Integer, ForeignKey("tipsters.id", ondelete="CASCADE"), index=True, nullable=False)

    plan_name = Column(String, nullable=True)       # e.g. "Monthly"
    price_cents = Column(Integer, nullable=True)    # 1500 = £15.00
    status = Column(String, nullable=False, default="active")  # active|canceled|past_due
    provider = Column(String, nullable=True, default="manual") # "stripe" later
    provider_sub_id = Column(String, nullable=True)

    started_at = Column(DateTime, default=datetime.utcnow, index=True)
    renews_at = Column(DateTime, nullable=True, index=True)
    canceled_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="subscriptions")
    tipster = relationship("Tipster")

    __table_args__ = (
        UniqueConstraint("user_id", "tipster_id", name="uq_user_tipster_sub"),
        Index("ix_subs_status", "status"),
    )

class TipsterFollow(Base):
    __tablename__ = "tipster_follows"

    id = Column(Integer, primary_key=True)
    tipster_id = Column(Integer, ForeignKey("tipsters.id"), nullable=False, index=True)
    follower_email = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    tipster = relationship("Tipster", backref="followers")


class PlayerPropModel(Base):
    __tablename__ = "player_prop_models"

    id = Column(Integer, primary_key=True)
    player_odds_id = Column(Integer, ForeignKey("player_odds.id", ondelete="CASCADE"), index=True)

    model_source = Column(String, nullable=False)   # e.g. "player_form_v1"
    prob = Column(Float, nullable=False)            # 0..1
    fair_price = Column(Float, nullable=False)      # 1/prob
    edge = Column(Float, nullable=False)            # prob * price - 1

    is_premium_only = Column(Boolean, default=True, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    player_odds = relationship("PlayerOdds", backref="models")