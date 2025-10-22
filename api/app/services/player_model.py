# api/app/services/player_model.py
from math import exp, factorial
from typing import Optional

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x into [lo, hi]."""
    return max(lo, min(hi, x))


def market_allow_factor(opp_rate: float, league_avg: float,
                        lo: float = 0.80, hi: float = 1.25) -> float:
    """
    Convert an opponent allowance rate vs league average into a multiplier.
    Example: opp allows 12.0 fouls/match, league avg 10.0 -> 1.20 (clamped).
    Safe for missing/zero inputs (returns 1.0).
    """
    try:
        opp_rate = float(opp_rate)
        league_avg = float(league_avg)
        if opp_rate <= 0 or league_avg <= 0:
            return 1.0
        return _clamp(opp_rate / league_avg, lo, hi)
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Poisson basics
# ---------------------------------------------------------------------------

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return exp(-lam) * (lam ** k) / factorial(k)


def poisson_cdf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    # naive sum is fine for small lines; we keep it simple & deterministic
    return sum(poisson_pmf(i, lam) for i in range(k + 1))


# ---------------------------------------------------------------------------
# Public probability helpers
# ---------------------------------------------------------------------------

def prob_over_xpoint5(
    per90: float,
    expected_minutes: int,
    x_half: float,
    pace_factor: float = 1.0,
    opponent_factor: float = 1.0,
    context_factor: float = 1.0,
) -> float:
    """
    Probability of exceeding X.5 events for count props (shots, fouls, tackles, etc.).
    Modeled as Poisson with λ scaled by minutes and contextual multipliers.

    Args:
        per90: player rate per 90 minutes.
        expected_minutes: projected minutes (0..120).
        x_half: line like 0.5, 1.5, 2.5 (we compute P(X > floor(x_half))).
        pace_factor: team/tempo scalar (default 1.0).
        opponent_factor: opponent allowance scalar (default 1.0).
        context_factor: any extra scalar (ref, venue, weather, etc.).

    Returns:
        Probability in [0, 1].
    """
    # guard + clamp to avoid wild swings from noisy inputs
    per90 = max(0.0, float(per90 or 0.0))
    expected_minutes = int(max(0, expected_minutes or 0))
    pace_factor = _clamp(float(pace_factor or 1.0), 0.50, 1.50)
    opponent_factor = _clamp(float(opponent_factor or 1.0), 0.50, 1.50)
    context_factor = _clamp(float(context_factor or 1.0), 0.50, 1.50)

    lam = per90 * (expected_minutes / 90.0) * pace_factor * opponent_factor * context_factor
    if lam <= 0:
        return 0.0

    k = int(x_half - 0.5)  # e.g. over 1.5 -> k=1
    return 1.0 - poisson_cdf(k, lam)


def prob_card(
    yellow_per90: float,
    expected_minutes: int,
    ref_factor: float = 1.0,
    opponent_factor: float = 1.0,
) -> float:
    """
    Probability of at least one booking (yellow) using rare-event (Poisson) approx.
    We scale the base rate by minutes, referee, and opponent context.

    Args:
        yellow_per90: player's yellow cards per 90.
        expected_minutes: projected minutes (0..120).
        ref_factor: referee scalar (card-happy vs strict) — mildly clamped.
        opponent_factor: opponent fouls-drawn/cards-drawn scalar.

    Returns:
        Probability in [0, 1].
    """
    rate90 = max(0.0, float(yellow_per90 or 0.0))
    expected_minutes = int(max(0, expected_minutes or 0))
    ref_factor = _clamp(float(ref_factor or 1.0), 0.75, 1.35)
    opponent_factor = _clamp(float(opponent_factor or 1.0), 0.80, 1.25)

    lam = rate90 * (expected_minutes / 90.0) * ref_factor * opponent_factor
    if lam <= 0:
        return 0.0
    return 1.0 - exp(-lam)


# ---------------------------------------------------------------------------
# Odds helpers
# ---------------------------------------------------------------------------

def fair_odds(p: float) -> Optional[float]:
    """Decimal fair odds from probability p."""
    try:
        if p is None or p <= 0.0 or p >= 1.0:
            return None
        return 1.0 / p
    except Exception:
        return None


def edge(p: float, price: float) -> Optional[float]:
    """
    EV edge as a fraction: (p * price - 1).
    Returns None if inputs are missing.
    """
    try:
        if p is None or price is None:
            return None
        return p * price - 1.0
    except Exception:
        return None