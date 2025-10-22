def implied_prob_from_price(price: float) -> float:
    return max(0.0001, min(1.0, 1.0/price))

def edge(prob: float, price: float) -> float:
    return prob * price - 1.0

def kelly_fraction(prob: float, price: float) -> float:
    b = price - 1.0
    if b <= 0: return 0.0
    return (b*prob - (1.0 - prob)) / b

def stake_half_kelly(bank: float, prob: float, price: float, cap=0.05) -> float:
    from math import fsum
    k = max(0.0, kelly_fraction(prob, price))
    return min(bank*cap, 0.5 * k * bank)