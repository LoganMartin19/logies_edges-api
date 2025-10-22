def confidence_from_prob(prob: float) -> str:
    """Return confidence tier based on probability."""
    if prob >= 0.65:
        return "High"
    elif prob >= 0.55:
        return "Medium"
    else:
        return "Low"