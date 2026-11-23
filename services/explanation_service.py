
def explain_hybrid_recommendation(row):
    """
    Generate a human-readable explanation for a hybrid recommendation.

    The scores are rank-based fusion scores, not probabilities.
    """

    components = {
        "User-CF": float(row.get("user_cf_score", 0) or 0),
        "NeuMF": float(row.get("ncf_score", 0) or 0),
        "Popularity": float(row.get("popularity_score", 0) or 0),
    }

    explanations = []

    # Identify the strongest contributing recommendation signal.
    strongest = max(components, key=components.get)

    explanation_text = {
        "User-CF": (
            "Listeners with similar music tastes have shown "
            "interest in this track."
        ),
        "NeuMF": (
            "The neural recommendation model identified "
            "a strong preference match for this track."
        ),
        "Popularity": (
            "This track has strong listening activity "
            "among users in the dataset."
        ),
    }

    explanations.append(explanation_text[strongest])

    # Add supporting signals.
    supporting = [
        name for name, score in components.items()
        if name != strongest and score >= 0.5
    ]

    if supporting:
        explanations.append(
            "It also receives support from "
            + ", ".join(supporting)
            + "."
        )

    return {
        "strongest_signal": strongest,
        "component_scores": components,
        "hybrid_score": float(row.get("hybrid_score", 0) or 0),
        "explanation": " ".join(explanations),
    }
