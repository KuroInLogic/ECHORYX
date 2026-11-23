import pandas as pd
from services.profile_service import recommend_from_preferences
from services.user_cf_service import (
    recommend_for_user as recommend_with_user_cf
)
from services.ncf_service import (
    recommend_for_user as recommend_with_ncf
)


def _rank_scores(recommendations, score_column):
    """
    Convert recommendation rankings into comparable scores.
    The highest-ranked track receives the highest score.
    """

    if recommendations.empty:
        return pd.DataFrame(columns=["track_idx", score_column])

    ranked = recommendations[["track_idx"]].copy()

    ranked = ranked.drop_duplicates("track_idx").reset_index(drop=True)

    count = len(ranked)

    if count == 1:
        ranked[score_column] = 1.0
    else:
        ranked[score_column] = [
            1 - (rank / (count - 1))
            for rank in range(count)
        ]

    return ranked


def _get_popularity_recommendations(user_idx, train, n):
    """Generate popularity recommendations using the supplied training data."""

    listened_tracks = set(
        train.loc[
            train["user_idx"] == user_idx,
            "track_idx"
        ]
    )

    popularity = (
        train.groupby("track_idx")
        .agg(
            listener_count=("user_idx", "nunique"),
            total_plays=("play_count", "sum")
        )
        .reset_index()
    )

    popularity = popularity[
        ~popularity["track_idx"].isin(listened_tracks)
    ]

    popularity = popularity.sort_values(
        ["listener_count", "total_plays"],
        ascending=[False, False]
    )

    return popularity.reset_index(drop=True)


def _apply_artist_diversity(recommendations, n, max_per_artist=2):
    """
    Limit the number of recommendations from each artist
    while preserving the original recommendation ranking.
    """

    if recommendations.empty:
        return recommendations

    if "artist_name" not in recommendations.columns:
        return recommendations.head(n).reset_index(drop=True)

    selected = []
    artist_counts = {}

    for _, row in recommendations.iterrows():
        artist = row["artist_name"]

        # Treat missing artist names as separate tracks.
        if pd.isna(artist):
            artist = f"unknown_{row['track_idx']}"

        count = artist_counts.get(artist, 0)

        if count < max_per_artist:
            selected.append(row)
            artist_counts[artist] = count + 1

        if len(selected) == n:
            break

    return pd.DataFrame(selected).reset_index(drop=True)


def recommend_hybrid(
    user_idx,
    train,
    tracks,
    n=10,
    candidate_pool=200,
    weights=None,
    min_interactions=5,
    max_per_artist=None,
    selected_artists=None,
    discovery_intensity=0.20
):
    """
    Generate hybrid recommendations using User-CF, NeuMF
    and popularity-based recommendation.

    Scores are converted into rank-based scores before fusion.
    """

    if n <= 0:
        raise ValueError("n must be greater than zero.")

    if candidate_pool <= 0:
        raise ValueError("candidate_pool must be greater than zero.")

    if not 0 <= discovery_intensity <= 1:
        raise ValueError(
            "discovery_intensity must be between 0 and 1."
        )

    required_columns = {
        "user_idx",
        "track_idx",
        "play_count"
    }

    if not required_columns.issubset(train.columns):
        raise ValueError(
            "Training data must contain user_idx, track_idx "
            "and play_count."
        )

    if weights is None:
        weights = {
            "user_cf": 0.60,
            "ncf": 0.25,
            "popularity": 0.15
        }

    expected_models = {"user_cf", "ncf", "popularity"}

    if set(weights) != expected_models:
        raise ValueError(
            "Weights must contain user_cf, ncf and popularity."
        )

    if any(value < 0 for value in weights.values()):
        raise ValueError("Weights cannot be negative.")

    if sum(weights.values()) <= 0:
        raise ValueError("At least one weight must be positive.")

    # Normalise weights so they add up to 1.
    total_weight = sum(weights.values())

    weights = {
        key: value / total_weight
        for key, value in weights.items()
    }

    # -------------------------------------------------
    # ARTIST PREFERENCE SIGNAL
    # -------------------------------------------------

    preference_scores = pd.DataFrame(
        columns=["track_idx", "artist_preference_score"]
    )

    if selected_artists and discovery_intensity > 0:

        preference_recommendations, preference_stats = (
            recommend_from_preferences(
                selected_artists=selected_artists,
                train=train,
                tracks=tracks,
                n=candidate_pool
            )
        )

        if not preference_recommendations.empty:

            preference_scores = _rank_scores(
                preference_recommendations,
                "artist_preference_score"
            )

    # -------------------------------------------------
    # COLD-START FALLBACK
    # -------------------------------------------------

    
    user_history = train[
        train["user_idx"] == user_idx
    ]

    if len(user_history) < min_interactions:

        popularity = _get_popularity_recommendations(
            user_idx=user_idx,
            train=train,
            n=candidate_pool
        )

        popularity = _rank_scores(
            popularity,
            "popularity_score"
        )

        if popularity.empty:
            return pd.DataFrame()

        popularity = popularity.merge(
            preference_scores,
            on="track_idx",
            how="left"
        )

        popularity["artist_preference_score"] = (
            popularity["artist_preference_score"].fillna(0)
        )

        popularity["user_cf_score"] = 0.0
        popularity["ncf_score"] = 0.0

        popularity["hybrid_score"] = (
            (1 - discovery_intensity)
            * popularity["popularity_score"]
            + discovery_intensity
            * popularity["artist_preference_score"]
        )

        metadata = tracks[
            ["track_idx", "track_name", "artist_name"]
        ].drop_duplicates("track_idx")

        popularity = popularity.merge(
            metadata,
            on="track_idx",
            how="left"
        )

        popularity["preference_score"] = (
            popularity["hybrid_score"]
        )

        popularity["fallback_used"] = True

        popularity["recommendation_reason"] = (
            "Popularity-based fallback due to limited "
            "listening history."
        )

        if max_per_artist is not None:
            popularity = _apply_artist_diversity(
                popularity,
                n,
                max_per_artist
            )
        else:
            popularity = popularity.head(n).reset_index(drop=True)

        return popularity



    # 1. Get User-CF recommendations.
    user_cf = recommend_with_user_cf(
        user_idx=user_idx,
        train=train,
        tracks=tracks,
        n=candidate_pool
    )

    # 2. Get NeuMF recommendations.
    ncf = recommend_with_ncf(
        user_idx=user_idx,
        train=train,
        tracks=tracks,
        n=candidate_pool
    )

    # 3. Get popularity recommendations.
    popularity = _get_popularity_recommendations(
        user_idx=user_idx,
        train=train,
        n=candidate_pool
    )

    # 4. Convert each model's ranking into comparable scores.
    cf_scores = _rank_scores(user_cf, "user_cf_score")
    ncf_scores = _rank_scores(ncf, "ncf_score")
    popularity_scores = _rank_scores(
        popularity,
        "popularity_score"
    )

    # 5. Combine candidates from all three models.
    candidates = pd.concat(
        [
            cf_scores[["track_idx"]],
            preference_scores[["track_idx"]],
            ncf_scores[["track_idx"]],
            popularity_scores[["track_idx"]]
        ],
        ignore_index=True
    ).drop_duplicates("track_idx")

    if candidates.empty:
        return pd.DataFrame()

    # 6. Combine individual model scores.
    candidates = candidates.merge(
        cf_scores,
        on="track_idx",
        how="left"
    )

    candidates = candidates.merge(
        ncf_scores,
        on="track_idx",
        how="left"
    )

    candidates = candidates.merge(
        popularity_scores,
        on="track_idx",
        how="left"
    )

    # Merge artist preference scores.
    candidates = candidates.merge(
        preference_scores[
            ["track_idx", "artist_preference_score"]
        ],
        on="track_idx",
        how="left"
    )

    candidates["artist_preference_score"] = (
        candidates["artist_preference_score"].fillna(0.0)
    )

    score_columns = [
        "user_cf_score",
        "ncf_score",
        "popularity_score"
    ]

    candidates[score_columns] = (
        candidates[score_columns].fillna(0)
    )

    
    # 7. Calculate the weighted hybrid score.
    base_hybrid_score = (
        candidates["user_cf_score"] * weights["user_cf"]
        + candidates["ncf_score"] * weights["ncf"]
        + candidates["popularity_score"] * weights["popularity"]
    )

    # Apply artist preference influence.
    candidates["hybrid_score"] = (
        (1 - discovery_intensity) * base_hybrid_score
        + discovery_intensity
        * candidates["artist_preference_score"]
    )


    # 8. Exclude previously listened tracks.
    listened_tracks = set(
        train.loc[
            train["user_idx"] == user_idx,
            "track_idx"
        ]
    )

    candidates = candidates[
        ~candidates["track_idx"].isin(listened_tracks)
    ]

    # 9. Attach track information.
    metadata = tracks[
        ["track_idx", "track_name", "artist_name"]
    ].drop_duplicates("track_idx")

    candidates = candidates.merge(
        metadata,
        on="track_idx",
        how="left"
    )

    # 10. Rank and return recommendations.
    candidates = candidates.sort_values(
        "hybrid_score",
        ascending=False
    )

    candidates["preference_score"] = candidates["hybrid_score"]

    candidates["fallback_used"] = False
    candidates["recommendation_reason"] = (
        "Generated using User-CF, NeuMF and Popularity."
    )
    if max_per_artist is not None:
        candidates = _apply_artist_diversity(
            candidates,
            n,
            max_per_artist
        )
    else:
        candidates = candidates.head(n).reset_index(drop=True)


    return candidates
