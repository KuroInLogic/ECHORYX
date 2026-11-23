import pandas as pd


def recommend_from_preferences(
    selected_artists,
    train,
    tracks,
    n=10
):
    """
    Generate recommendations from a new user's
    favourite artists using Last.fm listening patterns.
    """

    if not selected_artists:
        return pd.DataFrame(), {
            "similar_listeners": 0,
            "matched_artists": 0
        }

    # Normalise artist names for matching.
    selected = {
        artist.strip().casefold()
        for artist in selected_artists
    }

    track_data = tracks.copy()
    track_data["artist_normalized"] = (
        track_data["artist_name"].fillna("").str.casefold()
    )

    # Find tracks belonging to selected artists.
    seed_tracks = track_data[
        track_data["artist_normalized"].isin(selected)
    ]

    seed_track_ids = set(seed_tracks["track_idx"])

    if not seed_track_ids:
        return pd.DataFrame(), {
            "similar_listeners": 0,
            "matched_artists": 0
        }

    # Find Last.fm listeners who listened to these artists.
    seed_interactions = train[
        train["track_idx"].isin(seed_track_ids)
    ]

    if seed_interactions.empty:
        return pd.DataFrame(), {
            "similar_listeners": 0,
            "matched_artists": 0
        }

    # Calculate how many selected artists each listener follows.
    user_artist_history = seed_interactions.merge(
        seed_tracks[["track_idx", "artist_normalized"]],
        on="track_idx",
        how="left"
    )

    user_artist_counts = (
        user_artist_history.groupby("user_idx")["artist_normalized"]
        .nunique()
    )

    # Listeners matching more of the user's preferences
    # receive a higher recommendation weight.
    user_weights = (
        user_artist_counts / len(selected)
    )

    similar_user_ids = set(user_weights.index)

    # Collect tracks listened to by those users.
    candidate_interactions = train[
        train["user_idx"].isin(similar_user_ids)
        & ~train["track_idx"].isin(seed_track_ids)
    ].copy()

    if candidate_interactions.empty:
        return pd.DataFrame(), {
            "similar_listeners": len(similar_user_ids),
            "matched_artists": len(seed_tracks["artist_normalized"].unique())
        }

    # Apply preference weights to listening activity.
    candidate_interactions["preference_weight"] = (
        candidate_interactions["user_idx"].map(user_weights)
    )

    candidate_interactions["weighted_plays"] = (
        candidate_interactions["play_count"]
        * candidate_interactions["preference_weight"]
    )

    # Rank tracks by listener support and weighted plays.
    recommendations = (
        candidate_interactions.groupby("track_idx")
        .agg(
            listener_count=("user_idx", "nunique"),
            total_plays=("play_count", "sum"),
            preference_score=("weighted_plays", "sum")
        )
        .reset_index()
    )

    recommendations = recommendations.merge(
        tracks[["track_idx", "artist_name", "track_name"]],
        on="track_idx",
        how="left"
    )

    recommendations = recommendations.sort_values(
        ["preference_score", "listener_count", "total_plays"],
        ascending=[False, False, False]
    ).head(n).reset_index(drop=True)

    return recommendations, {
        "similar_listeners": len(similar_user_ids),
        "matched_artists": len(seed_tracks["artist_normalized"].unique())
    }
