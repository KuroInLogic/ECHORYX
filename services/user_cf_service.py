
import pandas as pd


def recommend_for_user(
    user_idx,
    train,
    tracks,
    n=10,
    similar_user_limit=50
):
    """
    User-Based Collaborative Filtering using cosine similarity.

    Parameters:
        user_idx: Target Last.fm user ID.
        train: Training interaction DataFrame.
        tracks: Track metadata DataFrame.
        n: Number of recommendations.
        similar_user_limit: Number of similar users to consider.

    Returns:
        DataFrame containing personalised recommendations.
    """

    # 1. Get the target user's listening history.
    user_history = train[
        train["user_idx"] == user_idx
    ]

    if user_history.empty:
        return pd.DataFrame()

    listened_tracks = set(user_history["track_idx"])
    target_track_count = len(listened_tracks)

    # 2. Find other users who listened to the same tracks.
    overlapping = train[
        train["track_idx"].isin(listened_tracks)
        & (train["user_idx"] != user_idx)
    ]

    if overlapping.empty:
        return pd.DataFrame()

    # 3. Calculate the number of shared tracks per user.
    overlap_counts = (
        overlapping.groupby("user_idx")["track_idx"]
        .nunique()
    )

    # 4. Calculate each user's total unique track count.
    user_track_counts = (
        train.groupby("user_idx")["track_idx"]
        .nunique()
    )

    # 5. Calculate cosine similarity.
    similarities = overlap_counts / (
        target_track_count
        * user_track_counts.loc[overlap_counts.index]
    ) ** 0.5

    # 6. Select the most similar listeners.
    similar_users = similarities.nlargest(
        similar_user_limit
    )

    if similar_users.empty:
        return pd.DataFrame()

    # 7. Collect tracks from similar listeners,
    # excluding tracks already heard by the target user.
    candidates = train[
        train["user_idx"].isin(similar_users.index)
        & ~train["track_idx"].isin(listened_tracks)
    ].copy()

    if candidates.empty:
        return pd.DataFrame()

    # 8. Weight listening activity by user similarity.
    candidates["similarity"] = (
        candidates["user_idx"].map(similar_users)
    )

    candidates["weighted_score"] = (
        candidates["play_count"]
        * candidates["similarity"]
    )

    # 9. Aggregate recommendation scores.
    recommendations = (
        candidates.groupby("track_idx")
        .agg(
            listener_count=("user_idx", "nunique"),
            total_plays=("play_count", "sum"),
            preference_score=("weighted_score", "sum")
        )
        .reset_index()
    )

    # 10. Attach track and artist information.
    recommendations = recommendations.merge(
        tracks[
            ["track_idx", "artist_name", "track_name"]
        ],
        on="track_idx",
        how="left"
    )

    # 11. Rank and return recommendations.
    recommendations = recommendations.sort_values(
        [
            "preference_score",
            "listener_count",
            "total_plays"
        ],
        ascending=[False, False, False]
    )

    return recommendations.head(n).reset_index(drop=True)
