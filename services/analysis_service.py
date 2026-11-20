import sqlite3
import tempfile
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

from services.data_service import load_music_data


def analyze_user_activity():

    user_plays = Counter()

    print("Processing listening records...")

    for chunk in load_music_data():

        counts = chunk["user_id"].value_counts()
        user_plays.update(counts.to_dict())

    play_counts = list(user_plays.values())

    activity_groups = {
        "Below 5,000": 0,
        "5,000 - 10,000": 0,
        "10,000 - 20,000": 0,
        "20,000 - 50,000": 0,
        "50,000 - 100,000": 0,
        "Above 100,000": 0
    }

    for plays in play_counts:

        if plays < 5000:
            activity_groups["Below 5,000"] += 1
        elif plays < 10000:
            activity_groups["5,000 - 10,000"] += 1
        elif plays < 20000:
            activity_groups["10,000 - 20,000"] += 1
        elif plays < 50000:
            activity_groups["20,000 - 50,000"] += 1
        elif plays < 100000:
            activity_groups["50,000 - 100,000"] += 1
        else:
            activity_groups["Above 100,000"] += 1

    return {
        "total_users": len(user_plays),
        "total_plays": sum(play_counts),
        "average_plays": np.mean(play_counts),
        "median_plays": np.median(play_counts),
        "minimum_plays": min(play_counts),
        "maximum_plays": max(play_counts),
        "percentile_25": np.percentile(play_counts, 25),
        "percentile_75": np.percentile(play_counts, 75),
        "top_users": user_plays.most_common(10),
        "activity_groups": activity_groups
    }



from collections import Counter, defaultdict

from services.data_service import load_music_data


def analyze_track_diversity():

    user_tracks = defaultdict(set)
    track_users = Counter()

    total_records = 0

    print("Analysing unique user-track interactions...")

    for chunk in load_music_data():

        total_records += len(chunk)

        # Use track ID when available.
        # Fall back to track name + artist name.
        track_keys = []

        for row in chunk.itertuples(index=False):
            if pd.notna(row.track_id) and row.track_id.strip():
                track_key = row.track_id
            else:
                track_key = (
                    f"{row.artist_name.strip().lower()}::"
                    f"{row.track_name.strip().lower()}"
                )

            track_keys.append(track_key)

            user_tracks[row.user_id].add(track_key)

        # Count distinct users per track within each chunk.
        interactions = set(zip(chunk["user_id"], track_keys))

        for user_id, track_key in interactions:
            track_users[track_key] += 1

    tracks_per_user = [
        len(tracks) for tracks in user_tracks.values()
    ]

    return {
        "total_records": total_records,
        "total_users": len(user_tracks),
        "unique_tracks": len(track_users),
        "average_tracks_per_user": sum(tracks_per_user) / len(tracks_per_user),
        "median_tracks_per_user": np.median(tracks_per_user),
        "minimum_tracks_per_user": min(tracks_per_user),
        "maximum_tracks_per_user": max(tracks_per_user),
        "top_diverse_users": sorted(
            [(user, len(tracks)) for user, tracks in user_tracks.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10],
        "most_shared_tracks": track_users.most_common(10)
    }


def analyze_matrix_sparsity():

    print("Building unique user-track interactions...")

    with tempfile.TemporaryDirectory() as temp_dir:

        db_path = Path(temp_dir) / "echoryx_analysis.db"
        connection = sqlite3.connect(db_path)

        connection.execute("""
            CREATE TABLE interactions (
                user_id TEXT NOT NULL,
                track_key TEXT NOT NULL,
                PRIMARY KEY (user_id, track_key)
            )
        """)

        try:
            for chunk in load_music_data():

                interactions = []

                for row in chunk.itertuples(index=False):

                    if pd.notna(row.track_id) and row.track_id.strip():
                        track_key = row.track_id.strip()
                    else:
                        artist = (
                            row.artist_name.strip().lower()
                            if pd.notna(row.artist_name) else ""
                        )
                        track = (
                            row.track_name.strip().lower()
                            if pd.notna(row.track_name) else ""
                        )
                        track_key = f"{artist}::{track}"

                    interactions.append((row.user_id, track_key))

                connection.executemany(
                    "INSERT OR IGNORE INTO interactions VALUES (?, ?)",
                    interactions
                )

                connection.commit()

            total_users = connection.execute(
                "SELECT COUNT(DISTINCT user_id) FROM interactions"
            ).fetchone()[0]

            total_tracks = connection.execute(
                "SELECT COUNT(DISTINCT track_key) FROM interactions"
            ).fetchone()[0]

            unique_interactions = connection.execute(
                "SELECT COUNT(*) FROM interactions"
            ).fetchone()[0]

            listener_counts = connection.execute("""
                SELECT listener_count, COUNT(*)
                FROM (
                    SELECT track_key, COUNT(*) AS listener_count
                    FROM interactions
                    GROUP BY track_key
                )
                GROUP BY listener_count
                ORDER BY listener_count
            """).fetchall()

            total_possible = total_users * total_tracks

            density = unique_interactions / total_possible
            sparsity = 1 - density

            tracks_by_listener_count = {
                count: tracks
                for count, tracks in listener_counts
            }

            return {
                "total_users": total_users,
                "total_tracks": total_tracks,
                "unique_interactions": unique_interactions,
                "total_possible": total_possible,
                "density": density,
                "sparsity": sparsity,
                "tracks_by_listener_count": tracks_by_listener_count
            }

        finally:
            connection.close()



def analyze_filtering_thresholds():

    print("Building interaction database for threshold analysis...")

    with tempfile.TemporaryDirectory() as temp_dir:

        db_path = Path(temp_dir) / "echoryx_thresholds.db"
        connection = sqlite3.connect(db_path)

        connection.execute("""
            CREATE TABLE interactions (
                user_id TEXT NOT NULL,
                track_key TEXT NOT NULL,
                PRIMARY KEY (user_id, track_key)
            )
        """)

        try:
            # Build unique user-track interactions
            for chunk in load_music_data():

                interactions = []

                for row in chunk.itertuples(index=False):

                    if pd.notna(row.track_id) and row.track_id.strip():
                        track_key = row.track_id.strip()
                    else:
                        artist = (
                            row.artist_name.strip().lower()
                            if pd.notna(row.artist_name) else ""
                        )
                        track = (
                            row.track_name.strip().lower()
                            if pd.notna(row.track_name) else ""
                        )
                        track_key = f"{artist}::{track}"

                    interactions.append((row.user_id, track_key))

                connection.executemany(
                    "INSERT OR IGNORE INTO interactions VALUES (?, ?)",
                    interactions
                )
                connection.commit()

            total_users, total_tracks, total_interactions = connection.execute("""
                SELECT
                    COUNT(DISTINCT user_id),
                    COUNT(DISTINCT track_key),
                    COUNT(*)
                FROM interactions
            """).fetchone()

            # Track filtering experiment
            track_thresholds = [1, 2, 5, 10, 20, 50]
            track_results = []

            for threshold in track_thresholds:

                tracks, interactions = connection.execute("""
                    SELECT
                        COUNT(*),
                        COALESCE(SUM(listener_count), 0)
                    FROM (
                        SELECT track_key, COUNT(*) AS listener_count
                        FROM interactions
                        GROUP BY track_key
                        HAVING COUNT(*) >= ?
                    )
                """, (threshold,)).fetchone()

                users = connection.execute("""
                    SELECT COUNT(DISTINCT user_id)
                    FROM interactions
                    WHERE track_key IN (
                        SELECT track_key
                        FROM interactions
                        GROUP BY track_key
                        HAVING COUNT(*) >= ?
                    )
                """, (threshold,)).fetchone()[0]

                track_results.append({
                    "threshold": threshold,
                    "tracks": tracks,
                    "users": users,
                    "interactions": interactions
                })

            # User filtering experiment
            user_thresholds = [100, 500, 1000, 2500, 5000]
            user_results = []

            for threshold in user_thresholds:

                users, interactions = connection.execute("""
                    SELECT COUNT(*), COALESCE(SUM(track_count), 0)
                    FROM (
                        SELECT user_id, COUNT(*) AS track_count
                        FROM interactions
                        GROUP BY user_id
                        HAVING COUNT(*) >= ?
                    )
                """, (threshold,)).fetchone()

                tracks = connection.execute("""
                    SELECT COUNT(DISTINCT track_key)
                    FROM interactions
                    WHERE user_id IN (
                        SELECT user_id
                        FROM interactions
                        GROUP BY user_id
                        HAVING COUNT(*) >= ?
                    )
                """, (threshold,)).fetchone()[0]

                user_results.append({
                    "threshold": threshold,
                    "users": users,
                    "tracks": tracks,
                    "interactions": interactions
                })

            return {
                "total_users": total_users,
                "total_tracks": total_tracks,
                "total_interactions": total_interactions,
                "track_results": track_results,
                "user_results": user_results
            }

        finally:
            connection.close()
