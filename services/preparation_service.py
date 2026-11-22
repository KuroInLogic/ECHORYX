
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from services.data_service import load_music_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def get_track_key(track_id, artist_name, track_name):
    """Create a consistent key, including a fallback for missing IDs."""

    if pd.notna(track_id) and str(track_id).strip():
        return f"mbid:{str(track_id).strip()}"

    artist = (
        str(artist_name).strip().casefold()
        if pd.notna(artist_name) else ""
    )
    track = (
        str(track_name).strip().casefold()
        if pd.notna(track_name) else ""
    )

    return f"name:{artist}::{track}"


def prepare_interactions(
    min_track_listeners=5,
    min_tracks_per_user=100,
    chunksize=100_000
):
    """Aggregate, filter and export the Last.fm interaction dataset."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:

        db_path = Path(temp_dir) / "echoryx_preparation.db"
        connection = sqlite3.connect(db_path)

        connection.execute("""
            CREATE TABLE interactions (
                user_id TEXT NOT NULL,
                track_key TEXT NOT NULL,
                artist_name TEXT,
                track_name TEXT,
                play_count INTEGER NOT NULL DEFAULT 0,
                last_timestamp TEXT NOT NULL,
                PRIMARY KEY (user_id, track_key)
            )
        """)

        try:
            print("Step 1: Aggregating listening events...")

            for chunk in load_music_data(chunksize):

                play_counts = Counter()
                metadata = {}
                latest_timestamps = {}

                for row in chunk.itertuples(index=False):

                    track_key = get_track_key(
                        row.track_id,
                        row.artist_name,
                        row.track_name
                    )

                    pair = (row.user_id, track_key)
                    play_counts[pair] += 1
                    timestamp = str(row.timestamp)
                    if pair not in latest_timestamps or timestamp > latest_timestamps[pair]:
                        latest_timestamps[pair] = timestamp

                    if pair not in metadata:
                        metadata[pair] = (
                            row.artist_name,
                            row.track_name
                        )

                records = [
                    (
                        user_id,
                        track_key,
                        metadata[(user_id, track_key)][0],
                        metadata[(user_id, track_key)][1],
                        count,
                        latest_timestamps[(user_id, track_key)]
                    )
                    for (user_id, track_key), count in play_counts.items()
                ]

                connection.executemany("""
                    INSERT INTO interactions
                    (user_id, track_key, artist_name, track_name, play_count, last_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, track_key)
                    DO UPDATE SET
                        play_count = play_count + excluded.play_count,
                        last_timestamp = MAX(interactions.last_timestamp, excluded.last_timestamp)
                """, records)

                connection.commit()

            original_interactions = connection.execute(
                "SELECT COUNT(*) FROM interactions"
            ).fetchone()[0]

            print(
                f"Aggregated unique interactions: "
                f"{original_interactions:,}"
            )

            print("\nStep 2: Applying combined filters...")

            iteration = 0

            while True:
                iteration += 1

                removed_tracks = connection.execute("""
                    DELETE FROM interactions
                    WHERE track_key IN (
                        SELECT track_key
                        FROM interactions
                        GROUP BY track_key
                        HAVING COUNT(*) < ?
                    )
                """, (min_track_listeners,)).rowcount

                removed_users = connection.execute("""
                    DELETE FROM interactions
                    WHERE user_id IN (
                        SELECT user_id
                        FROM interactions
                        GROUP BY user_id
                        HAVING COUNT(*) < ?
                    )
                """, (min_tracks_per_user,)).rowcount

                connection.commit()

                print(
                    f"Iteration {iteration}: "
                    f"{removed_tracks:,} track interactions and "
                    f"{removed_users:,} user interactions removed"
                )

                if removed_tracks == 0 and removed_users == 0:
                    break

            print("\nStep 3: Creating ID mappings...")

            connection.execute("""
                CREATE TABLE user_map AS
                SELECT
                    user_id,
                    ROW_NUMBER() OVER (ORDER BY user_id) - 1 AS user_idx
                FROM (
                    SELECT DISTINCT user_id FROM interactions
                )
            """)

            connection.execute("""
                CREATE TABLE track_map AS
                SELECT
                    track_key,
                    MAX(artist_name) AS artist_name,
                    MAX(track_name) AS track_name,
                    ROW_NUMBER() OVER (ORDER BY track_key) - 1 AS track_idx
                FROM interactions
                GROUP BY track_key
            """)

            connection.commit()

            users = connection.execute(
                "SELECT COUNT(*) FROM user_map"
            ).fetchone()[0]

            tracks = connection.execute(
                "SELECT COUNT(*) FROM track_map"
            ).fetchone()[0]

            interactions = connection.execute(
                "SELECT COUNT(*) FROM interactions"
            ).fetchone()[0]

            print("\nStep 4: Exporting prepared datasets...")

            user_df = pd.read_sql_query(
                "SELECT user_idx, user_id FROM user_map ORDER BY user_idx",
                connection
            )

            track_df = pd.read_sql_query("""
                SELECT track_idx, track_key, artist_name, track_name
                FROM track_map
                ORDER BY track_idx
            """, connection)

            user_df.to_parquet(
                OUTPUT_DIR / "users.parquet",
                index=False
            )

            track_df.to_parquet(
                OUTPUT_DIR / "tracks.parquet",
                index=False
            )

            query = """
                SELECT
                    um.user_idx,
                    tm.track_idx,
                    i.play_count,
                    i.last_timestamp
                FROM interactions i
                JOIN user_map um ON i.user_id = um.user_id
                JOIN track_map tm ON i.track_key = tm.track_key
                ORDER BY um.user_idx, tm.track_idx
            """

            writer = None

            for batch in pd.read_sql_query(
                query, connection, chunksize=100_000
            ):
                table = pa.Table.from_pandas(batch, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(
                        OUTPUT_DIR / "interactions.parquet",
                        table.schema
                    )

                writer.write_table(table)

            if writer is not None:
                writer.close()

            return {
                "original_interactions": original_interactions,
                "users": users,
                "tracks": tracks,
                "interactions": interactions,
                "min_track_listeners": min_track_listeners,
                "min_tracks_per_user": min_tracks_per_user,
                "output_directory": str(OUTPUT_DIR)
            }

        finally:
            connection.close()
