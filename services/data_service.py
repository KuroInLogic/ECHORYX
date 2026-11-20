
import pandas as pd
from pathlib import Path

DATA_PATH = (
    Path(__file__).parent.parent
    / "data"
    / "userid-timestamp-artid-artname-traid-traname.tsv"
)

COLUMNS = [
    "user_id",
    "timestamp",
    "artist_id",
    "artist_name",
    "track_id",
    "track_name"
]


def load_music_data(chunksize=100_000):
    """Read Last.fm data without CSV quoting issues."""

    rows = []

    with open(DATA_PATH, "r", encoding="utf-8", errors="replace") as file:
        for line in file:
            fields = line.rstrip("\r\n").split("\t")

            if len(fields) != 6:
                continue

            rows.append(fields)

            if len(rows) >= chunksize:
                yield pd.DataFrame(rows, columns=COLUMNS)
                rows = []

    if rows:
        yield pd.DataFrame(rows, columns=COLUMNS)


def inspect_dataset():
    """Inspect dataset statistics using manageable chunks."""

    total_records = 0
    unique_users = set()
    unique_artists = set()
    unique_tracks = set()

    for chunk in load_music_data():
        total_records += len(chunk)

        unique_users.update(chunk["user_id"].dropna().unique())
        unique_artists.update(chunk["artist_id"].dropna().unique())
        unique_tracks.update(chunk["track_id"].dropna().unique())

    return {
        "total_records": total_records,
        "unique_users": len(unique_users),
        "unique_artists": len(unique_artists),
        "unique_tracks": len(unique_tracks)
    }
