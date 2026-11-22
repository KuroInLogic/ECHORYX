
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.popularity_service import build_popularity_ranking


if __name__ == "__main__":
    ranking = build_popularity_ranking()

    print(f"Tracks ranked: {len(ranking)}")
    print("\nTop 10 popular tracks:\n")

    print(
        ranking[
            ["artist_name", "track_name", "listener_count", "total_plays"]
        ].head(10).to_string(index=False)
    )

    print("\nPopularity ranking saved successfully.")
