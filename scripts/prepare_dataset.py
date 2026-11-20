
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.preparation_service import prepare_interactions


def main():

    print("\n========== ECHORYX DATA PREPARATION ==========")

    results = prepare_interactions(
        min_track_listeners=5,
        min_tracks_per_user=100
    )

    print("\n========== PREPARATION REPORT ==========")

    print(
        f"Original unique interactions: "
        f"{results['original_interactions']:,}"
    )

    print(f"Remaining users: {results['users']:,}")
    print(f"Remaining tracks: {results['tracks']:,}")
    print(f"Remaining interactions: {results['interactions']:,}")

    print(
        f"\nTrack listener threshold: "
        f"{results['min_track_listeners']}"
    )

    print(
        f"User track threshold: "
        f"{results['min_tracks_per_user']}"
    )

    print(f"\nSaved to: {results['output_directory']}")

    print("\n========================================")


if __name__ == "__main__":
    main()
