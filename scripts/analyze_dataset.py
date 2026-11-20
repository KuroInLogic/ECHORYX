import sys
import argparse
from pathlib import Path

# Make the project root accessible to Python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.data_service import inspect_dataset
from services.analysis_service import (
    analyze_user_activity,
    analyze_track_diversity,
    analyze_matrix_sparsity,
    analyze_filtering_thresholds
)


def show_dataset():
    results = inspect_dataset()

    print("\n========== DATASET OVERVIEW ==========")
    for key, value in results.items():
        print(f"{key}: {value:,}")


def show_activity():
    results = analyze_user_activity()

    print("\n========== USER ACTIVITY ==========")
    print(f"Total users: {results['total_users']:,}")
    print(f"Total plays: {results['total_plays']:,}")
    print(f"Average plays: {results['average_plays']:,.2f}")
    print(f"Median plays: {results['median_plays']:,.2f}")
    print(f"Minimum plays: {results['minimum_plays']:,}")
    print(f"Maximum plays: {results['maximum_plays']:,}")

    print("\nTop 10 active users:")
    for user, plays in results["top_users"]:
        print(f"{user}: {plays:,}")

    print("\nActivity distribution:")
    for group, count in results["activity_groups"].items():
        print(f"{group}: {count} users")
    print("\nUser activity percentiles:")
    print(f"25th percentile: {results['percentile_25']:,.2f}")
    print(f"75th percentile: {results['percentile_75']:,.2f}")


def show_diversity():
    results = analyze_track_diversity()

    print("\n========== TRACK DIVERSITY ==========")
    print(f"Total records: {results['total_records']:,}")
    print(f"Total users: {results['total_users']:,}")
    print(f"Unique tracks: {results['unique_tracks']:,}")
    print(
        f"Average tracks per user: "
        f"{results['average_tracks_per_user']:,.2f}"
    )
    print(
        f"Median tracks per user: "
        f"{results['median_tracks_per_user']:,.2f}"
    )

    print("\nTop 10 diverse users:")
    for user, tracks in results["top_diverse_users"]:
        print(f"{user}: {tracks:,}")


def show_sparsity():
    results = analyze_matrix_sparsity()

    print("\n========== MATRIX SPARSITY ==========")
    print(f"Users: {results['total_users']:,}")
    print(f"Tracks: {results['total_tracks']:,}")
    print(f"Interactions: {results['unique_interactions']:,}")
    print(f"Possible interactions: {results['total_possible']:,}")
    print(f"Density: {results['density'] * 100:.6f}%")
    print(f"Sparsity: {results['sparsity'] * 100:.6f}%")


def show_thresholds():
    results = analyze_filtering_thresholds()

    print("\n========== FILTERING THRESHOLDS ==========")
    print("\nTrack filtering:")
    print("Minimum listeners | Users | Tracks | Interactions")

    for row in results["track_results"]:
        print(
            f"{row['threshold']:>17} | "
            f"{row['users']:>5,} | "
            f"{row['tracks']:>7,} | "
            f"{row['interactions']:>12,}"
        )

    print("\nUser filtering:")
    print("Minimum tracks | Users | Tracks | Interactions")

    for row in results["user_results"]:
        print(
            f"{row['threshold']:>13,} | "
            f"{row['users']:>5,} | "
            f"{row['tracks']:>7,} | "
            f"{row['interactions']:>12,}"
        )


ANALYSIS_OPTIONS = {
    "dataset": show_dataset,
    "activity": show_activity,
    "diversity": show_diversity,
    "sparsity": show_sparsity,
    "thresholds": show_thresholds
}


def main():
    parser = argparse.ArgumentParser(
        description="ECHORYX dataset analysis utility"
    )

    parser.add_argument(
        "analysis",
        choices=ANALYSIS_OPTIONS.keys(),
        help="Choose the analysis to perform"
    )

    args = parser.parse_args()

    print(f"\nECHORYX | Running {args.analysis} analysis")

    ANALYSIS_OPTIONS[args.analysis]()


if __name__ == "__main__":
    main()
