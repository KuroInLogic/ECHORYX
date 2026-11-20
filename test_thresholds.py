
from services.analysis_service import analyze_filtering_thresholds

print("Starting ECHORYX filtering threshold experiment...")
print("This may take several minutes.\n")

results = analyze_filtering_thresholds()

print("\n========== FILTERING EXPERIMENT ==========")

print("\nOriginal Dataset:")
print(f"Users: {results['total_users']:,}")
print(f"Tracks: {results['total_tracks']:,}")
print(f"Interactions: {results['total_interactions']:,}")

print("\n--- TRACK FILTERING ---")
print("Minimum listeners | Users | Tracks | Interactions")

for row in results["track_results"]:
    print(
        f"{row['threshold']:>17} | "
        f"{row['users']:>5,} | "
        f"{row['tracks']:>7,} | "
        f"{row['interactions']:>12,}"
    )

print("\n--- USER FILTERING ---")
print("Minimum unique tracks | Users | Tracks | Interactions")

for row in results["user_results"]:
    print(
        f"{row['threshold']:>20,} | "
        f"{row['users']:>5,} | "
        f"{row['tracks']:>7,} | "
        f"{row['interactions']:>12,}"
    )

print("\n==========================================")
