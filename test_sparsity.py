
from services.analysis_service import analyze_matrix_sparsity

print("Starting ECHORYX matrix sparsity analysis...")
print("This may take several minutes.\n")

results = analyze_matrix_sparsity()

print("\n========== ECHORYX SPARSITY REPORT ==========")

print(f"\nTotal users: {results['total_users']:,}")
print(f"Total unique tracks: {results['total_tracks']:,}")

print(
    f"Unique user-track interactions: "
    f"{results['unique_interactions']:,}"
)

print(f"Possible interactions: {results['total_possible']:,}")

print("\nInteraction Matrix:")
print(f"Density: {results['density'] * 100:.6f}%")
print(f"Sparsity: {results['sparsity'] * 100:.6f}%")

print("\nTracks by Number of Listeners:")

for listeners, tracks in results["tracks_by_listener_count"].items():
    print(f"{listeners} listener(s): {tracks:,} tracks")

print("\n============================================")
