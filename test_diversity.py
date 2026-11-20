
from services.analysis_service import analyze_track_diversity

print("Starting ECHORYX track diversity analysis...")
print("This may take a few minutes.\n")

results = analyze_track_diversity()

print("\n========== ECHORYX TRACK DIVERSITY ==========")

print(f"\nTotal records: {results['total_records']:,}")
print(f"Total users: {results['total_users']:,}")
print(f"Unique tracks: {results['unique_tracks']:,}")

print("\nTracks Per User:")
print(f"Average: {results['average_tracks_per_user']:,.2f}")
print(f"Median: {results['median_tracks_per_user']:,.2f}")
print(f"Minimum: {results['minimum_tracks_per_user']:,}")
print(f"Maximum: {results['maximum_tracks_per_user']:,}")

print("\nTop 10 Users by Track Diversity:")
for user, tracks in results["top_diverse_users"]:
    print(f"{user}: {tracks:,} unique tracks")

print("\nTop 10 Tracks by Number of Listeners:")
for track, users in results["most_shared_tracks"]:
    print(f"{track}: {users:,} users")

print("\n============================================")
