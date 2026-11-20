
from services.analysis_service import analyze_listening_data

print("Analyzing Last.fm listening behaviour...")
print("This may take a few minutes.\n")

results = analyze_listening_data()

print("\n========== ECHORYX LISTENING ANALYSIS ==========")

print("\nTop 10 Artists:")
for artist, plays in results["top_artists"]:
    print(f"{artist}: {plays}")

print("\nTop 10 Tracks:")
for track, plays in results["top_tracks"]:
    print(f"{track[0]} - {track[1]}: {plays}")

print("\nTop 10 Most Active Users:")
for user, plays in results["top_users"]:
    print(f"{user}: {plays}")

print(f"\nTotal users: {results['total_users']}")
print(f"Average plays per user: {results['average_plays_per_user']:.2f}")

print("\n================================================")
