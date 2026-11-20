
from services.analysis_service import analyze_user_activity

print("Analyzing user listening behaviour...")
print("This may take a few minutes.\n")

results = analyze_user_activity()

print("\n========== ECHORYX USER ANALYSIS ==========")

print(f"\nTotal users: {results['total_users']}")
print(f"Total listening events: {results['total_plays']:,}")

print("\nListening Activity Statistics:")
print(f"Average plays: {results['average_plays']:,.2f}")
print(f"Median plays: {results['median_plays']:,.2f}")
print(f"Minimum plays: {results['minimum_plays']:,}")
print(f"Maximum plays: {results['maximum_plays']:,}")

print("\nUser Activity Percentiles:")
print(f"25th percentile: {results['percentile_25']:,.2f}")
print(f"75th percentile: {results['percentile_75']:,.2f}")

print("\nTop 10 Most Active Users:")
for user, plays in results["top_users"]:
    print(f"{user}: {plays:,}")

print("\nUser Activity Distribution:")
for group, count in results["activity_groups"].items():
    print(f"{group}: {count} users")

print("\n===========================================")

