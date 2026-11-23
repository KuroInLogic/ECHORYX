
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from services.hybrid_service import _apply_artist_diversity


# Simulated recommendation list
recommendations = pd.DataFrame({
    "track_idx": [1, 2, 3, 4, 5, 6, 7],
    "track_name": [
        "Track A", "Track B", "Track C",
        "Track D", "Track E", "Track F", "Track G"
    ],
    "artist_name": [
        "Artist X", "Artist X", "Artist X",
        "Artist Y", "Artist Y", "Artist Z", "Artist W"
    ],
    "hybrid_score": [
        0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70
    ]
})

print("\nTEST: Artist diversity")

result = _apply_artist_diversity(
    recommendations,
    n=5,
    max_per_artist=2
)

# Check recommendation count
assert len(result) == 5, "Incorrect recommendation count."

# Check artist limit
artist_counts = result["artist_name"].value_counts()

assert artist_counts.max() <= 2, (
    "More than two tracks from the same artist were recommended."
)

# Check ranking order is preserved
assert result["hybrid_score"].is_monotonic_decreasing, (
    "Original recommendation ranking was not preserved."
)

print("PASS: Artist limit respected.")
print("PASS: Recommendation ranking preserved.")
print("\nFinal recommendations:")

print(result[["track_name", "artist_name", "hybrid_score"]])

print("\nAll diversity tests passed!")
