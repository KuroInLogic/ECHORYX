
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.splitting_service import create_temporal_split


if __name__ == "__main__":
    print("\n========== ECHORYX TEMPORAL SPLIT ==========\n")

    results = create_temporal_split()

    for key, value in results.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    print("\nTrain and test datasets saved successfully.")
