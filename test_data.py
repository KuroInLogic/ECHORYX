
from services.data_service import inspect_dataset

print("Starting full Last.fm dataset inspection...")
print("This may take a few minutes.\n")

results = inspect_dataset()

print("\n========== ECHORYX DATASET REPORT ==========")

for key, value in results.items():
    print(f"{key}: {value}")

print("============================================")
