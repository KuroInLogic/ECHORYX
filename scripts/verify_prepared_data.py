
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "processed"

users = pd.read_parquet(DATA_DIR / "users.parquet")
tracks = pd.read_parquet(DATA_DIR / "tracks.parquet")
interactions = pd.read_parquet(DATA_DIR / "interactions.parquet")

print("\n========== PREPARED DATA VERIFICATION ==========")

print("\nUsers:")
print(f"Shape: {users.shape}")
print(users.head())

print("\nTracks:")
print(f"Shape: {tracks.shape}")
print(tracks.head())

print("\nInteractions:")
print(f"Shape: {interactions.shape}")
print(interactions.head())

print("\nData types:")
print(interactions.dtypes)

print("\nMissing values:")
print(interactions.isnull().sum())

print("\nPlay count statistics:")
print(interactions["play_count"].describe())

print("\nUnique users:", interactions["user_idx"].nunique())
print("Unique tracks:", interactions["track_idx"].nunique())

print("\n================================================")
