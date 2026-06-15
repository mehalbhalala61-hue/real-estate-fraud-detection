"""
generate_training_stats.py — Run once to create configs/training_stats.yaml
Drift monitor ke liye city-wise price baseline save karta hai.

Run from project root:
  python generate_training_stats.py
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import pandas as pd
from src.drift_monitor import save_training_stats

print("Loading X_train...")
splits_path = Path("data/splits/X_train.parquet")

if not splits_path.exists():
    print("❌ data/splits/X_train.parquet not found!")
    print("   Day 4 notebook pehle run karo")
    sys.exit(1)

X_train = pd.read_parquet(splits_path)
print(f"✅ X_train loaded: {X_train.shape}")

out = save_training_stats(X_train, "configs/training_stats.yaml")
print(f"✅ Training stats saved → {out}")
print(f"   Cities tracked: {len(pd.read_parquet(splits_path)['city'].dropna().unique())}")
print("\n→ Drift monitor ready. Run: python src/drift_monitor.py")
