"""
Emergency Model Save Script

Manually saves the model with correct metadata formatting.
Use this if model_training.py failed at the save step.
"""

import joblib
import json
import time
from pathlib import Path
import numpy as np

# Paths
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "svd_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# Your training results (from terminal output)
TRAIN_RMSE = 0.6695
TEST_RMSE = 0.7717
TRAINING_TIME = 306.82
TARGET_RMSE = 0.87

HYPERPARAMETERS = {
    'n_factors': 100,
    'n_epochs': 20,
    'lr_all': 0.005,
    'reg_all': 0.02,
    'random_state': 42
}

print("Emergency Model Save Script")
print("=" * 70)

# Check if model file exists (might have been saved despite error)
if MODEL_PATH.exists():
    print(f"✅ Model file already exists: {MODEL_PATH}")
    model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"   Size: {model_size_mb:.2f} MB")
else:
    print(f"❌ Model file not found: {MODEL_PATH}")
    print("   The model needs to be re-trained.")

# Save metadata with proper type conversion
print("\nSaving metadata...")
metadata = {
    'train_rmse': float(TRAIN_RMSE),
    'test_rmse': float(TEST_RMSE),
    'training_time_seconds': float(TRAINING_TIME),
    'hyperparameters': {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v 
                       for k, v in HYPERPARAMETERS.items()},
    'target_rmse': float(TARGET_RMSE),
    'success': bool(TEST_RMSE < TARGET_RMSE),
    'model_type': 'SVD (Surprise)',
    'trained_on': time.strftime('%Y-%m-%d %H:%M:%S')
}

with open(METADATA_PATH, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Metadata saved: {METADATA_PATH}")

# Print summary
print("\n" + "=" * 70)
print("TRAINING RESULTS SUMMARY")
print("=" * 70)
print(f"Training RMSE: {TRAIN_RMSE}")
print(f"Test RMSE: {TEST_RMSE}")
print(f"Target RMSE: {TARGET_RMSE}")
print(f"Success: {'✅ YES' if TEST_RMSE < TARGET_RMSE else '❌ NO'}")
print(f"Training Time: {TRAINING_TIME:.2f} seconds ({TRAINING_TIME/60:.2f} minutes)")

if MODEL_PATH.exists():
    print("\n🎉 Model is ready to use!")
    print(f"   Run: python scripts/validate_system.py")
else:
    print("\n⚠️ Model file missing - need to check if it was saved")
