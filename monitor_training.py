"""Monitor training progress by checking model file sizes and timestamps"""
import os
from pathlib import Path
from datetime import datetime
import time

models_dir = Path("models")
target_files = [
    "svd_model.pkl",
    "svd_model_sklearn.pkl",
    "user_knn_model.pkl",
    "item_knn_model.pkl"
]

print("CineMatch Model Training Progress Monitor")
print("=" * 60)
print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
print("\nMonitoring models directory for new/updated files...")
print("=" * 60)

while True:
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status:")
    
    for filename in target_files:
        filepath = models_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024**2)
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
            age = (datetime.now() - mtime).total_seconds() / 60
            print(f"  [OK] {filename}: {size_mb:.1f} MB (modified {age:.1f}min ago)")
        else:
            print(f"  [ ] {filename}: Not created yet")
    
    time.sleep(60)  # Check every minute
