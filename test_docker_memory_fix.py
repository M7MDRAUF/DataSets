"""
CineMatch V2.1.1 - Docker Memory Fix Validation

Test SVD (Surprise) loading with memory-optimized sequential loading.

Author: CineMatch Development Team
Date: November 14, 2025
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.utils.memory_manager import (
    load_models_sequential,
    print_memory_report,
    get_memory_usage_mb
)

print("=" * 80)
print("CINEMATCH V2.1.1 - MEMORY-OPTIMIZED MODEL LOADING TEST")
print("=" * 80)

# Define all 5 models
model_paths = {
    'SVD (Surprise)': 'models/svd_model.pkl',
    'SVD (sklearn)': 'models/svd_model_sklearn.pkl',
    'User-KNN': 'models/user_knn_model.pkl',
    'Item-KNN': 'models/item_knn_model.pkl',
    'Content-Based': 'models/content_based_model.pkl'
}

# Print memory requirement analysis
print_memory_report(model_paths)

# Load models with memory optimization
print("\nAttempting memory-optimized sequential loading...\n")

loaded_models = load_models_sequential(model_paths, verbose=True)

# Test loaded models
print("\n" + "=" * 80)
print("TESTING LOADED MODELS")
print("=" * 80)

for name, model in loaded_models.items():
    print(f"\n{name}:")
    print(f"  Type: {type(model).__name__}")
    print(f"  Has get_recommendations: {hasattr(model, 'get_recommendations')}")
    print(f"  Has predict: {hasattr(model, 'predict')}")
    
    # Test recommendations if available
    if hasattr(model, 'get_recommendations'):
        try:
            recs = model.get_recommendations(1, n=5)
            print(f"  Test recommendations: {len(recs)} generated ✓")
        except Exception as e:
            print(f"  Test recommendations: FAILED - {str(e)[:50]}")

# Final summary
print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

total_models = len(model_paths)
loaded_count = len(loaded_models)
success_rate = (loaded_count / total_models * 100) if total_models > 0 else 0

print(f"\nModels loaded: {loaded_count}/{total_models} ({success_rate:.0f}%)")

if loaded_count == total_models:
    print("\n✓ SUCCESS! All 5 models loaded successfully!")
    print("  SVD (Surprise) memory issue RESOLVED with sequential loading.")
elif loaded_count >= 4:
    print(f"\n⚠ PARTIAL SUCCESS: {loaded_count}/{total_models} models working")
    
    missing = set(model_paths.keys()) - set(loaded_models.keys())
    print(f"  Missing: {', '.join(missing)}")
    print(f"  May require Docker Desktop memory increase to 8GB+")
else:
    print(f"\n✗ INSUFFICIENT: Only {loaded_count}/{total_models} models loaded")

mem_usage = get_memory_usage_mb()
if mem_usage > 0:
    print(f"\nFinal memory usage: {mem_usage:.1f} MB")

print("=" * 80)
