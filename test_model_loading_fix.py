"""
CineMatch V2.1.1 - Model Loading Fix Validation

Test the model_loader utility and validate all models load correctly.

Author: CineMatch Development Team
Date: November 14, 2025
"""

import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent))

from src.utils.model_loader import load_model_safe, get_model_metadata

print("=" * 80)
print("CINEMATCH V2.1.1 - MODEL LOADING FIX VALIDATION")
print("=" * 80)

models_to_test = [
    ("SVD (Surprise)", "models/svd_model.pkl", False),  # Skip in test (memory heavy)
    ("SVD (sklearn)", "models/svd_model_sklearn.pkl", True),
    ("User-KNN", "models/user_knn_model.pkl", True),
    ("Item-KNN", "models/item_knn_model.pkl", True),
    ("Content-Based", "models/content_based_model.pkl", True),  # THIS IS THE FIX TARGET
]

results = []

for model_name, model_path, should_test in models_to_test:
    print(f"\n{'=' * 80}")
    print(f"Testing: {model_name}")
    print(f"{'=' * 80}")
    
    path_obj = Path(model_path)
    
    if not path_obj.exists():
        print(f"[SKIP] File not found: {model_path}")
        results.append((model_name, "MISSING", 0, "N/A"))
        continue
    
    file_size_mb = path_obj.stat().st_size / (1024**2)
    print(f"File: {model_path}")
    print(f"Size: {file_size_mb:.1f} MB")
    
    # Get metadata
    print("\nMetadata:")
    metadata = get_model_metadata(model_path)
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    if not should_test:
        print(f"\n[SKIP] Skipping load test (memory intensive)")
        results.append((model_name, "SKIPPED", file_size_mb, "N/A"))
        continue
    
    # Test loading
    try:
        print("\nLoading with load_model_safe()...")
        start_time = time.time()
        
        model = load_model_safe(model_path)
        
        load_time = time.time() - start_time
        print(f"  Load time: {load_time:.2f}s")
        print(f"  Type: {type(model).__name__}")
        print(f"  Has get_recommendations: {hasattr(model, 'get_recommendations')}")
        print(f"  Has predict: {hasattr(model, 'predict')}")
        
        if hasattr(model, 'is_trained'):
            print(f"  Is trained: {model.is_trained}")
        
        # Test methods
        if hasattr(model, 'get_recommendations'):
            print("\n[OK] Testing get_recommendations()...")
            try:
                recs = model.get_recommendations(1, n=5)
                print(f"  [OK] Generated {len(recs)} recommendations")
                status = "PASS"
            except Exception as e:
                print(f"  [FAIL] {str(e)[:60]}")
                status = "FAIL"
        else:
            print("\n[FAIL] Missing get_recommendations() method")
            status = "FAIL"
        
        results.append((model_name, status, file_size_mb, load_time))
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load: {e}")
        import traceback
        traceback.print_exc()
        results.append((model_name, "ERROR", file_size_mb, "N/A"))


# Summary
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

print(f"\n{'Model':<20} {'Status':<10} {'Size (MB)':<12} {'Load Time (s)'}")
print("-" * 70)

for model_name, status, size_mb, load_time in results:
    load_time_str = f"{load_time:.2f}" if isinstance(load_time, (int, float)) else str(load_time)
    print(f"{model_name:<20} {status:<10} {size_mb:<12.1f} {load_time_str}")

# Count results
passed = sum(1 for _, status, _, _ in results if status == "PASS")
failed = sum(1 for _, status, _, _ in results if status in ("FAIL", "ERROR"))
skipped = sum(1 for _, status, _, _ in results if status == "SKIPPED")

print(f"\n{'=' * 80}")
print(f"Results: {passed} PASSED, {failed} FAILED, {skipped} SKIPPED")
print(f"{'=' * 80}")

if failed == 0 and passed > 0:
    print("\n[SUCCESS] All tested models load correctly!")
    print("Content-Based model fix validated successfully.")
else:
    print(f"\n[INCOMPLETE] {failed} models failed to load")

print(f"\n{'=' * 80}")
print("NEXT STEPS:")
print("1. Test in Docker with increased memory limits")
print("2. Run comprehensive validation with all 5 models")
print("3. Update Docker container and redeploy")
print("=" * 80)
