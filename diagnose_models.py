"""
CineMatch V2.1.1 - Model Diagnostic Script

Diagnose and fix model loading issues:
1. Content-Based: Dict serialization vs BaseRecommender interface
2. SVD (Surprise): Memory allocation failure in Docker

Author: CineMatch Development Team
Date: November 14, 2025
"""

import sys
from pathlib import Path
import pickle
import time
import traceback

sys.path.append(str(Path(__file__).parent))

print("=" * 80)
print("CINEMATCH V2.1.1 - MODEL DIAGNOSTIC TOOL")
print("=" * 80)

# Test 1: Content-Based Model Structure
print("\n" + "=" * 80)
print("TEST 1: CONTENT-BASED MODEL STRUCTURE ANALYSIS")
print("=" * 80)

cb_model_path = Path("models/content_based_model.pkl")

if cb_model_path.exists():
    print(f"\nFile: {cb_model_path}")
    file_size_mb = cb_model_path.stat().st_size / (1024**2)
    print(f"Size: {file_size_mb:.1f} MB")
    
    try:
        print("\nLoading with pickle...")
        with open(cb_model_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        print(f"Type: {type(loaded_data)}")
        
        if isinstance(loaded_data, dict):
            print("\n[ISSUE CONFIRMED] Model saved as DICT, not class instance!")
            print("\nDict keys:", list(loaded_data.keys()))
            
            if 'model' in loaded_data:
                actual_model = loaded_data['model']
                print(f"\nActual model type: {type(actual_model)}")
                print(f"Has get_recommendations: {hasattr(actual_model, 'get_recommendations')}")
                print(f"Has predict: {hasattr(actual_model, 'predict')}")
                print(f"Is trained: {actual_model.is_trained if hasattr(actual_model, 'is_trained') else 'N/A'}")
                
                # Test methods
                if hasattr(actual_model, 'get_recommendations'):
                    print("\n[OK] Model instance has required methods")
                    print("[FIX NEEDED] Save model directly, not wrapped in dict")
                else:
                    print("\n[ERROR] Model instance missing required methods")
            
            if 'metrics' in loaded_data:
                print(f"\nMetrics:", loaded_data['metrics'])
            if 'metadata' in loaded_data:
                print(f"\nMetadata:", loaded_data['metadata'])
        
        else:
            print(f"\n[OK] Model is {type(loaded_data).__name__} instance")
            print(f"Has get_recommendations: {hasattr(loaded_data, 'get_recommendations')}")
            print(f"Has predict: {hasattr(loaded_data, 'predict')}")
    
    except Exception as e:
        print(f"\n[ERROR] Failed to load: {e}")
        traceback.print_exc()
else:
    print(f"\n[MISSING] File not found: {cb_model_path}")


# Test 2: SVD (Surprise) Model Memory Analysis
print("\n" + "=" * 80)
print("TEST 2: SVD (SURPRISE) MODEL MEMORY ANALYSIS")
print("=" * 80)

svd_model_path = Path("models/svd_model.pkl")

if svd_model_path.exists():
    print(f"\nFile: {svd_model_path}")
    file_size_mb = svd_model_path.stat().st_size / (1024**2)
    print(f"Size: {file_size_mb:.1f} MB")
    
    # Get current memory usage
    try:
        import psutil
        process = psutil.Process()
        mem_before_mb = process.memory_info().rss / (1024**2)
        print(f"\nMemory before load: {mem_before_mb:.1f} MB")
    except:
        mem_before_mb = None
        print("\npsutil not available, skipping memory tracking")
    
    try:
        print("\nLoading SVD (Surprise) model...")
        start_time = time.time()
        
        with open(svd_model_path, 'rb') as f:
            svd_model = pickle.load(f)
        
        load_time = time.time() - start_time
        print(f"Load time: {load_time:.2f}s")
        print(f"Type: {type(svd_model)}")
        
        if mem_before_mb:
            mem_after_mb = process.memory_info().rss / (1024**2)
            mem_increase_mb = mem_after_mb - mem_before_mb
            print(f"Memory after load: {mem_after_mb:.1f} MB")
            print(f"Memory increase: {mem_increase_mb:.1f} MB")
            print(f"Memory overhead: {(mem_increase_mb / file_size_mb):.2f}x file size")
        
        print("\n[OK] Model loaded successfully on host")
        print("[INFO] Docker failure may be due to container memory limits")
        
    except MemoryError as e:
        print(f"\n[ERROR] Memory allocation failed: {e}")
        print("[INFO] This is the same error occurring in Docker")
    except Exception as e:
        print(f"\n[ERROR] Failed to load: {e}")
        traceback.print_exc()
else:
    print(f"\n[MISSING] File not found: {svd_model_path}")


# Test 3: Compare All Model Serialization Formats
print("\n" + "=" * 80)
print("TEST 3: COMPARE ALL MODEL SERIALIZATION FORMATS")
print("=" * 80)

model_files = [
    "models/svd_model.pkl",
    "models/svd_model_sklearn.pkl",
    "models/user_knn_model.pkl",
    "models/item_knn_model.pkl",
    "models/content_based_model.pkl"
]

for model_path_str in model_files:
    model_path = Path(model_path_str)
    if not model_path.exists():
        continue
    
    model_name = model_path.stem
    file_size_mb = model_path.stat().st_size / (1024**2)
    
    print(f"\n{model_name}:")
    print(f"  Size: {file_size_mb:.1f} MB")
    
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        data_type = type(data).__name__
        print(f"  Type: {data_type}")
        
        if isinstance(data, dict):
            print(f"  [DICT] Keys: {list(data.keys())}")
            if 'model' in data:
                print(f"  [DICT] Inner model type: {type(data['model']).__name__}")
        else:
            print(f"  [INSTANCE] Has get_recommendations: {hasattr(data, 'get_recommendations')}")
            print(f"  [INSTANCE] Has predict: {hasattr(data, 'predict')}")
    
    except Exception as e:
        print(f"  [ERROR] {str(e)[:60]}")


# Summary and Recommendations
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
print("=" * 80)

print("""
ISSUE 1: Content-Based Model Serialization Mismatch
─────────────────────────────────────────────────────
Problem: train_content_based.py saves model wrapped in dict
Expected: Direct model instance (like other algorithms)

Current save pattern (train_content_based.py):
    model_data = {'model': model, 'metrics': {...}, 'metadata': {...}}
    pickle.dump(model_data, f)  # Saves DICT

Expected save pattern (train_knn_models.py, etc.):
    pickle.dump(model, f)  # Saves MODEL INSTANCE directly

Fix: Line 222 in train_content_based.py
    Change: pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    To:     pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

OR: Update test code to handle dict wrapper
    loaded_data = pickle.load(f)
    model = loaded_data['model'] if isinstance(loaded_data, dict) else loaded_data


ISSUE 2: SVD (Surprise) Memory Allocation in Docker
─────────────────────────────────────────────────────
Problem: Model loads on host but fails in Docker with [Errno 12]
Analysis: 
  • File size: 1115 MB
  • Docker RAM: 6.645 GiB available
  • Should have enough memory

Possible causes:
1. Memory fragmentation (container has used memory before)
2. Pickle unpickling creates temporary copies (2-3x file size)
3. Python process memory limits in container
4. Surprise library allocates large temporary structures

Solutions:
A) Increase Docker memory limit in docker-compose.yml:
   deploy:
     resources:
       limits:
         memory: 8G  # Increase from default

B) Load models sequentially with gc.collect() between loads

C) Consider alternative serialization (joblib with compression)

D) Reduce model size (fewer factors in training)


RECOMMENDED NEXT STEPS:
──────────────────────
1. Fix Content-Based model serialization (retrain or update loaders)
2. Test memory-optimized loading sequence in Docker
3. Consider increasing Docker memory limits
4. Validate all 5 algorithms work after fixes
""")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
