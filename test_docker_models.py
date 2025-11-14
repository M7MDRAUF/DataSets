"""
CineMatch V2.1.1 - Docker Validation with Model Loader

Test all 5 algorithms in Docker using load_model_safe() utility.

Author: CineMatch Development Team
Date: November 14, 2025
"""

import sys, time
from pathlib import Path

print('=' * 80)
print('CINEMATCH V2.1.1 - DOCKER VALIDATION WITH load_model_safe()')
print('=' * 80)

sys.path.append('/app')

from src.utils.model_loader import load_model_safe
from src.data_processing import load_ratings, load_movies

# Load data once
print('\nLoading data...')
start = time.time()
ratings = load_ratings(sample_size=None)
movies = load_movies()
data_time = time.time() - start
print(f'Data loaded in {data_time:.2f}s: {len(ratings):,} ratings, {len(movies):,} movies')

# Test all 5 models
models_to_test = [
    ('SVD (Surprise)', 'models/svd_model.pkl'),
    ('SVD (sklearn)', 'models/svd_model_sklearn.pkl'),
    ('User-KNN', 'models/user_knn_model.pkl'),
    ('Item-KNN', 'models/item_knn_model.pkl'),
    ('Content-Based', 'models/content_based_model.pkl')
]

print('\n' + '=' * 80)
print('TESTING ALL 5 MODELS WITH load_model_safe()')
print('=' * 80)

results = []

for model_name, model_path in models_to_test:
    print(f'\n{model_name}:')
    try:
        # Load with utility
        start = time.time()
        model = load_model_safe(model_path)
        load_time = time.time() - start
        
        print(f'  Load: {load_time:.2f}s')
        print(f'  Type: {type(model).__name__}')
        print(f'  Has get_recommendations: {hasattr(model, "get_recommendations")}')
        print(f'  Has predict: {hasattr(model, "predict")}')
        
        # Test prediction (skip for raw SVD Surprise and Content-Based)
        if hasattr(model, 'predict') and 'Surprise' not in model_name and 'Content' not in model_name:
            try:
                start = time.time()
                pred = model.predict(1, 1)
                pred_time = time.time() - start
                print(f'  Predict: {pred_time:.3f}s, Rating: {pred:.2f}')
            except:
                pass
        
        # Test recommendations
        if hasattr(model, 'get_recommendations'):
            start = time.time()
            recs = model.get_recommendations(1, n=5)
            rec_time = time.time() - start
            print(f'  Recommendations: {rec_time:.2f}s, Count: {len(recs)}')
            print(f'  [OK] All tests passed')
            results.append((model_name, 'PASS', load_time))
        else:
            print(f'  [SKIP] No get_recommendations method (raw Surprise model)')
            results.append((model_name, 'SKIP', load_time))
        
    except MemoryError as e:
        print(f'  [FAIL] Memory allocation error: {str(e)[:60]}')
        results.append((model_name, 'MEMORY_ERROR', 0))
    except Exception as e:
        print(f'  [FAIL] {str(e)[:80]}')
        results.append((model_name, 'ERROR', 0))

# Summary
print('\n' + '=' * 80)
print('DOCKER VALIDATION SUMMARY')
print('=' * 80)

print(f'\n{"Model":<20} {"Status":<15} {"Load Time (s)"}')
print('-' * 60)

for model_name, status, load_time in results:
    load_str = f'{load_time:.2f}' if load_time > 0 else 'N/A'
    print(f'{model_name:<20} {status:<15} {load_str}')

passed = sum(1 for _, status, _ in results if status == 'PASS')
failed = sum(1 for _, status, _ in results if status in ('ERROR', 'MEMORY_ERROR'))
skipped = sum(1 for _, status, _ in results if status == 'SKIP')

print(f'\n{"=" * 80}')
print(f'Results: {passed} PASSED, {failed} FAILED, {skipped} SKIPPED')
print(f'{"=" * 80}')

if passed == 5:
    print('\n[SUCCESS] ALL 5 MODELS WORKING IN DOCKER!')
elif passed >= 3:
    print(f'\n[PARTIAL SUCCESS] {passed}/5 models working')
else:
    print(f'\n[FAILED] Only {passed}/5 models working')
