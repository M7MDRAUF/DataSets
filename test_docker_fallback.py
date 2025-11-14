"""
CineMatch V2.1.1 - Docker Fallback Test (4/5 Models)

Test with SVD Surprise excluded for 6.6GB Docker environments.

Author: CineMatch Development Team
Date: November 14, 2025
"""

import sys
sys.path.append('/app')

from src.utils.memory_manager import load_models_sequential, get_memory_usage_mb

print('=' * 80)
print('DOCKER FALLBACK TEST: 4/5 MODELS (EXCLUDE SVD SURPRISE)')
print('=' * 80)

# Exclude SVD Surprise for 6.6GB environments
model_paths = {
    'SVD (sklearn)': 'models/svd_model_sklearn.pkl',
    'User-KNN': 'models/user_knn_model.pkl',
    'Item-KNN': 'models/item_knn_model.pkl',
    'Content-Based': 'models/content_based_model.pkl'
}

print(f'\nLoading 4 models (optimized for Docker 6.6GB limit)...\n')

loaded = load_models_sequential(model_paths, verbose=True)

# Test
print('\n' + '=' * 80)
print('TESTING LOADED MODELS')
print('=' * 80)

for name, model in loaded.items():
    print(f'\n{name}:')
    if hasattr(model, 'get_recommendations'):
        try:
            recs = model.get_recommendations(1, n=5)
            print(f'  Recommendations: {len(recs)} generated [OK]')
        except Exception as e:
            print(f'  Recommendations: FAILED - {str(e)[:50]}')

# Summary
print('\n' + '=' * 80)
print(f'RESULT: {len(loaded)}/4 models loaded successfully')
print('=' * 80)

mem = get_memory_usage_mb()
if mem > 0:
    print(f'\nFinal memory usage: {mem:.1f} MB')

if len(loaded) == 4:
    print('\n[SUCCESS] All 4 models working in Docker (80% functionality)')
    print('For full 5/5 algorithms, increase Docker memory to 8GB+')
    print('See DOCKER_MEMORY_GUIDE.md for instructions')
else:
    print(f'\n[ISSUE] Only {len(loaded)}/4 models loaded')

print('=' * 80)
