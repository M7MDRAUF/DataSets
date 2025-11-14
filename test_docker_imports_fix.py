"""
CineMatch V2.1.1 - Docker Import Fix Validation

Tests that all Streamlit page imports work correctly in Docker after fixing
the src/utils/__init__.py circular import issue.

Author: CineMatch Development Team
Date: November 14, 2025
"""

print('=' * 80)
print('CINEMATCH V2.1.1 - DOCKER IMPORT FIX VALIDATION')
print('=' * 80)

import sys
from pathlib import Path

# Add src to path (same as Streamlit pages do)
sys.path.append(str(Path(__file__).parent))

results = []

# Test 1: Home page imports
print('\n[TEST 1] Home page imports (1_🏠_Home.py)')
try:
    from src.utils import (
        format_genres,
        create_rating_stars,
        get_genre_emoji
    )
    print('  ✅ format_genres imported')
    print('  ✅ create_rating_stars imported')
    print('  ✅ get_genre_emoji imported')
    
    # Functional test
    test = format_genres('Action|Comedy|Drama', max_genres=2)
    print(f'  ✅ format_genres works: "{test}"')
    results.append(('Home page', 'PASS'))
except Exception as e:
    print(f'  ❌ FAILED: {e}')
    results.append(('Home page', 'FAIL'))

# Test 2: Recommend page imports
print('\n[TEST 2] Recommend page imports (2_🎬_Recommend.py)')
try:
    from src.utils import (
        format_genres,
        create_rating_stars,
        get_genre_emoji
    )
    print('  ✅ format_genres imported')
    print('  ✅ create_rating_stars imported')
    print('  ✅ get_genre_emoji imported')
    
    # Functional test
    stars = create_rating_stars(4.5)
    print(f'  ✅ create_rating_stars works: {len(stars)} chars HTML')
    results.append(('Recommend page', 'PASS'))
except Exception as e:
    print(f'  ❌ FAILED: {e}')
    results.append(('Recommend page', 'FAIL'))

# Test 3: Analytics page imports
print('\n[TEST 3] Analytics page imports (3_📊_Analytics.py)')
try:
    from src.utils import extract_year_from_title, create_genre_color_map
    print('  ✅ extract_year_from_title imported')
    print('  ✅ create_genre_color_map imported')
    
    # Functional test
    year = extract_year_from_title('The Matrix (1999)')
    print(f'  ✅ extract_year_from_title works: {year}')
    
    colors = create_genre_color_map()
    print(f'  ✅ create_genre_color_map works: {len(colors)} genres')
    results.append(('Analytics page', 'PASS'))
except Exception as e:
    print(f'  ❌ FAILED: {e}')
    results.append(('Analytics page', 'FAIL'))

# Test 4: Model loading utilities (no regression)
print('\n[TEST 4] Model loading utilities (memory_manager, model_loader)')
try:
    from src.utils import (
        load_model_safe,
        load_models_sequential,
        aggressive_gc,
        get_memory_usage_mb
    )
    print('  ✅ load_model_safe imported')
    print('  ✅ load_models_sequential imported')
    print('  ✅ aggressive_gc imported')
    print('  ✅ get_memory_usage_mb imported')
    
    # Functional test
    aggressive_gc()
    mem = get_memory_usage_mb()
    print(f'  ✅ Functions work: {mem:.1f} MB memory usage')
    results.append(('Model loading', 'PASS'))
except Exception as e:
    print(f'  ❌ FAILED: {e}')
    results.append(('Model loading', 'FAIL'))

# Summary
print('\n' + '=' * 80)
print('VALIDATION SUMMARY')
print('=' * 80)

for test_name, status in results:
    symbol = '✅' if status == 'PASS' else '❌'
    print(f'{symbol} {test_name:<30} {status}')

pass_count = sum(1 for _, s in results if s == 'PASS')
total = len(results)

print('\n' + '=' * 80)
if pass_count == total:
    print(f'[SUCCESS] All {total}/{total} tests passed!')
    print('Docker container is ready for production use.')
else:
    print(f'[FAILED] Only {pass_count}/{total} tests passed')
    print('Review errors above and fix imports.')
print('=' * 80)
