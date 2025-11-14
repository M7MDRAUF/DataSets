"""
CineMatch V2.1.1 - Model File Integrity Checker

Verifies all model files are not corrupted (not Git LFS pointers, valid pickle format).

Author: CineMatch Development Team
Date: November 14, 2025
"""

import pickle
from pathlib import Path
import sys

print('=' * 80)
print('CINEMATCH V2.1.1 - MODEL FILE INTEGRITY CHECK')
print('=' * 80)

models_dir = Path('models')
model_files = list(models_dir.glob('*.pkl'))

if not model_files:
    print('\n[ERROR] No model files found in models directory!')
    sys.exit(1)

print(f'\nFound {len(model_files)} model files to check...\n')

results = []

for model_file in sorted(model_files):
    print(f'Checking: {model_file.name}')
    
    # Check file size
    size_mb = model_file.stat().st_size / (1024**2)
    print(f'  Size: {size_mb:.1f} MB')
    
    # Check if it's a Git LFS pointer (corrupted)
    is_lfs_pointer = False
    try:
        with open(model_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if 'version https://git-lfs.github.com' in first_line:
                is_lfs_pointer = True
                print(f'  [CORRUPTED] Git LFS pointer file (not actual binary)')
                results.append((model_file.name, 'CORRUPTED_LFS', size_mb))
                continue
    except:
        pass  # Not a text file, good - it's binary
    
    # Try to load with pickle
    try:
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
        
        # Check type
        data_type = type(data).__name__
        
        # Verify it's a valid model
        if isinstance(data, dict):
            if 'model' in data:
                actual_model = data['model']
                has_methods = hasattr(actual_model, 'predict') or hasattr(actual_model, 'get_recommendations')
                print(f'  Type: Dict wrapper -> {type(actual_model).__name__}')
            else:
                has_methods = False
                print(f'  Type: Dict (no model key)')
        else:
            has_methods = hasattr(data, 'predict') or hasattr(data, 'get_recommendations')
            print(f'  Type: {data_type}')
        
        if has_methods:
            print(f'  Status: [OK] Valid model file')
            results.append((model_file.name, 'OK', size_mb))
        else:
            print(f'  Status: [WARNING] Loads but missing methods')
            results.append((model_file.name, 'WARNING', size_mb))
            
    except Exception as e:
        print(f'  Status: [ERROR] {str(e)[:60]}')
        results.append((model_file.name, 'ERROR', size_mb))

# Summary
print('\n' + '=' * 80)
print('INTEGRITY CHECK SUMMARY')
print('=' * 80)

print(f'\n{"Model File":<35} {"Status":<15} {"Size (MB)"}')
print('-' * 70)

for name, status, size in results:
    print(f'{name:<35} {status:<15} {size:.1f}')

# Count results
ok_count = sum(1 for _, s, _ in results if s == 'OK')
corrupted_count = sum(1 for _, s, _ in results if 'CORRUPTED' in s)
error_count = sum(1 for _, s, _ in results if s == 'ERROR')
warning_count = sum(1 for _, s, _ in results if s == 'WARNING')

print('\n' + '=' * 80)
print(f'Results: {ok_count} OK, {corrupted_count} CORRUPTED, {error_count} ERRORS, {warning_count} WARNINGS')
print('=' * 80)

if corrupted_count > 0:
    print('\n[CRITICAL] Found corrupted Git LFS pointer files!')
    print('These files were not properly downloaded from Git LFS.')
    print('Solution: Retrain models locally or run: git lfs pull')
elif error_count > 0:
    print(f'\n[ISSUE] Found {error_count} files with loading errors')
    print('Recommendation: Check error messages above and retrain affected models')
elif warning_count > 0:
    print(f'\n[MINOR] Found {warning_count} files with warnings')
    print('Models load but may have unexpected structure')
elif ok_count == len(results):
    print('\n[SUCCESS] All model files are valid and uncorrupted!')
    print('Models directory is clean and ready for production use.')
else:
    print('\n[UNKNOWN] Unexpected results')

print('=' * 80)
