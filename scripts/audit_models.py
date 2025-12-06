"""Quick audit script for model files."""
from src.utils.secure_serialization import audit_pickle_file
from pathlib import Path

models_dir = Path('models')
for pkl_file in models_dir.glob('*.pkl'):
    print(f'\nAuditing: {pkl_file.name}')
    result = audit_pickle_file(pkl_file)
    size_mb = result['size_bytes'] / (1024*1024)
    print(f'  Size: {size_mb:.1f} MB')
    print(f'  Safe: {result["safe"]}')
    if result.get('unknown_found'):
        print(f'  Unknown modules: {result["unknown_found"]}')
    if result.get('blocked_found'):
        print(f'  BLOCKED: {result["blocked_found"]}')
    if result.get('error'):
        print(f'  Error: {result["error"]}')
