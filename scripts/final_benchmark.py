#!/usr/bin/env python3
"""Final performance benchmark for model loading optimizations."""

import sys
sys.path.insert(0, '/app' if '/app' in sys.path or __file__.startswith('/app') else '.')

import time
import joblib
from pathlib import Path

def run_benchmark():
    print('=' * 60)
    print('FINAL PERFORMANCE BENCHMARK - Model Loading')
    print('=' * 60)
    
    # Clear cache
    try:
        from src.memory.model_cache import get_model_cache
        cache = get_model_cache()
        cache.clear()
    except ImportError:
        cache = None
    
    models_dir = Path('/app/models') if Path('/app/models').exists() else Path('models')
    
    # Test 1: Direct loading
    print('\n1. DIRECT JOBLIB LOADING (mmap mode):')
    total_direct = 0
    total_size = 0
    
    for model_file in sorted(models_dir.glob('*.pkl')):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        total_size += size_mb
        
        start = time.perf_counter()
        model = joblib.load(model_file, mmap_mode='r')
        elapsed = time.perf_counter() - start
        total_direct += elapsed
        
        print(f'   {model_file.name}: {size_mb:.1f}MB in {elapsed:.2f}s')
    
    print(f'   TOTAL: {total_direct:.2f}s for {total_size:.1f}MB')
    
    # Test 2: Cache performance
    if cache is not None:
        print('\n2. CACHE PERFORMANCE:')
        cache.clear()
        
        # First load (miss)
        start = time.perf_counter()
        for model_file in models_dir.glob('*.pkl'):
            key = str(model_file)
            cache.get_or_load(key, lambda f=model_file: joblib.load(f, mmap_mode='r'))
        first_load = time.perf_counter() - start
        print(f'   First load (cache miss): {first_load:.2f}s')
        
        # Second load (hit)
        start = time.perf_counter()
        for model_file in models_dir.glob('*.pkl'):
            key = str(model_file)
            cache.get_or_load(key, lambda: None)
        second_load = time.perf_counter() - start
        print(f'   Second load (cache hit): {second_load*1000:.4f}ms')
        
        if second_load > 0:
            speedup = first_load / second_load
            print(f'   Speedup: {speedup:,.0f}x')
    
    # Summary
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    model_count = len(list(models_dir.glob('*.pkl')))
    print(f'Total models: {model_count}')
    print(f'Total size: {total_size:.1f}MB')
    print(f'Cold load time: {total_direct:.2f}s')
    print(f'Throughput: {total_size/total_direct:.1f}MB/s')
    
    # Compare to baseline
    baseline_time = 45.51  # Original 45.51 seconds for ONE model
    print(f'\nVs Original (45.51s for 1 model):')
    print(f'  New: {total_direct:.2f}s for {model_count} models')
    print(f'  Improvement: {baseline_time/total_direct:.1f}x faster per model')
    print('=' * 60)

if __name__ == '__main__':
    run_benchmark()
