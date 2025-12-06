#!/usr/bin/env python3
"""
CineMatch Model Loading Benchmark

Benchmark model loading performance with different methods:
- Standard pickle
- Secure pickle (RestrictedUnpickler)
- Joblib (standard)
- Joblib (mmap_mode='r')

Usage:
    python scripts/benchmark_model_loading.py
    python scripts/benchmark_model_loading.py --models svd_model_sklearn.pkl
    python scripts/benchmark_model_loading.py --iterations 5

Author: CineMatch Development Team
Date: December 5, 2025
"""

import os
import sys
import time
import gc
import argparse
import pickle
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import psutil


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def get_file_size_mb(path: Path) -> float:
    """Get file size in MB."""
    return path.stat().st_size / (1024 * 1024)


def benchmark_method(
    model_path: Path,
    method: str,
    iterations: int = 3
) -> dict:
    """
    Benchmark a model loading method.
    
    Args:
        model_path: Path to model file
        method: 'pickle', 'secure', 'joblib', 'joblib_mmap'
        iterations: Number of iterations
        
    Returns:
        Dict with benchmark results
    """
    times = []
    memory_deltas = []
    
    for i in range(iterations):
        gc.collect()
        mem_before = get_memory_mb()
        
        start = time.time()
        
        try:
            if method == 'pickle':
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    
            elif method == 'secure':
                from src.utils.model_loader import load_model_safe
                model = load_model_safe(str(model_path))
                
            elif method == 'joblib':
                model = joblib.load(str(model_path))
                
            elif method == 'joblib_mmap':
                model = joblib.load(str(model_path), mmap_mode='r')
                
            else:
                raise ValueError(f"Unknown method: {method}")
                
            elapsed = time.time() - start
            mem_after = get_memory_mb()
            
            times.append(elapsed)
            memory_deltas.append(mem_after - mem_before)
            
            # Clear model to free memory
            del model
            gc.collect()
            
        except Exception as e:
            return {
                'method': method,
                'success': False,
                'error': str(e)
            }
    
    return {
        'method': method,
        'success': True,
        'iterations': iterations,
        'times': times,
        'min_time': min(times),
        'max_time': max(times),
        'avg_time': sum(times) / len(times),
        'memory_deltas_mb': memory_deltas,
        'avg_memory_delta_mb': sum(memory_deltas) / len(memory_deltas)
    }


def run_benchmarks(
    models_dir: Path,
    model_filter: list = None,
    iterations: int = 3
) -> dict:
    """
    Run benchmarks on all models.
    
    Args:
        models_dir: Directory containing model files
        model_filter: List of specific models to test (or None for all)
        iterations: Number of iterations per method
        
    Returns:
        Dict with all benchmark results
    """
    methods = ['pickle', 'secure', 'joblib', 'joblib_mmap']
    
    # Find model files
    pkl_files = list(models_dir.glob('*.pkl'))
    joblib_files = list(models_dir.glob('*.joblib'))
    all_files = pkl_files + joblib_files
    
    if model_filter:
        all_files = [f for f in all_files if f.name in model_filter]
    
    if not all_files:
        print("No model files found!")
        return {}
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'models_dir': str(models_dir),
        'iterations': iterations,
        'models': {}
    }
    
    for model_path in all_files:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_path.name}")
        print(f"Size: {get_file_size_mb(model_path):.1f}MB")
        print(f"{'='*60}")
        
        model_results = {
            'file_size_mb': get_file_size_mb(model_path),
            'methods': {}
        }
        
        for method in methods:
            # Skip methods that don't make sense for the file type
            if model_path.suffix == '.joblib' and method in ['pickle', 'secure']:
                continue
            
            print(f"\n  Testing: {method}")
            result = benchmark_method(model_path, method, iterations)
            
            if result['success']:
                print(f"    Times: {[f'{t:.2f}s' for t in result['times']]}")
                print(f"    Avg: {result['avg_time']:.2f}s")
                print(f"    Memory: +{result['avg_memory_delta_mb']:.0f}MB")
            else:
                print(f"    ERROR: {result.get('error', 'Unknown')}")
            
            model_results['methods'][method] = result
        
        results['models'][model_path.name] = model_results
    
    return results


def print_summary(results: dict):
    """Print a summary comparison table."""
    print(f"\n\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}\n")
    
    # Header
    print(f"{'Model':<30} {'Size(MB)':<10} {'Pickle':<12} {'Secure':<12} {'Joblib':<12} {'Joblib+mmap':<12}")
    print("-" * 88)
    
    for model_name, model_data in results.get('models', {}).items():
        size = model_data.get('file_size_mb', 0)
        methods = model_data.get('methods', {})
        
        def get_time(method):
            m = methods.get(method, {})
            if m.get('success'):
                return f"{m['avg_time']:.2f}s"
            return "N/A"
        
        print(f"{model_name:<30} {size:<10.1f} {get_time('pickle'):<12} {get_time('secure'):<12} {get_time('joblib'):<12} {get_time('joblib_mmap'):<12}")
    
    # Speedup analysis
    print(f"\n{'='*80}")
    print("SPEEDUP ANALYSIS (vs pickle)")
    print(f"{'='*80}\n")
    
    for model_name, model_data in results.get('models', {}).items():
        methods = model_data.get('methods', {})
        
        pickle_time = methods.get('pickle', {}).get('avg_time')
        mmap_time = methods.get('joblib_mmap', {}).get('avg_time')
        
        if pickle_time and mmap_time:
            speedup = pickle_time / mmap_time
            print(f"{model_name}: {speedup:.1f}x faster with joblib+mmap ({pickle_time:.2f}s -> {mmap_time:.2f}s)")


def main():
    parser = argparse.ArgumentParser(description='Benchmark model loading performance')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory containing model files')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Specific model files to test')
    parser.add_argument('--iterations', type=int, default=3,
                       help='Number of iterations per method')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("CineMatch Model Loading Benchmark")
    print(f"{'='*60}")
    print(f"Models directory: {models_dir}")
    print(f"Iterations: {args.iterations}")
    print(f"{'='*60}")
    
    results = run_benchmarks(
        models_dir,
        model_filter=args.models,
        iterations=args.iterations
    )
    
    print_summary(results)
    
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
