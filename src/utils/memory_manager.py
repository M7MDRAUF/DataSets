"""
CineMatch V2.1.1 - Memory Management Utilities

Optimized model loading with memory management for Docker environments.
Handles large models (SVD Surprise) that require significant memory overhead.

Author: CineMatch Development Team
Date: November 14, 2025
"""

import gc
import sys
import time
from pathlib import Path
from typing import Any, Union, Optional, Dict
import pickle


def get_memory_usage_mb() -> float:
    """
    Get current process memory usage in MB.
    
    Returns:
        Memory usage in MB, or -1 if psutil not available
    """
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024**2)
    except ImportError:
        return -1


def aggressive_gc() -> None:
    """
    Perform aggressive garbage collection to free memory.
    Runs multiple GC cycles to ensure maximum cleanup.
    """
    for _ in range(3):
        gc.collect()


def load_model_with_gc(model_path: Union[str, Path], verbose: bool = True) -> Any:
    """
    Load a model with aggressive garbage collection before and after.
    
    This helps reduce memory fragmentation and peak memory usage,
    especially important for large models like SVD (Surprise).
    
    Args:
        model_path: Path to pickled model file
        verbose: Print memory usage info
        
    Returns:
        Loaded model instance
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Get initial memory
    mem_before = get_memory_usage_mb()
    
    if verbose and mem_before > 0:
        print(f"  Memory before: {mem_before:.1f} MB")
    
    # Aggressive GC before loading
    aggressive_gc()
    
    # Load model
    start_time = time.time()
    with open(model_path, 'rb') as f:
        loaded_data = pickle.load(f)
    
    load_time = time.time() - start_time
    
    # Handle dict wrapper format (Content-Based model)
    if isinstance(loaded_data, dict) and 'model' in loaded_data:
        model = loaded_data['model']
    else:
        model = loaded_data
    
    # Get memory after loading
    mem_after = get_memory_usage_mb()
    
    if verbose:
        file_size_mb = model_path.stat().st_size / (1024**2)
        print(f"  File size: {file_size_mb:.1f} MB")
        print(f"  Load time: {load_time:.2f}s")
        
        if mem_after > 0:
            mem_increase = mem_after - mem_before
            overhead = mem_increase / file_size_mb if file_size_mb > 0 else 0
            print(f"  Memory after: {mem_after:.1f} MB")
            print(f"  Memory increase: {mem_increase:.1f} MB ({overhead:.2f}x file size)")
    
    # GC after loading (cleanup any temporary objects)
    aggressive_gc()
    
    return model


def load_models_sequential(
    model_paths: Dict[str, Union[str, Path]],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Load multiple models sequentially with memory management.
    
    Loads models one at a time with garbage collection between loads
    to minimize peak memory usage and fragmentation.
    
    Optimized loading order:
    1. Largest/most memory-intensive models first (when memory is cleanest)
    2. Smaller models after
    
    Args:
        model_paths: Dict mapping model names to file paths
        verbose: Print loading progress
        
    Returns:
        Dict mapping model names to loaded model instances
    """
    loaded_models = {}
    
    if verbose:
        print("=" * 80)
        print("MEMORY-OPTIMIZED SEQUENTIAL MODEL LOADING")
        print("=" * 80)
    
    # Sort by file size (largest first) for optimal memory usage
    sorted_paths = []
    for name, path in model_paths.items():
        path_obj = Path(path)
        if path_obj.exists():
            size_mb = path_obj.stat().st_size / (1024**2)
            sorted_paths.append((name, path, size_mb))
        else:
            print(f"\n[WARNING] Model not found: {name} ({path})")
    
    sorted_paths.sort(key=lambda x: x[2], reverse=True)  # Largest first
    
    # Load each model
    for i, (name, path, size_mb) in enumerate(sorted_paths, 1):
        if verbose:
            print(f"\n[{i}/{len(sorted_paths)}] Loading {name} ({size_mb:.1f} MB)...")
        
        try:
            model = load_model_with_gc(path, verbose=verbose)
            loaded_models[name] = model
            
            if verbose:
                print(f"  ✓ {name} loaded successfully")
        
        except MemoryError as e:
            print(f"  ✗ {name} FAILED: Memory allocation error")
            print(f"    Error: {e}")
            print(f"    Continuing with remaining models...")
        
        except Exception as e:
            print(f"  ✗ {name} FAILED: {str(e)[:80]}")
            print(f"    Continuing with remaining models...")
    
    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Loaded {len(loaded_models)}/{len(sorted_paths)} models successfully")
        print(f"{'=' * 80}")
    
    return loaded_models


def check_available_memory() -> Optional[float]:
    """
    Check available system memory in MB.
    
    Returns:
        Available memory in MB, or None if psutil not available
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.available / (1024**2)
    except ImportError:
        return None


def estimate_model_memory_requirement(model_path: Union[str, Path]) -> Dict[str, float]:
    """
    Estimate memory requirement for loading a model.
    
    Uses heuristics based on file size and model type:
    - SVD (Surprise): 3.8x file size
    - Other models: 1.5x file size (conservative estimate)
    
    Args:
        model_path: Path to model file
        
    Returns:
        Dict with 'file_size_mb', 'estimated_memory_mb', 'overhead_factor'
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        return {'file_size_mb': 0, 'estimated_memory_mb': 0, 'overhead_factor': 0}
    
    file_size_mb = model_path.stat().st_size / (1024**2)
    
    # Detect model type from filename
    if 'svd_model.pkl' in str(model_path):  # SVD Surprise
        overhead_factor = 3.8
    else:
        overhead_factor = 1.5  # Conservative estimate for other models
    
    estimated_memory_mb = file_size_mb * overhead_factor
    
    return {
        'file_size_mb': file_size_mb,
        'estimated_memory_mb': estimated_memory_mb,
        'overhead_factor': overhead_factor
    }


def print_memory_report(model_paths: Dict[str, Union[str, Path]]) -> None:
    """
    Print a memory requirement report for all models.
    
    Args:
        model_paths: Dict mapping model names to file paths
    """
    print("=" * 80)
    print("MEMORY REQUIREMENT ANALYSIS")
    print("=" * 80)
    
    total_file_size = 0
    total_estimated_memory = 0
    
    print(f"\n{'Model':<20} {'File (MB)':<12} {'Est. Mem (MB)':<15} {'Overhead'}")
    print("-" * 70)
    
    for name, path in model_paths.items():
        estimates = estimate_model_memory_requirement(path)
        
        if estimates['file_size_mb'] > 0:
            total_file_size += estimates['file_size_mb']
            total_estimated_memory += estimates['estimated_memory_mb']
            
            print(f"{name:<20} {estimates['file_size_mb']:<12.1f} "
                  f"{estimates['estimated_memory_mb']:<15.1f} "
                  f"{estimates['overhead_factor']:.1f}x")
    
    print("-" * 70)
    print(f"{'TOTAL':<20} {total_file_size:<12.1f} {total_estimated_memory:<15.1f}")
    
    # Check available memory
    available_mb = check_available_memory()
    if available_mb:
        print(f"\nAvailable system memory: {available_mb:.1f} MB")
        
        if available_mb >= total_estimated_memory:
            print(f"✓ Sufficient memory available ({available_mb:.1f} MB > {total_estimated_memory:.1f} MB)")
        else:
            deficit = total_estimated_memory - available_mb
            print(f"⚠ Insufficient memory! Need {deficit:.1f} MB more")
            print(f"  Recommendation: Increase Docker memory or load models selectively")
    
    print("=" * 80)
