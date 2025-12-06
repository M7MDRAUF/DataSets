"""
CineMatch V2.1.6 - Memory Profiling Utilities

Tools for identifying memory hotspots and optimizing memory usage.
Uses tracemalloc and custom tracking for detailed analysis.

Author: CineMatch Development Team
Date: December 5, 2025

Features:
    - Memory snapshot comparison
    - Per-function memory tracking
    - Object size analysis
    - Leak detection
    - Memory usage reporting
"""

import gc
import logging
import sys
import tracemalloc
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Memory snapshot at a point in time."""
    
    timestamp: float
    label: str
    current_mb: float
    peak_mb: float
    top_allocations: List[Tuple[str, float]] = field(default_factory=list)
    
    def __str__(self):
        return f"[{self.label}] Current: {self.current_mb:.1f}MB, Peak: {self.peak_mb:.1f}MB"


@dataclass
class MemoryDiff:
    """Difference between two memory snapshots."""
    
    label: str
    before_mb: float
    after_mb: float
    diff_mb: float
    duration_seconds: float
    new_allocations: List[Tuple[str, float]] = field(default_factory=list)
    
    @property
    def increased(self) -> bool:
        return self.diff_mb > 0
    
    def __str__(self):
        sign = "+" if self.increased else ""
        return f"[{self.label}] {sign}{self.diff_mb:.2f}MB ({self.duration_seconds:.2f}s)"


class MemoryProfiler:
    """
    Memory profiler for tracking and analyzing memory usage.
    
    Usage:
        profiler = MemoryProfiler()
        profiler.start()
        
        # Your code here
        snapshot = profiler.snapshot("after_loading")
        
        profiler.stop()
        profiler.report()
    """
    
    def __init__(self, top_n: int = 10):
        """
        Initialize profiler.
        
        Args:
            top_n: Number of top allocations to track
        """
        self.top_n = top_n
        self._snapshots: List[MemorySnapshot] = []
        self._is_running = False
        self._start_time: float = 0
    
    def start(self) -> None:
        """Start memory tracking."""
        if self._is_running:
            return
        
        gc.collect()
        tracemalloc.start()
        self._is_running = True
        self._start_time = time.time()
        self._snapshots = []
        
        logger.info("Memory profiler started")
    
    def stop(self) -> None:
        """Stop memory tracking."""
        if not self._is_running:
            return
        
        tracemalloc.stop()
        self._is_running = False
        
        logger.info("Memory profiler stopped")
    
    def snapshot(self, label: str = "snapshot") -> MemorySnapshot:
        """
        Take a memory snapshot.
        
        Args:
            label: Label for this snapshot
            
        Returns:
            MemorySnapshot with current memory state
        """
        if not self._is_running:
            logger.warning("Profiler not running, starting...")
            self.start()
        
        current, peak = tracemalloc.get_traced_memory()
        
        # Get top allocations
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')[:self.top_n]
        
        top_allocations = [
            (str(stat.traceback), stat.size / 1e6)
            for stat in top_stats
        ]
        
        mem_snapshot = MemorySnapshot(
            timestamp=time.time(),
            label=label,
            current_mb=current / 1e6,
            peak_mb=peak / 1e6,
            top_allocations=top_allocations
        )
        
        self._snapshots.append(mem_snapshot)
        return mem_snapshot
    
    def compare(self, label1: str, label2: str) -> Optional[MemoryDiff]:
        """
        Compare two snapshots.
        
        Args:
            label1: First snapshot label
            label2: Second snapshot label
            
        Returns:
            MemoryDiff or None if snapshots not found
        """
        snap1 = next((s for s in self._snapshots if s.label == label1), None)
        snap2 = next((s for s in self._snapshots if s.label == label2), None)
        
        if not snap1 or not snap2:
            return None
        
        return MemoryDiff(
            label=f"{label1} → {label2}",
            before_mb=snap1.current_mb,
            after_mb=snap2.current_mb,
            diff_mb=snap2.current_mb - snap1.current_mb,
            duration_seconds=snap2.timestamp - snap1.timestamp
        )
    
    def report(self) -> Dict[str, Any]:
        """
        Generate memory profiling report.
        
        Returns:
            Dict with profiling results
        """
        if not self._snapshots:
            return {'error': 'No snapshots taken'}
        
        report = {
            'duration_seconds': time.time() - self._start_time,
            'snapshots': len(self._snapshots),
            'peak_mb': max(s.peak_mb for s in self._snapshots),
            'final_mb': self._snapshots[-1].current_mb if self._snapshots else 0,
            'timeline': []
        }
        
        for snap in self._snapshots:
            report['timeline'].append({
                'label': snap.label,
                'current_mb': round(snap.current_mb, 2),
                'peak_mb': round(snap.peak_mb, 2)
            })
        
        return report
    
    def print_report(self) -> None:
        """Print formatted memory report."""
        report = self.report()
        
        print("\n" + "=" * 60)
        print("MEMORY PROFILING REPORT")
        print("=" * 60)
        
        print(f"\nDuration: {report.get('duration_seconds', 0):.2f}s")
        print(f"Peak Memory: {report.get('peak_mb', 0):.2f}MB")
        print(f"Final Memory: {report.get('final_mb', 0):.2f}MB")
        
        print("\nTimeline:")
        print("-" * 60)
        
        for entry in report.get('timeline', []):
            print(f"  [{entry['label']}] Current: {entry['current_mb']:.2f}MB, Peak: {entry['peak_mb']:.2f}MB")
        
        print("-" * 60)


@contextmanager
def memory_profile(label: str = "operation"):
    """
    Context manager for quick memory profiling.
    
    Usage:
        with memory_profile("loading_data"):
            df = pd.read_csv("large_file.csv")
    """
    gc.collect()
    tracemalloc.start()
    
    start_time = time.time()
    current_before, _ = tracemalloc.get_traced_memory()
    
    try:
        yield
    finally:
        current_after, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        duration = time.time() - start_time
        diff = (current_after - current_before) / 1e6
        
        sign = "+" if diff > 0 else ""
        logger.info(
            f"[{label}] Memory: {sign}{diff:.2f}MB, "
            f"Peak: {peak/1e6:.2f}MB, Time: {duration:.2f}s"
        )


def profile_memory(func: Callable) -> Callable:
    """
    Decorator to profile function memory usage.
    
    Usage:
        @profile_memory
        def expensive_function():
            # ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        gc.collect()
        tracemalloc.start()
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            duration = time.time() - start_time
            
            logger.info(
                f"[{func.__name__}] Memory: {current/1e6:.2f}MB, "
                f"Peak: {peak/1e6:.2f}MB, Time: {duration:.2f}s"
            )
    
    return wrapper


def get_object_size(obj: Any, seen: Optional[set] = None) -> int:
    """
    Recursively get size of object and all referenced objects.
    
    Args:
        obj: Object to measure
        seen: Set of seen object IDs (for recursion)
        
    Returns:
        Size in bytes
    """
    size = sys.getsizeof(obj)
    
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    
    seen.add(obj_id)
    
    if isinstance(obj, dict):
        size += sum([get_object_size(v, seen) for v in obj.values()])
        size += sum([get_object_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_object_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_object_size(i, seen) for i in obj])
    
    return size


def format_bytes(size: int) -> str:
    """Format byte size to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(size) < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def get_dataframe_memory(df: pd.DataFrame, deep: bool = True) -> Dict[str, Any]:
    """
    Get detailed memory usage of a DataFrame.
    
    Args:
        df: DataFrame to analyze
        deep: Deep memory inspection
        
    Returns:
        Dict with memory breakdown
    """
    mem_usage = df.memory_usage(deep=deep)
    total = mem_usage.sum()
    
    result = {
        'total_bytes': total,
        'total_formatted': format_bytes(total),
        'rows': len(df),
        'columns': len(df.columns),
        'bytes_per_row': total / max(len(df), 1),
        'column_breakdown': {}
    }
    
    for col in df.columns:
        col_bytes = mem_usage[col]
        result['column_breakdown'][col] = {
            'bytes': col_bytes,
            'formatted': format_bytes(col_bytes),
            'dtype': str(df[col].dtype),
            'percent': round(col_bytes / total * 100, 1)
        }
    
    return result


def find_memory_leaks(iterations: int = 3) -> List[Tuple[str, int]]:
    """
    Simple memory leak detection by running GC multiple times.
    
    Args:
        iterations: Number of GC iterations
        
    Returns:
        List of (type_name, count) for objects that persist
    """
    gc.collect()
    
    # Get baseline
    baseline = {}
    for obj in gc.get_objects():
        type_name = type(obj).__name__
        baseline[type_name] = baseline.get(type_name, 0) + 1
    
    # Run GC multiple times
    for _ in range(iterations):
        gc.collect()
    
    # Get final count
    final = {}
    for obj in gc.get_objects():
        type_name = type(obj).__name__
        final[type_name] = final.get(type_name, 0) + 1
    
    # Find increases
    increases = []
    for type_name, count in final.items():
        baseline_count = baseline.get(type_name, 0)
        if count > baseline_count:
            increases.append((type_name, count - baseline_count))
    
    # Sort by increase
    increases.sort(key=lambda x: x[1], reverse=True)
    
    return increases[:20]


# =============================================================================
# CINEMATCH-SPECIFIC PROFILING
# =============================================================================

def profile_algorithm_memory(algorithm_manager, algorithm_type, sample_users: int = 5):
    """
    Profile memory usage of a specific algorithm.
    
    Args:
        algorithm_manager: AlgorithmManager instance
        algorithm_type: Algorithm type to profile
        sample_users: Number of users to test
    """
    import random
    
    profiler = MemoryProfiler()
    profiler.start()
    
    profiler.snapshot("start")
    
    # Load algorithm
    algorithm = algorithm_manager.get_algorithm(algorithm_type)
    profiler.snapshot("algorithm_loaded")
    
    # Get sample users
    ratings_df = algorithm_manager._training_data[0]
    user_ids = random.sample(list(ratings_df['userId'].unique()), sample_users)
    
    # Generate recommendations
    for i, user_id in enumerate(user_ids):
        algorithm.get_recommendations(user_id, n=10)
        if i == 0:
            profiler.snapshot("first_recommendation")
    
    profiler.snapshot("all_recommendations")
    
    profiler.stop()
    profiler.print_report()
    
    return profiler.report()


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory Profiling Tools")
    parser.add_argument('--demo', action='store_true', help='Run demo')
    args = parser.parse_args()
    
    print("Memory Profiler Demo")
    print("=" * 40)
    
    # Demo basic profiling
    profiler = MemoryProfiler()
    profiler.start()
    
    profiler.snapshot("start")
    
    # Allocate some memory
    data = [i ** 2 for i in range(1000000)]
    profiler.snapshot("after_list")
    
    # More allocation
    import numpy as np
    arr = np.random.random((1000, 1000))
    profiler.snapshot("after_numpy")
    
    # Cleanup
    del data
    del arr
    gc.collect()
    profiler.snapshot("after_cleanup")
    
    profiler.stop()
    profiler.print_report()
    
    # Demo context manager
    print("\n\nContext Manager Demo:")
    with memory_profile("numpy_allocation"):
        arr = np.random.random((2000, 2000))
    
    # Demo DataFrame analysis
    print("\n\nDataFrame Memory Analysis:")
    df = pd.DataFrame({
        'int_col': np.random.randint(0, 1000, 100000),
        'float_col': np.random.random(100000),
        'str_col': [f'value_{i}' for i in range(100000)]
    })
    
    mem_info = get_dataframe_memory(df)
    print(f"Total: {mem_info['total_formatted']}")
    print(f"Bytes per row: {mem_info['bytes_per_row']:.2f}")
    print("\nColumn breakdown:")
    for col, info in mem_info['column_breakdown'].items():
        print(f"  {col}: {info['formatted']} ({info['percent']}%) - {info['dtype']}")
