"""
Memory Profiler for CineMatch V2.1.6

Implements memory profiling and monitoring:
- Memory usage tracking
- Object size analysis
- Memory leak detection
- Profiling decorators

Phase 2 - Task 2.5: Memory Profiling
"""

import logging
import sys
import gc
import time
import threading
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, TypeVar, Tuple
from dataclasses import dataclass, field
from functools import wraps
from contextlib import contextmanager
import weakref

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage."""
    timestamp: float
    total_bytes: int
    available_bytes: int
    percent_used: float
    process_bytes: int
    gc_counts: Tuple[int, int, int]
    
    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)
    
    @property
    def process_mb(self) -> float:
        return self.process_bytes / (1024 * 1024)


@dataclass
class AllocationInfo:
    """Information about a memory allocation."""
    size: int
    file: str
    line: int
    traceback: List[str]


@dataclass
class MemoryProfile:
    """Profile of a function's memory usage."""
    function_name: str
    peak_memory_bytes: int
    allocated_bytes: int
    freed_bytes: int
    duration_seconds: float
    
    @property
    def peak_mb(self) -> float:
        return self.peak_memory_bytes / (1024 * 1024)
    
    @property
    def allocated_mb(self) -> float:
        return self.allocated_bytes / (1024 * 1024)


class MemoryTracker:
    """
    Track memory usage over time.
    
    Features:
    - Continuous monitoring
    - Snapshot history
    - Trend analysis
    """
    
    def __init__(self, max_snapshots: int = 1000):
        self.max_snapshots = max_snapshots
        self._snapshots: List[MemorySnapshot] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot."""
        import psutil  # type: ignore
        
        process = psutil.Process()
        memory_info = process.memory_info()
        virtual_memory = psutil.virtual_memory()
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            total_bytes=virtual_memory.total,
            available_bytes=virtual_memory.available,
            percent_used=virtual_memory.percent,
            process_bytes=memory_info.rss,
            gc_counts=gc.get_count()
        )
        
        with self._lock:
            self._snapshots.append(snapshot)
            if len(self._snapshots) > self.max_snapshots:
                self._snapshots.pop(0)
        
        return snapshot
    
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start background monitoring."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self, interval: float) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                self.take_snapshot()
            except Exception as e:
                logger.error(f"Memory snapshot error: {e}")
            time.sleep(interval)
    
    def get_snapshots(self) -> List[MemorySnapshot]:
        """Get all snapshots."""
        with self._lock:
            return list(self._snapshots)
    
    def get_trend(self) -> Dict[str, Any]:
        """Analyze memory trend."""
        snapshots = self.get_snapshots()
        
        if len(snapshots) < 2:
            return {'trend': 'insufficient_data'}
        
        # Calculate trend
        first = snapshots[0]
        last = snapshots[-1]
        
        memory_change = last.process_bytes - first.process_bytes
        time_delta = last.timestamp - first.timestamp
        rate_per_second = memory_change / max(1, time_delta)
        
        return {
            'start_mb': first.process_mb,
            'end_mb': last.process_mb,
            'change_mb': memory_change / (1024 * 1024),
            'rate_mb_per_hour': rate_per_second * 3600 / (1024 * 1024),
            'duration_seconds': time_delta,
            'snapshots_count': len(snapshots),
            'trend': 'increasing' if memory_change > 0 else 'stable_or_decreasing'
        }


class MemoryProfiler:
    """
    Profile memory usage with tracemalloc.
    
    Features:
    - Allocation tracking
    - Top allocations
    - Memory diffs
    """
    
    def __init__(self, nframes: int = 25):
        self.nframes = nframes
        self._baseline: Optional[Any] = None
    
    def start(self) -> None:
        """Start memory profiling."""
        if not tracemalloc.is_tracing():
            tracemalloc.start(self.nframes)
            logger.info("Memory profiling started")
    
    def stop(self) -> None:
        """Stop memory profiling."""
        if tracemalloc.is_tracing():
            tracemalloc.stop()
            logger.info("Memory profiling stopped")
    
    def take_baseline(self) -> None:
        """Take baseline snapshot."""
        if tracemalloc.is_tracing():
            self._baseline = tracemalloc.take_snapshot()
    
    def get_top_allocations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top memory allocations."""
        if not tracemalloc.is_tracing():
            return []
        
        snapshot = tracemalloc.take_snapshot()
        stats = snapshot.statistics('lineno')
        
        result = []
        for stat in stats[:limit]:
            result.append({
                'file': stat.traceback.format()[0] if stat.traceback else 'unknown',
                'size_bytes': stat.size,
                'size_mb': stat.size / (1024 * 1024),
                'count': stat.count
            })
        
        return result
    
    def get_memory_diff(self) -> List[Dict[str, Any]]:
        """Get memory diff from baseline."""
        if not tracemalloc.is_tracing() or not self._baseline:
            return []
        
        current = tracemalloc.take_snapshot()
        stats = current.compare_to(self._baseline, 'lineno')
        
        result = []
        for stat in stats[:20]:
            if stat.size_diff != 0:
                result.append({
                    'file': stat.traceback.format()[0] if stat.traceback else 'unknown',
                    'size_diff_bytes': stat.size_diff,
                    'size_diff_mb': stat.size_diff / (1024 * 1024),
                    'count_diff': stat.count_diff
                })
        
        return result
    
    @contextmanager
    def profile_block(self, name: str = "block"):
        """Context manager for profiling a block."""
        if not tracemalloc.is_tracing():
            tracemalloc.start(self.nframes)
        
        gc.collect()
        start_snapshot = tracemalloc.take_snapshot()
        start_time = time.time()
        
        try:
            yield
        finally:
            gc.collect()
            end_snapshot = tracemalloc.take_snapshot()
            duration = time.time() - start_time
            
            stats = end_snapshot.compare_to(start_snapshot, 'lineno')
            
            total_diff = sum(s.size_diff for s in stats)
            logger.info(
                f"Memory profile [{name}]: {total_diff / 1024:.1f}KB "
                f"in {duration:.2f}s"
            )


def profile_memory(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to profile memory usage of a function.
    
    Usage:
        @profile_memory
        def my_function():
            # do something
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        gc.collect()
        
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        start_snapshot = tracemalloc.take_snapshot()
        
        try:
            result = func(*args, **kwargs)
        finally:
            gc.collect()
            end_snapshot = tracemalloc.take_snapshot()
            
            stats = end_snapshot.compare_to(start_snapshot, 'lineno')
            total_allocated = sum(s.size for s in stats if s.size_diff > 0)
            total_freed = abs(sum(s.size for s in stats if s.size_diff < 0))
            
            logger.info(
                f"Memory [{func.__name__}]: "
                f"+{total_allocated / 1024:.1f}KB, "
                f"-{total_freed / 1024:.1f}KB"
            )
        
        return result
    
    return wrapper


def get_object_size(obj: Any) -> int:
    """
    Get the deep size of an object in bytes.
    
    Uses recursive traversal to calculate total size.
    """
    seen = set()
    
    def _sizeof(o: Any) -> int:
        obj_id = id(o)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        
        size = sys.getsizeof(o)
        
        if isinstance(o, dict):
            size += sum(_sizeof(k) + _sizeof(v) for k, v in o.items())
        elif isinstance(o, (list, tuple, set, frozenset)):
            size += sum(_sizeof(item) for item in o)
        elif hasattr(o, '__dict__'):
            size += _sizeof(o.__dict__)
        elif hasattr(o, '__slots__'):
            size += sum(
                _sizeof(getattr(o, s))
                for s in o.__slots__
                if hasattr(o, s)
            )
        
        return size
    
    return _sizeof(obj)


def get_object_size_mb(obj: Any) -> float:
    """Get object size in MB."""
    return get_object_size(obj) / (1024 * 1024)


def force_gc() -> Dict[str, int]:
    """Force garbage collection and return stats."""
    counts_before = gc.get_count()
    collected = gc.collect()
    counts_after = gc.get_count()
    
    return {
        'collected': collected,
        'gen0_before': counts_before[0],
        'gen1_before': counts_before[1],
        'gen2_before': counts_before[2],
        'gen0_after': counts_after[0],
        'gen1_after': counts_after[1],
        'gen2_after': counts_after[2]
    }


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage."""
    try:
        import psutil  # type: ignore
        
        process = psutil.Process()
        memory_info = process.memory_info()
        virtual_memory = psutil.virtual_memory()
        
        return {
            'process_mb': memory_info.rss / (1024 * 1024),
            'process_vms_mb': memory_info.vms / (1024 * 1024),
            'system_total_mb': virtual_memory.total / (1024 * 1024),
            'system_available_mb': virtual_memory.available / (1024 * 1024),
            'system_percent': virtual_memory.percent
        }
    except ImportError:
        return {
            'error': 'psutil not installed'
        }


# Global tracker instance
_memory_tracker: Optional[MemoryTracker] = None


def get_memory_tracker() -> MemoryTracker:
    global _memory_tracker
    if _memory_tracker is None:
        _memory_tracker = MemoryTracker()
    return _memory_tracker


def start_memory_monitoring(interval: float = 1.0) -> None:
    """Start global memory monitoring."""
    get_memory_tracker().start_monitoring(interval)


def stop_memory_monitoring() -> None:
    """Stop global memory monitoring."""
    get_memory_tracker().stop_monitoring()


def get_memory_trend() -> Dict[str, Any]:
    """Get global memory trend."""
    return get_memory_tracker().get_trend()


if __name__ == "__main__":
    print("Memory Profiler Demo")
    print("=" * 50)
    
    # Demo object size
    print("\n1. Object Size Analysis")
    print("-" * 30)
    
    small_list = [1, 2, 3]
    large_list = list(range(10000))
    nested_dict = {
        'data': list(range(1000)),
        'nested': {'a': list(range(100)), 'b': list(range(100))}
    }
    
    print(f"Small list size: {get_object_size(small_list)} bytes")
    print(f"Large list size: {get_object_size(large_list) / 1024:.1f} KB")
    print(f"Nested dict size: {get_object_size(nested_dict) / 1024:.1f} KB")
    
    # Demo memory usage
    print("\n\n2. Current Memory Usage")
    print("-" * 30)
    
    usage = get_memory_usage()
    for key, value in usage.items():
        print(f"  {key}: {value}")
    
    # Demo GC
    print("\n\n3. Garbage Collection")
    print("-" * 30)
    
    # Create some garbage
    for _ in range(1000):
        _ = [0] * 1000
    
    gc_stats = force_gc()
    print(f"GC Stats: {gc_stats}")
    
    # Demo profiler
    print("\n\n4. Memory Profiler")
    print("-" * 30)
    
    profiler = MemoryProfiler()
    profiler.start()
    
    # Allocate memory
    data = [list(range(1000)) for _ in range(100)]
    
    top_allocs = profiler.get_top_allocations(5)
    print("Top allocations:")
    for alloc in top_allocs:
        print(f"  {alloc['size_mb']:.2f}MB - {alloc['file'][:50]}")
    
    profiler.stop()
    
    # Demo decorated function
    print("\n\n5. Memory Profiled Function")
    print("-" * 30)
    
    @profile_memory
    def allocate_memory():
        data = []
        for _ in range(100):
            data.append(list(range(1000)))
        return len(data)
    
    result = allocate_memory()
    print(f"Function returned: {result}")
