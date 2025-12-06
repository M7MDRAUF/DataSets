"""
Performance Benchmarks for CineMatch V2.1.6

Implements performance benchmarking utilities:
- Function timing
- Memory benchmarks
- Throughput measurement
- Comparison reports

Phase 2 - Task 2.9: Performance Benchmarking
"""

import logging
import time
import gc
import statistics
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar
from dataclasses import dataclass, field
from functools import wraps
from contextlib import contextmanager
import threading

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class TimingResult:
    """Result of a timing benchmark."""
    name: str
    iterations: int
    total_seconds: float
    mean_seconds: float
    median_seconds: float
    std_dev: float
    min_seconds: float
    max_seconds: float
    throughput: float  # operations per second
    
    def __str__(self) -> str:
        return (
            f"{self.name}: {self.mean_seconds*1000:.3f}ms ± {self.std_dev*1000:.3f}ms "
            f"(min: {self.min_seconds*1000:.3f}ms, max: {self.max_seconds*1000:.3f}ms, "
            f"n={self.iterations})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'iterations': self.iterations,
            'total_seconds': self.total_seconds,
            'mean_ms': self.mean_seconds * 1000,
            'median_ms': self.median_seconds * 1000,
            'std_dev_ms': self.std_dev * 1000,
            'min_ms': self.min_seconds * 1000,
            'max_ms': self.max_seconds * 1000,
            'throughput_ops': self.throughput
        }


@dataclass
class MemoryResult:
    """Result of a memory benchmark."""
    name: str
    peak_bytes: int
    delta_bytes: int
    gc_collections: int
    
    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024 * 1024)
    
    @property
    def delta_mb(self) -> float:
        return self.delta_bytes / (1024 * 1024)
    
    def __str__(self) -> str:
        return f"{self.name}: peak={self.peak_mb:.2f}MB, delta={self.delta_mb:.2f}MB"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks."""
    warmup_iterations: int = 3
    benchmark_iterations: int = 10
    gc_before: bool = True
    gc_after: bool = True
    timeout_seconds: float = 60.0


class Benchmark:
    """
    Performance benchmark utility.
    
    Features:
    - Timing benchmarks
    - Memory benchmarks  
    - Warmup support
    - Statistical analysis
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self._results: List[TimingResult] = []
    
    def time_function(
        self,
        func: Callable[..., T],
        *args: Any,
        name: Optional[str] = None,
        iterations: Optional[int] = None,
        **kwargs: Any
    ) -> TimingResult:
        """
        Time a function execution.
        
        Args:
            func: Function to benchmark
            args: Arguments to pass to function
            name: Name for the benchmark
            iterations: Number of iterations (default from config)
            kwargs: Keyword arguments for function
            
        Returns:
            TimingResult with statistics
        """
        name = name or func.__name__
        iterations = iterations or self.config.benchmark_iterations
        
        # Warmup
        for _ in range(self.config.warmup_iterations):
            func(*args, **kwargs)
        
        # GC before
        if self.config.gc_before:
            gc.collect()
        
        # Benchmark
        times: List[float] = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        # GC after
        if self.config.gc_after:
            gc.collect()
        
        # Calculate statistics
        result = TimingResult(
            name=name,
            iterations=iterations,
            total_seconds=sum(times),
            mean_seconds=statistics.mean(times),
            median_seconds=statistics.median(times),
            std_dev=statistics.stdev(times) if len(times) > 1 else 0,
            min_seconds=min(times),
            max_seconds=max(times),
            throughput=iterations / sum(times) if sum(times) > 0 else 0
        )
        
        self._results.append(result)
        logger.info(str(result))
        
        return result
    
    @contextmanager
    def time_block(self, name: str = "block"):
        """Context manager for timing a code block."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            logger.info(f"[{name}] {elapsed*1000:.3f}ms")
    
    def compare(
        self,
        funcs: Dict[str, Callable[..., Any]],
        *args: Any,
        iterations: Optional[int] = None,
        **kwargs: Any
    ) -> Dict[str, TimingResult]:
        """
        Compare multiple functions.
        
        Args:
            funcs: Dictionary of name -> function
            args: Arguments for all functions
            iterations: Number of iterations
            kwargs: Keyword arguments for all functions
            
        Returns:
            Dictionary of name -> TimingResult
        """
        results = {}
        
        for name, func in funcs.items():
            results[name] = self.time_function(
                func, *args, name=name, iterations=iterations, **kwargs
            )
        
        # Find fastest
        if results:
            fastest = min(results.values(), key=lambda r: r.mean_seconds)
            
            print("\n" + "=" * 50)
            print("Benchmark Comparison")
            print("=" * 50)
            
            for name, result in sorted(results.items(), key=lambda x: x[1].mean_seconds):
                speedup = fastest.mean_seconds / result.mean_seconds if result.mean_seconds > 0 else 0
                marker = " (fastest)" if result == fastest else f" ({speedup:.2f}x slower)"
                print(f"{name}: {result.mean_seconds*1000:.3f}ms{marker}")
        
        return results
    
    def get_results(self) -> List[TimingResult]:
        """Get all benchmark results."""
        return list(self._results)
    
    def clear_results(self) -> None:
        """Clear stored results."""
        self._results.clear()


def timed(
    name: Optional[str] = None,
    log_result: bool = True
):
    """
    Decorator to time function execution.
    
    Usage:
        @timed()
        def my_function():
            pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        func_name = name or func.__name__
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                if log_result:
                    logger.info(f"[{func_name}] {elapsed*1000:.3f}ms")
        
        return wrapper
    return decorator


def profile_throughput(
    func: Callable[..., T],
    *args: Any,
    duration_seconds: float = 5.0,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Profile throughput of a function.
    
    Runs function repeatedly for a duration and calculates ops/sec.
    """
    count = 0
    start = time.perf_counter()
    end_time = start + duration_seconds
    
    while time.perf_counter() < end_time:
        func(*args, **kwargs)
        count += 1
    
    elapsed = time.perf_counter() - start
    
    return {
        'function': func.__name__,
        'duration_seconds': elapsed,
        'total_calls': count,
        'throughput_ops': count / elapsed,
        'avg_latency_ms': (elapsed / count) * 1000
    }


class RecommendationBenchmark:
    """
    Specialized benchmarks for recommendation system.
    
    Features:
    - Model loading time
    - Prediction latency
    - Batch prediction throughput
    """
    
    def __init__(self):
        self.benchmark = Benchmark()
        self._results: Dict[str, Any] = {}
    
    def benchmark_model_load(
        self,
        load_func: Callable[[], Any],
        name: str = "model_load"
    ) -> TimingResult:
        """Benchmark model loading time."""
        return self.benchmark.time_function(load_func, name=name, iterations=3)
    
    def benchmark_single_prediction(
        self,
        predict_func: Callable[[int, int], float],
        user_ids: List[int],
        item_ids: List[int],
        name: str = "single_prediction"
    ) -> TimingResult:
        """Benchmark single prediction latency."""
        import random
        
        def predict():
            u = random.choice(user_ids)
            i = random.choice(item_ids)
            predict_func(u, i)
        
        return self.benchmark.time_function(predict, name=name)
    
    def benchmark_batch_prediction(
        self,
        batch_predict_func: Callable[[int, int], List[tuple]],
        user_id: int,
        n_recommendations: int,
        name: str = "batch_prediction"
    ) -> TimingResult:
        """Benchmark batch prediction."""
        return self.benchmark.time_function(
            batch_predict_func, user_id, n_recommendations, name=name
        )
    
    def benchmark_throughput(
        self,
        predict_func: Callable[[int, int], float],
        user_ids: List[int],
        item_ids: List[int],
        duration_seconds: float = 5.0
    ) -> Dict[str, Any]:
        """Benchmark prediction throughput."""
        import random
        
        def predict():
            u = random.choice(user_ids)
            i = random.choice(item_ids)
            predict_func(u, i)
        
        return profile_throughput(predict, duration_seconds=duration_seconds)
    
    def run_full_benchmark(
        self,
        model_loader: Callable[[], Any],
        predictor: Any,
        user_ids: List[int],
        item_ids: List[int]
    ) -> Dict[str, Any]:
        """Run a full benchmark suite."""
        results = {}
        
        print("\n" + "=" * 60)
        print("CineMatch Recommendation Benchmark Suite")
        print("=" * 60)
        
        # Model loading
        print("\n[1] Model Loading")
        results['model_load'] = self.benchmark_model_load(model_loader).to_dict()
        
        # Single prediction
        if hasattr(predictor, 'predict'):
            print("\n[2] Single Prediction Latency")
            results['single_prediction'] = self.benchmark_single_prediction(
                predictor.predict, user_ids, item_ids
            ).to_dict()
        
        # Batch prediction
        if hasattr(predictor, 'recommend'):
            print("\n[3] Batch Prediction (Top-10)")
            import random
            results['batch_prediction'] = self.benchmark_batch_prediction(
                predictor.recommend, random.choice(user_ids), 10
            ).to_dict()
        
        # Throughput
        if hasattr(predictor, 'predict'):
            print("\n[4] Throughput Test (5 seconds)")
            results['throughput'] = self.benchmark_throughput(
                predictor.predict, user_ids, item_ids, duration_seconds=5.0
            )
            print(f"  Throughput: {results['throughput']['throughput_ops']:.0f} ops/sec")
        
        self._results = results
        return results
    
    def get_report(self) -> str:
        """Generate benchmark report."""
        lines = [
            "=" * 60,
            "BENCHMARK REPORT",
            "=" * 60,
            ""
        ]
        
        for name, data in self._results.items():
            lines.append(f"[{name}]")
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, float):
                        lines.append(f"  {k}: {v:.3f}")
                    else:
                        lines.append(f"  {k}: {v}")
            lines.append("")
        
        return "\n".join(lines)


# Convenience functions
_default_benchmark: Optional[Benchmark] = None


def get_benchmark() -> Benchmark:
    global _default_benchmark
    if _default_benchmark is None:
        _default_benchmark = Benchmark()
    return _default_benchmark


def time_function(func: Callable[..., T], *args: Any, **kwargs: Any) -> TimingResult:
    """Time a function."""
    return get_benchmark().time_function(func, *args, **kwargs)


def compare_functions(
    funcs: Dict[str, Callable[..., Any]],
    *args: Any,
    **kwargs: Any
) -> Dict[str, TimingResult]:
    """Compare functions."""
    return get_benchmark().compare(funcs, *args, **kwargs)


if __name__ == "__main__":
    import math
    import random
    
    print("Performance Benchmark Demo")
    print("=" * 50)
    
    # Demo basic timing
    print("\n1. Basic Timing")
    print("-" * 30)
    
    benchmark = Benchmark(BenchmarkConfig(
        warmup_iterations=2,
        benchmark_iterations=5
    ))
    
    def slow_function(n: int) -> int:
        total = 0
        for i in range(n):
            total += math.sqrt(i)
        return int(total)
    
    result = benchmark.time_function(slow_function, 100000, name="slow_function")
    print(result)
    
    # Demo comparison
    print("\n\n2. Function Comparison")
    print("-" * 30)
    
    def sort_builtin(data: List[int]) -> List[int]:
        return sorted(data)
    
    def sort_manual(data: List[int]) -> List[int]:
        data = data.copy()
        # Simple bubble sort for comparison
        n = len(data)
        for i in range(min(n, 100)):  # Limited for demo
            for j in range(n - i - 1):
                if data[j] > data[j + 1]:
                    data[j], data[j + 1] = data[j + 1], data[j]
        return data
    
    test_data = [random.randint(1, 1000) for _ in range(500)]
    
    benchmark.compare(
        {
            'builtin_sort': sort_builtin,
            'manual_sort': sort_manual
        },
        test_data,
        iterations=5
    )
    
    # Demo decorator
    print("\n\n3. Timed Decorator")
    print("-" * 30)
    
    @timed(name="decorated_function")
    def decorated_function():
        time.sleep(0.01)
        return sum(range(1000))
    
    decorated_function()
    decorated_function()
    
    # Demo throughput
    print("\n\n4. Throughput Profiling")
    print("-" * 30)
    
    def fast_function():
        return sum(range(100))
    
    throughput = profile_throughput(fast_function, duration_seconds=2.0)
    print(f"Function: {throughput['function']}")
    print(f"Duration: {throughput['duration_seconds']:.2f}s")
    print(f"Total calls: {throughput['total_calls']}")
    print(f"Throughput: {throughput['throughput_ops']:.0f} ops/sec")
    print(f"Avg latency: {throughput['avg_latency_ms']:.3f}ms")
    
    # Demo context manager
    print("\n\n5. Context Manager Timing")
    print("-" * 30)
    
    with benchmark.time_block("sleep_block"):
        time.sleep(0.05)
    
    with benchmark.time_block("compute_block"):
        _ = [x**2 for x in range(10000)]
