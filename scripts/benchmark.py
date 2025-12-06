"""
CineMatch V2.1.6 - Performance Benchmarking Suite

Comprehensive benchmarking for recommendation algorithms and system components.
Use to track performance regressions and optimize critical paths.

Author: CineMatch Development Team
Date: December 5, 2025

Usage:
    python -m scripts.benchmark --suite all
    python -m scripts.benchmark --suite recommendations
    python -m scripts.benchmark --output results.json
"""

import argparse
import gc
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import tracemalloc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    
    name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_dev: float
    memory_peak_mb: float
    throughput: float  # operations per second
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    
    name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'timestamp': self.timestamp,
            'system_info': self.system_info,
            'results': [r.to_dict() for r in self.results]
        }


class Benchmark:
    """
    Benchmark runner with timing and memory tracking.
    
    Usage:
        benchmark = Benchmark()
        
        @benchmark.measure("my_function")
        def my_function():
            # expensive operation
            pass
        
        result = benchmark.run("my_function", iterations=100)
    """
    
    def __init__(self, warmup: int = 3, gc_collect: bool = True):
        """
        Initialize benchmark.
        
        Args:
            warmup: Number of warmup iterations before timing
            gc_collect: Run garbage collection before each benchmark
        """
        self.warmup = warmup
        self.gc_collect = gc_collect
        self._functions: Dict[str, Callable] = {}
    
    def register(self, name: str, func: Callable) -> None:
        """Register a function for benchmarking."""
        self._functions[name] = func
    
    def measure(self, name: str) -> Callable:
        """Decorator to register a function for benchmarking."""
        def decorator(func: Callable) -> Callable:
            self.register(name, func)
            return func
        return decorator
    
    def run(
        self,
        name: str,
        iterations: int = 10,
        track_memory: bool = True
    ) -> BenchmarkResult:
        """
        Run benchmark for a registered function.
        
        Args:
            name: Name of registered function
            iterations: Number of iterations to run
            track_memory: Track peak memory usage
            
        Returns:
            BenchmarkResult with timing and memory info
        """
        if name not in self._functions:
            raise ValueError(f"Function '{name}' not registered")
        
        func = self._functions[name]
        
        # Garbage collection
        if self.gc_collect:
            gc.collect()
        
        # Warmup
        for _ in range(self.warmup):
            func()
        
        # Memory tracking
        if track_memory:
            tracemalloc.start()
        
        # Timed runs
        times = []
        for _ in range(iterations):
            if self.gc_collect:
                gc.collect()
            
            start = time.perf_counter()
            func()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        # Memory peak
        memory_peak = 0
        if track_memory:
            _, memory_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_peak /= 1e6  # Convert to MB
        
        # Calculate statistics
        total_time = sum(times)
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        
        return BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min(times),
            max_time=max(times),
            std_dev=std_dev,
            memory_peak_mb=memory_peak,
            throughput=iterations / total_time if total_time > 0 else 0
        )


# =============================================================================
# CINEMATCH BENCHMARKS
# =============================================================================

def get_system_info() -> Dict[str, Any]:
    """Get system information for benchmark context."""
    import platform
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'processor': platform.processor(),
    }
    
    try:
        import psutil
        info['cpu_count'] = psutil.cpu_count()
        info['memory_total_gb'] = psutil.virtual_memory().total / 1e9
    except ImportError:
        pass
    
    return info


def benchmark_data_loading() -> List[BenchmarkResult]:
    """Benchmark data loading operations."""
    from src.data_processing import load_ratings, load_movies
    
    benchmark = Benchmark(warmup=1)
    results = []
    
    @benchmark.measure("load_ratings_sample")
    def load_ratings_sample():
        return load_ratings(sample_size=100000)
    
    @benchmark.measure("load_movies")
    def load_movies_bench():
        return load_movies()
    
    results.append(benchmark.run("load_ratings_sample", iterations=3))
    results.append(benchmark.run("load_movies", iterations=3))
    
    return results


def benchmark_recommendations(sample_users: int = 10) -> List[BenchmarkResult]:
    """Benchmark recommendation generation."""
    from src.data_processing import load_ratings, load_movies
    from src.algorithms.algorithm_manager import AlgorithmManager, AlgorithmType
    import random
    
    results = []
    
    # Load data once
    print("Loading data for benchmarks...")
    ratings_df = load_ratings(sample_size=500000)
    movies_df = load_movies()
    
    # Get sample user IDs
    user_ids = random.sample(list(ratings_df['userId'].unique()), min(sample_users, 100))
    
    # Initialize manager
    manager = AlgorithmManager()
    manager.initialize_data(ratings_df, movies_df)
    
    benchmark = Benchmark(warmup=1)
    
    # Benchmark each algorithm
    for algo_type in [AlgorithmType.SVD]:  # Start with SVD only for speed
        algo_name = algo_type.value
        
        try:
            algorithm = manager.get_algorithm(algo_type)
            
            def get_recs():
                user_id = random.choice(user_ids)
                return algorithm.get_recommendations(user_id, n=10, exclude_rated=True)
            
            benchmark.register(f"recommend_{algo_name}", get_recs)
            result = benchmark.run(f"recommend_{algo_name}", iterations=sample_users)
            result.metadata['algorithm'] = algo_name
            result.metadata['sample_users'] = sample_users
            results.append(result)
            
        except Exception as e:
            logger.warning(f"Failed to benchmark {algo_name}: {e}")
    
    return results


def benchmark_cache() -> List[BenchmarkResult]:
    """Benchmark caching operations."""
    from src.utils.cache import TTLCache, RecommendationCache
    import pandas as pd
    
    benchmark = Benchmark(warmup=10)
    results = []
    
    # TTL Cache operations
    cache = TTLCache(maxsize=1000, ttl=60)
    
    @benchmark.measure("cache_set")
    def cache_set():
        for i in range(100):
            cache.set(f"key_{i}", f"value_{i}")
    
    @benchmark.measure("cache_get")
    def cache_get():
        for i in range(100):
            cache.get(f"key_{i}")
    
    results.append(benchmark.run("cache_set", iterations=100))
    results.append(benchmark.run("cache_get", iterations=100))
    
    # Recommendation cache with DataFrame
    rec_cache = RecommendationCache(maxsize=100, ttl=60)
    sample_df = pd.DataFrame({
        'movieId': range(10),
        'title': [f'Movie {i}' for i in range(10)],
        'predicted_rating': [4.0] * 10
    })
    
    @benchmark.measure("rec_cache_set")
    def rec_cache_set():
        for i in range(50):
            rec_cache.cache_recommendations(i, 'svd', sample_df)
    
    @benchmark.measure("rec_cache_get")
    def rec_cache_get():
        for i in range(50):
            rec_cache.get_recommendations(i, 'svd')
    
    results.append(benchmark.run("rec_cache_set", iterations=50))
    results.append(benchmark.run("rec_cache_get", iterations=50))
    
    return results


def benchmark_search() -> List[BenchmarkResult]:
    """Benchmark search operations."""
    from src.data_processing import load_movies
    from src.search_engine import search_movies_by_criteria
    
    benchmark = Benchmark(warmup=3)
    results = []
    
    movies_df = load_movies()
    
    @benchmark.measure("search_by_title")
    def search_title():
        return search_movies_by_criteria(movies_df, title_query="matrix")
    
    @benchmark.measure("search_by_genre")
    def search_genre():
        return search_movies_by_criteria(movies_df, genre_filter="Action")
    
    results.append(benchmark.run("search_by_title", iterations=100))
    results.append(benchmark.run("search_by_genre", iterations=100))
    
    return results


def run_all_benchmarks() -> BenchmarkSuite:
    """Run all benchmarks and return suite."""
    suite = BenchmarkSuite(
        name="CineMatch Full Benchmark Suite",
        system_info=get_system_info()
    )
    
    print("\n" + "=" * 60)
    print("CineMatch Benchmark Suite")
    print("=" * 60)
    
    # Data loading
    print("\n📊 Benchmarking data loading...")
    try:
        suite.results.extend(benchmark_data_loading())
    except Exception as e:
        logger.error(f"Data loading benchmark failed: {e}")
    
    # Search
    print("\n🔍 Benchmarking search...")
    try:
        suite.results.extend(benchmark_search())
    except Exception as e:
        logger.error(f"Search benchmark failed: {e}")
    
    # Cache
    print("\n💾 Benchmarking cache...")
    try:
        suite.results.extend(benchmark_cache())
    except Exception as e:
        logger.error(f"Cache benchmark failed: {e}")
    
    # Recommendations (longer)
    print("\n🎬 Benchmarking recommendations...")
    try:
        suite.results.extend(benchmark_recommendations(sample_users=5))
    except Exception as e:
        logger.error(f"Recommendations benchmark failed: {e}")
    
    return suite


def print_results(suite: BenchmarkSuite) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    print(f"\nSystem: {suite.system_info.get('platform', 'Unknown')}")
    print(f"Python: {suite.system_info.get('python_version', 'Unknown')}")
    print(f"Timestamp: {suite.timestamp}")
    
    print("\n" + "-" * 80)
    print(f"{'Name':<30} {'Avg Time':>12} {'Min':>10} {'Max':>10} {'Throughput':>12}")
    print("-" * 80)
    
    for result in suite.results:
        avg = f"{result.avg_time*1000:.2f}ms"
        min_t = f"{result.min_time*1000:.2f}ms"
        max_t = f"{result.max_time*1000:.2f}ms"
        throughput = f"{result.throughput:.1f}/s"
        
        print(f"{result.name:<30} {avg:>12} {min_t:>10} {max_t:>10} {throughput:>12}")
    
    print("-" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CineMatch Performance Benchmarks")
    parser.add_argument(
        "--suite",
        choices=["all", "data", "search", "cache", "recommendations"],
        default="all",
        help="Benchmark suite to run"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for JSON results"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations per benchmark"
    )
    
    args = parser.parse_args()
    
    # Run benchmarks
    if args.suite == "all":
        suite = run_all_benchmarks()
    else:
        suite = BenchmarkSuite(
            name=f"CineMatch {args.suite.title()} Benchmarks",
            system_info=get_system_info()
        )
        
        if args.suite == "data":
            suite.results.extend(benchmark_data_loading())
        elif args.suite == "search":
            suite.results.extend(benchmark_search())
        elif args.suite == "cache":
            suite.results.extend(benchmark_cache())
        elif args.suite == "recommendations":
            suite.results.extend(benchmark_recommendations())
    
    # Print results
    print_results(suite)
    
    # Save to file
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(suite.to_dict(), f, indent=2)
        print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
