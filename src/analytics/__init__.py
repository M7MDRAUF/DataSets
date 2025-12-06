"""
CineMatch V2.1.6 - Analytics Module

Provides benchmark engine, performance metrics, and analytics utilities.
"""

from src.analytics.benchmark_engine import (
    BenchmarkEngine,
    BenchmarkResult,
    BenchmarkProgress,
    BenchmarkStatus,
    BenchmarkCache,
    get_benchmark_cache,
    run_parallel_benchmarks
)

__all__ = [
    'BenchmarkEngine',
    'BenchmarkResult',
    'BenchmarkProgress',
    'BenchmarkStatus',
    'BenchmarkCache',
    'get_benchmark_cache',
    'run_parallel_benchmarks'
]
