"""
CineMatch V2.1.6 - Benchmark Engine

Provides parallel benchmark execution with caching, progress reporting,
timeouts, and cancellation support for the Analytics page.

Author: CineMatch Team
Date: December 2025
"""

import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib

logger = logging.getLogger(__name__)


class BenchmarkStatus(Enum):
    """Status of a benchmark run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class BenchmarkResult:
    """Result of a single algorithm benchmark."""
    algorithm_name: str
    status: BenchmarkStatus
    rmse: Optional[float] = None
    mae: Optional[float] = None
    training_time: Optional[float] = None
    sample_size: Optional[int] = None
    coverage: Optional[float] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'Algorithm': self.algorithm_name,
            'RMSE': self.rmse if self.rmse is not None else 'N/A',
            'MAE': self.mae if self.mae is not None else 'N/A',
            'Training Time (s)': self.training_time if self.training_time is not None else 'N/A',
            'Sample Size': self.sample_size if self.sample_size is not None else 'N/A',
            'Coverage (%)': self.coverage if self.coverage is not None else 'N/A',
            'Status': self.status.value,
            'Execution Time (s)': round(self.execution_time, 2)
        }


@dataclass
class BenchmarkProgress:
    """Progress tracking for benchmark execution."""
    total_algorithms: int
    completed_count: int = 0
    current_algorithm: Optional[str] = None
    results: List[BenchmarkResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    
    @property
    def progress_percentage(self) -> float:
        """Get progress as percentage."""
        if self.total_algorithms == 0:
            return 100.0
        return (self.completed_count / self.total_algorithms) * 100
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return (datetime.now() - self.start_time).total_seconds()


class BenchmarkCache:
    """
    Cache for benchmark results with expiration.
    
    Results are cached based on algorithm type and data hash to avoid
    re-running benchmarks unnecessarily.
    """
    
    def __init__(self, expiration_minutes: int = 30):
        self._cache: Dict[str, Tuple[List[BenchmarkResult], datetime]] = {}
        self._expiration = timedelta(minutes=expiration_minutes)
        self._lock = threading.Lock()
    
    def _generate_cache_key(self, algorithm_types: List[str], data_hash: str) -> str:
        """Generate cache key from algorithm types and data hash."""
        algo_str = ",".join(sorted(algorithm_types))
        combined = f"{algo_str}:{data_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, algorithm_types: List[str], data_hash: str) -> Optional[List[BenchmarkResult]]:
        """Get cached results if available and not expired."""
        with self._lock:
            key = self._generate_cache_key(algorithm_types, data_hash)
            if key in self._cache:
                results, timestamp = self._cache[key]
                if datetime.now() - timestamp < self._expiration:
                    logger.info(f"Benchmark cache hit for key {key[:8]}...")
                    return results
                else:
                    # Expired, remove from cache
                    del self._cache[key]
                    logger.info(f"Benchmark cache expired for key {key[:8]}...")
            return None
    
    def set(self, algorithm_types: List[str], data_hash: str, results: List[BenchmarkResult]) -> None:
        """Store results in cache."""
        with self._lock:
            key = self._generate_cache_key(algorithm_types, data_hash)
            self._cache[key] = (results, datetime.now())
            logger.info(f"Benchmark results cached with key {key[:8]}...")
    
    def clear(self) -> None:
        """Clear all cached results."""
        with self._lock:
            self._cache.clear()
            logger.info("Benchmark cache cleared")


# Global benchmark cache instance
_benchmark_cache = BenchmarkCache()


def get_benchmark_cache() -> BenchmarkCache:
    """Get the global benchmark cache instance."""
    return _benchmark_cache


class BenchmarkEngine:
    """
    Engine for running algorithm benchmarks with parallel execution,
    progress tracking, timeouts, and cancellation support.
    """
    
    def __init__(
        self,
        manager,  # AlgorithmManager instance
        max_workers: int = 3,
        timeout_seconds: int = 120,
        use_cache: bool = True
    ):
        """
        Initialize benchmark engine.
        
        Args:
            manager: AlgorithmManager instance for accessing algorithms
            max_workers: Maximum parallel benchmark threads
            timeout_seconds: Timeout for each individual benchmark
            use_cache: Whether to use result caching
        """
        self._manager = manager
        self._max_workers = max_workers
        self._timeout = timeout_seconds
        self._use_cache = use_cache
        self._cancel_requested = threading.Event()
        self._progress: Optional[BenchmarkProgress] = None
        self._progress_callback: Optional[Callable[[BenchmarkProgress], None]] = None
        self._lock = threading.Lock()
    
    def set_progress_callback(self, callback: Callable[[BenchmarkProgress], None]) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback
    
    def request_cancellation(self) -> None:
        """Request cancellation of running benchmarks."""
        logger.info("Benchmark cancellation requested")
        self._cancel_requested.set()
    
    def reset_cancellation(self) -> None:
        """Reset cancellation flag for new benchmark run."""
        self._cancel_requested.clear()
    
    def _benchmark_single_algorithm(
        self,
        algo_type,  # AlgorithmType enum
        algo_name: str
    ) -> BenchmarkResult:
        """
        Run benchmark for a single algorithm.
        
        Args:
            algo_type: AlgorithmType enum value
            algo_name: Human-readable algorithm name
            
        Returns:
            BenchmarkResult with metrics or error information
        """
        start_time = time.time()
        
        # Check for cancellation
        if self._cancel_requested.is_set():
            return BenchmarkResult(
                algorithm_name=algo_name,
                status=BenchmarkStatus.CANCELLED,
                execution_time=time.time() - start_time
            )
        
        try:
            # Try to get cached metrics first
            metrics_data = self._manager.get_algorithm_metrics(algo_type)
            
            # If not cached or not trained, load the algorithm
            if not metrics_data or metrics_data.get('status') != 'Trained':
                logger.info(f"Loading algorithm {algo_name} for benchmarking...")
                algorithm = self._manager.get_algorithm(algo_type)
                metrics_data = self._manager.get_algorithm_metrics(algo_type)
            
            # Check for cancellation again after loading
            if self._cancel_requested.is_set():
                return BenchmarkResult(
                    algorithm_name=algo_name,
                    status=BenchmarkStatus.CANCELLED,
                    execution_time=time.time() - start_time
                )
            
            if metrics_data and metrics_data.get('status') == 'Trained':
                metrics = metrics_data.get('metrics', {})
                return BenchmarkResult(
                    algorithm_name=algo_name,
                    status=BenchmarkStatus.COMPLETED,
                    rmse=metrics.get('rmse'),
                    mae=metrics.get('mae'),
                    training_time=metrics_data.get('training_time'),
                    sample_size=metrics.get('sample_size'),
                    coverage=metrics.get('coverage'),
                    execution_time=time.time() - start_time
                )
            else:
                return BenchmarkResult(
                    algorithm_name=algo_name,
                    status=BenchmarkStatus.FAILED,
                    error_message="Algorithm not trained or metrics unavailable",
                    execution_time=time.time() - start_time
                )
                
        except Exception as e:
            logger.error(f"Benchmark failed for {algo_name}: {e}")
            return BenchmarkResult(
                algorithm_name=algo_name,
                status=BenchmarkStatus.FAILED,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def _update_progress(self, result: BenchmarkResult) -> None:
        """Update progress tracking and notify callback."""
        with self._lock:
            if self._progress:
                self._progress.completed_count += 1
                self._progress.results.append(result)
                self._progress.current_algorithm = None
                
                if self._progress_callback:
                    try:
                        self._progress_callback(self._progress)
                    except Exception as e:
                        logger.error(f"Progress callback failed: {e}")
    
    def run_benchmarks(
        self,
        algorithm_types: List[Tuple[Any, str]],  # List of (AlgorithmType, name) tuples
        data_hash: str = ""
    ) -> List[BenchmarkResult]:
        """
        Run benchmarks for multiple algorithms in parallel.
        
        Args:
            algorithm_types: List of (AlgorithmType, display_name) tuples
            data_hash: Hash of data for cache key (optional)
            
        Returns:
            List of BenchmarkResult objects
        """
        # Reset cancellation flag
        self.reset_cancellation()
        
        # Check cache first
        if self._use_cache and data_hash:
            algo_names = [name for _, name in algorithm_types]
            cached = get_benchmark_cache().get(algo_names, data_hash)
            if cached:
                logger.info("Returning cached benchmark results")
                return cached
        
        # Initialize progress tracking
        self._progress = BenchmarkProgress(total_algorithms=len(algorithm_types))
        
        results: List[BenchmarkResult] = []
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            # Submit all benchmark tasks
            future_to_algo = {
                executor.submit(
                    self._benchmark_single_algorithm,
                    algo_type,
                    algo_name
                ): (algo_type, algo_name)
                for algo_type, algo_name in algorithm_types
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_algo):
                algo_type, algo_name = future_to_algo[future]
                
                # Check for cancellation
                if self._cancel_requested.is_set():
                    # Cancel remaining futures
                    for remaining_future in future_to_algo:
                        remaining_future.cancel()
                    break
                
                try:
                    # Get result with timeout
                    result = future.result(timeout=self._timeout)
                    results.append(result)
                    self._update_progress(result)
                    
                except FuturesTimeoutError:
                    logger.warning(f"Benchmark timeout for {algo_name}")
                    result = BenchmarkResult(
                        algorithm_name=algo_name,
                        status=BenchmarkStatus.TIMEOUT,
                        error_message=f"Timed out after {self._timeout}s"
                    )
                    results.append(result)
                    self._update_progress(result)
                    
                except Exception as e:
                    logger.error(f"Benchmark error for {algo_name}: {e}")
                    result = BenchmarkResult(
                        algorithm_name=algo_name,
                        status=BenchmarkStatus.FAILED,
                        error_message=str(e)
                    )
                    results.append(result)
                    self._update_progress(result)
        
        # Cache results if not cancelled
        if self._use_cache and data_hash and not self._cancel_requested.is_set():
            algo_names = [name for _, name in algorithm_types]
            get_benchmark_cache().set(algo_names, data_hash, results)
        
        # Sort results by algorithm order
        algo_order = {name: i for i, (_, name) in enumerate(algorithm_types)}
        results.sort(key=lambda r: algo_order.get(r.algorithm_name, 999))
        
        return results
    
    def get_progress(self) -> Optional[BenchmarkProgress]:
        """Get current progress information."""
        return self._progress


def run_parallel_benchmarks(
    manager,
    algorithm_types: List[Tuple[Any, str]],
    data_hash: str = "",
    max_workers: int = 3,
    timeout_seconds: int = 120,
    progress_callback: Optional[Callable[[BenchmarkProgress], None]] = None,
    use_cache: bool = True
) -> List[BenchmarkResult]:
    """
    Convenience function to run parallel benchmarks.
    
    Args:
        manager: AlgorithmManager instance
        algorithm_types: List of (AlgorithmType, display_name) tuples
        data_hash: Hash of data for cache key
        max_workers: Maximum parallel threads
        timeout_seconds: Timeout per algorithm
        progress_callback: Callback for progress updates
        use_cache: Whether to use result caching
        
    Returns:
        List of BenchmarkResult objects
    """
    engine = BenchmarkEngine(
        manager=manager,
        max_workers=max_workers,
        timeout_seconds=timeout_seconds,
        use_cache=use_cache
    )
    
    if progress_callback:
        engine.set_progress_callback(progress_callback)
    
    return engine.run_benchmarks(algorithm_types, data_hash)
