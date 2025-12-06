"""
CineMatch V2.1.6 - Model Preloader

Background model preloading for reduced first-request latency.
Implements Tasks 27-30 of the model loading performance optimization plan.

Features:
- Startup preloading for default algorithm
- Background thread for async preloading
- Priority queue based on usage frequency
- Usage statistics tracking

Usage:
    from src.memory.model_preloader import ModelPreloader, preload_on_startup
    
    # At application startup
    preload_on_startup()
    
    # Or with custom settings
    preloader = ModelPreloader()
    preloader.preload_async(['hybrid', 'svd'])

Author: CineMatch Development Team
Date: December 5, 2025
"""

import os
import time
import logging
import threading
import json
from typing import Any, Dict, List, Optional, Callable
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from queue import PriorityQueue

logger = logging.getLogger(__name__)


@dataclass
class UsageStats:
    """Track model usage statistics for prioritization."""
    algorithm: str
    load_count: int = 0
    total_load_time: float = 0.0
    last_used: float = field(default_factory=time.time)
    request_count: int = 0
    
    @property
    def avg_load_time(self) -> float:
        """Average load time in seconds."""
        return self.total_load_time / self.load_count if self.load_count > 0 else 0.0
    
    @property
    def priority_score(self) -> float:
        """
        Higher score = higher priority for preloading.
        Based on request frequency and recency.
        """
        recency_factor = 1.0 / (time.time() - self.last_used + 1)
        return self.request_count * recency_factor


@dataclass(order=True)
class PreloadTask:
    """Task for priority queue ordering."""
    priority: float
    algorithm: str = field(compare=False)


class ModelPreloader:
    """
    Background model preloader with priority-based loading.
    
    Features:
    - Preload default algorithm on startup
    - Background thread for non-blocking preload
    - Priority queue based on usage frequency
    - Usage statistics for intelligent preloading
    
    Example:
        preloader = ModelPreloader()
        
        # Preload synchronously
        preloader.preload_sync(['hybrid'])
        
        # Preload in background
        preloader.preload_async(['svd', 'content_based'])
        
        # Record usage for priority calculation
        preloader.record_usage('hybrid')
    """
    
    # Default priority order
    DEFAULT_PRIORITY = [
        'hybrid',
        'svd',
        'content_based',
        'item_knn',
        'user_knn'
    ]
    
    def __init__(
        self,
        algorithm_loader: Optional[Callable[[str], Any]] = None,
        priority_order: Optional[List[str]] = None,
        stats_file: Optional[str] = None
    ):
        """
        Initialize the model preloader.
        
        Args:
            algorithm_loader: Function to load an algorithm by name
            priority_order: Custom priority order for preloading
            stats_file: Path to persist usage statistics
        """
        self._loader = algorithm_loader
        self._priority = priority_order or self.DEFAULT_PRIORITY
        self._stats_file = stats_file or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'models',
            'preload_stats.json'
        )
        
        # Usage statistics
        self._usage_stats: Dict[str, UsageStats] = {}
        self._load_usage_stats()
        
        # Background preloading
        self._preload_queue: PriorityQueue = PriorityQueue()
        self._preload_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # Preload status tracking
        self._preloaded: set = set()
        self._preloading: set = set()
        self._preload_errors: Dict[str, str] = {}
    
    def _load_usage_stats(self) -> None:
        """Load usage statistics from disk."""
        try:
            if os.path.exists(self._stats_file):
                with open(self._stats_file, 'r') as f:
                    data = json.load(f)
                for algo, stats in data.items():
                    self._usage_stats[algo] = UsageStats(
                        algorithm=algo,
                        load_count=stats.get('load_count', 0),
                        total_load_time=stats.get('total_load_time', 0.0),
                        last_used=stats.get('last_used', time.time()),
                        request_count=stats.get('request_count', 0)
                    )
                logger.debug(f"Loaded usage stats for {len(self._usage_stats)} algorithms")
        except Exception as e:
            logger.warning(f"Could not load usage stats: {e}")
    
    def _save_usage_stats(self) -> None:
        """Save usage statistics to disk."""
        try:
            data = {}
            for algo, stats in self._usage_stats.items():
                data[algo] = {
                    'load_count': stats.load_count,
                    'total_load_time': stats.total_load_time,
                    'last_used': stats.last_used,
                    'request_count': stats.request_count
                }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self._stats_file), exist_ok=True)
            
            with open(self._stats_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved usage stats for {len(data)} algorithms")
        except Exception as e:
            logger.warning(f"Could not save usage stats: {e}")
    
    def set_loader(self, loader: Callable[[str], Any]) -> None:
        """Set the algorithm loader function."""
        self._loader = loader
    
    def record_usage(
        self,
        algorithm: str,
        load_time: Optional[float] = None,
        was_loaded: bool = False
    ) -> None:
        """
        Record usage of an algorithm for prioritization.
        
        Args:
            algorithm: Algorithm name
            load_time: Time taken to load (if loaded)
            was_loaded: Whether the model was loaded (vs cache hit)
        """
        with self._lock:
            if algorithm not in self._usage_stats:
                self._usage_stats[algorithm] = UsageStats(algorithm=algorithm)
            
            stats = self._usage_stats[algorithm]
            stats.request_count += 1
            stats.last_used = time.time()
            
            if was_loaded and load_time is not None:
                stats.load_count += 1
                stats.total_load_time += load_time
            
            # Periodically save stats
            if stats.request_count % 10 == 0:
                self._save_usage_stats()
    
    def get_priority_order(self) -> List[str]:
        """
        Get algorithms ordered by priority for preloading.
        
        Combines configured priority with usage-based prioritization.
        """
        with self._lock:
            # Start with configured priority
            ordered = list(self._priority)
            
            # If we have usage data, re-order by priority score
            if self._usage_stats:
                # Get algorithms with usage data, sorted by score
                scored = sorted(
                    self._usage_stats.values(),
                    key=lambda s: s.priority_score,
                    reverse=True
                )
                
                # Merge: usage-based first, then configured order for remaining
                usage_order = [s.algorithm for s in scored]
                remaining = [a for a in ordered if a not in usage_order]
                ordered = usage_order + remaining
            
            return ordered
    
    def preload_sync(
        self,
        algorithms: Optional[List[str]] = None,
        timeout_per_model: float = 60.0
    ) -> Dict[str, bool]:
        """
        Preload models synchronously (blocking).
        
        Args:
            algorithms: List of algorithms to preload (default: priority order)
            timeout_per_model: Timeout for each model load
            
        Returns:
            Dict mapping algorithm name to success status
        """
        if self._loader is None:
            logger.error("No algorithm loader configured")
            return {}
        
        algorithms = algorithms or self.get_priority_order()
        results = {}
        
        logger.info(f"Preloading {len(algorithms)} models synchronously: {algorithms}")
        
        for algo in algorithms:
            if algo in self._preloaded:
                logger.debug(f"Skipping {algo} (already preloaded)")
                results[algo] = True
                continue
            
            try:
                self._preloading.add(algo)
                start_time = time.time()
                
                logger.info(f"Preloading model: {algo}")
                self._loader(algo)
                
                load_time = time.time() - start_time
                self._preloaded.add(algo)
                self.record_usage(algo, load_time=load_time, was_loaded=True)
                
                results[algo] = True
                logger.info(f"Preloaded {algo} in {load_time:.2f}s")
                
            except Exception as e:
                self._preload_errors[algo] = str(e)
                results[algo] = False
                logger.error(f"Failed to preload {algo}: {e}")
            finally:
                self._preloading.discard(algo)
        
        return results
    
    def preload_async(
        self,
        algorithms: Optional[List[str]] = None
    ) -> None:
        """
        Preload models in background thread (non-blocking).
        
        Args:
            algorithms: List of algorithms to preload
        """
        algorithms = algorithms or self.get_priority_order()
        
        # Add to priority queue
        for i, algo in enumerate(algorithms):
            if algo not in self._preloaded and algo not in self._preloading:
                # Lower priority number = higher priority
                priority = i
                self._preload_queue.put(PreloadTask(priority, algo))
        
        # Start background thread if not running
        self._start_background_thread()
    
    def _start_background_thread(self) -> None:
        """Start the background preload thread."""
        if self._preload_thread is not None and self._preload_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._preload_thread = threading.Thread(
            target=self._background_preload_loop,
            daemon=True,
            name="ModelPreloader-Background"
        )
        self._preload_thread.start()
        logger.info("Started background model preloader thread")
    
    def _background_preload_loop(self) -> None:
        """Background thread loop for async preloading."""
        while not self._stop_event.is_set():
            try:
                # Wait for task with timeout
                try:
                    task = self._preload_queue.get(timeout=1.0)
                except:
                    continue
                
                if task.algorithm in self._preloaded:
                    continue
                
                # Preload the model
                try:
                    self._preloading.add(task.algorithm)
                    start_time = time.time()
                    
                    logger.info(f"[Background] Preloading: {task.algorithm}")
                    
                    if self._loader:
                        self._loader(task.algorithm)
                        
                    load_time = time.time() - start_time
                    self._preloaded.add(task.algorithm)
                    self.record_usage(
                        task.algorithm,
                        load_time=load_time,
                        was_loaded=True
                    )
                    
                    logger.info(
                        f"[Background] Preloaded {task.algorithm} in {load_time:.2f}s"
                    )
                    
                except Exception as e:
                    self._preload_errors[task.algorithm] = str(e)
                    logger.error(f"[Background] Failed to preload {task.algorithm}: {e}")
                finally:
                    self._preloading.discard(task.algorithm)
                    
            except Exception as e:
                logger.error(f"[Background] Preload loop error: {e}")
        
        logger.info("Background preloader thread stopped")
    
    def stop(self) -> None:
        """Stop background preloading."""
        self._stop_event.set()
        if self._preload_thread:
            self._preload_thread.join(timeout=5.0)
        self._save_usage_stats()
    
    def status(self) -> Dict[str, Any]:
        """Get preloader status."""
        with self._lock:
            return {
                'preloaded': list(self._preloaded),
                'preloading': list(self._preloading),
                'queued': self._preload_queue.qsize(),
                'errors': dict(self._preload_errors),
                'background_running': (
                    self._preload_thread is not None and
                    self._preload_thread.is_alive()
                ),
                'usage_stats': {
                    algo: {
                        'request_count': stats.request_count,
                        'avg_load_time': stats.avg_load_time,
                        'priority_score': stats.priority_score
                    }
                    for algo, stats in self._usage_stats.items()
                }
            }
    
    def is_preloaded(self, algorithm: str) -> bool:
        """Check if an algorithm has been preloaded."""
        return algorithm in self._preloaded


# Global singleton instance
_global_preloader: Optional[ModelPreloader] = None
_global_preloader_lock = threading.Lock()


def get_model_preloader() -> ModelPreloader:
    """Get or create the global model preloader instance."""
    global _global_preloader
    
    with _global_preloader_lock:
        if _global_preloader is None:
            _global_preloader = ModelPreloader()
        return _global_preloader


def preload_on_startup(
    algorithms: Optional[List[str]] = None,
    background: bool = True
) -> None:
    """
    Preload models at application startup.
    
    Call this function early in application startup to begin
    preloading models. By default, runs in background to not
    block startup.
    
    Args:
        algorithms: List of algorithms to preload (uses config default)
        background: If True, preload in background thread
        
    Example:
        # In app/main.py
        from src.memory.model_preloader import preload_on_startup
        preload_on_startup()  # Non-blocking background preload
    """
    # Get config settings
    preload_enabled = os.environ.get('MODEL_PRELOAD_ON_STARTUP', 'false').lower() == 'true'
    preload_background = os.environ.get('MODEL_PRELOAD_BACKGROUND', 'true').lower() == 'true'
    priority_str = os.environ.get(
        'MODEL_PRELOAD_PRIORITY',
        'hybrid,svd,content_based,item_knn,user_knn'
    )
    
    if not preload_enabled:
        logger.info("Model preloading disabled (MODEL_PRELOAD_ON_STARTUP=false)")
        return
    
    # Parse priority order from config
    if algorithms is None:
        algorithms = [a.strip() for a in priority_str.split(',') if a.strip()]
    
    # Get or create preloader
    preloader = get_model_preloader()
    
    # Set up loader function (lazy import to avoid circular dependencies)
    def algorithm_loader(name: str) -> Any:
        try:
            from src.algorithms.algorithm_manager import get_algorithm
            return get_algorithm(name, fast_load=True)
        except ImportError:
            logger.warning("Could not import algorithm_manager for preloading")
            return None
    
    preloader.set_loader(algorithm_loader)
    
    # Start preloading
    use_background = background and preload_background
    
    if use_background:
        logger.info(f"Starting background preload for: {algorithms}")
        preloader.preload_async(algorithms)
    else:
        logger.info(f"Starting synchronous preload for: {algorithms}")
        preloader.preload_sync(algorithms)


def record_model_usage(
    algorithm: str,
    load_time: Optional[float] = None,
    was_loaded: bool = False
) -> None:
    """
    Record model usage for preload prioritization.
    
    Call this after loading or using a model to help the preloader
    make intelligent decisions about what to preload first.
    
    Args:
        algorithm: Algorithm name
        load_time: Time taken to load (if applicable)
        was_loaded: Whether model was loaded (vs cache hit)
    """
    preloader = get_model_preloader()
    preloader.record_usage(algorithm, load_time, was_loaded)


# Export convenience functions
__all__ = [
    'ModelPreloader',
    'UsageStats',
    'PreloadTask',
    'get_model_preloader',
    'preload_on_startup',
    'record_model_usage',
]
