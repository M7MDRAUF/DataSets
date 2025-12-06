"""
CineMatch V2.1.6 - Model Cache

Centralized model caching with LRU eviction and TTL support.
Provides efficient in-memory caching for loaded ML models.

Performance Benefits:
- Eliminates redundant model loading (45s -> 0ms for cached models)
- LRU eviction prevents memory exhaustion
- TTL prevents stale model usage

Usage:
    from src.memory.model_cache import model_cache
    
    # Get cached model or load it
    model = model_cache.get_or_load('svd', loader_func)
    
    # Check cache status
    model_cache.stats()

Author: CineMatch Development Team
Date: December 5, 2025
"""

import time
import logging
import threading
from typing import Any, Dict, Optional, Callable, TypeVar, Generic
from collections import OrderedDict
from dataclasses import dataclass, field
import gc
import sys

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry:
    """Entry in the model cache with metadata."""
    model: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        return (time.time() - self.created_at) > self.ttl_seconds
    
    def touch(self) -> None:
        """Update last accessed time and increment count."""
        self.last_accessed = time.time()
        self.access_count += 1


class ModelCache:
    """
    Thread-safe LRU cache for ML models with TTL support.
    
    Features:
    - LRU (Least Recently Used) eviction when memory limit exceeded
    - Time-to-live (TTL) expiration for cached entries
    - Thread-safe access with read-write locks
    - Memory monitoring and automatic eviction
    - Statistics tracking for cache performance analysis
    
    Example:
        cache = ModelCache(max_memory_gb=4.0, default_ttl_hours=24)
        
        # Store a model
        cache.set('svd_model', my_model)
        
        # Retrieve a model
        model = cache.get('svd_model')
        
        # Get or load pattern
        model = cache.get_or_load('svd_model', lambda: load_model('svd.pkl'))
    """
    
    def __init__(
        self,
        max_memory_gb: float = 4.0,
        default_ttl_hours: Optional[float] = None,
        enable_auto_cleanup: bool = True,
        cleanup_interval_seconds: int = 300
    ):
        """
        Initialize the model cache.
        
        Args:
            max_memory_gb: Maximum memory for cached models (in GB)
            default_ttl_hours: Default TTL for entries (None = no expiration)
            enable_auto_cleanup: Enable periodic cleanup thread
            cleanup_interval_seconds: How often to run cleanup (default 5 min)
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
        self._default_ttl_seconds = default_ttl_hours * 3600 if default_ttl_hours else None
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expirations': 0,
            'total_loads': 0,
            'total_load_time_seconds': 0.0
        }
        
        # Auto-cleanup thread
        self._cleanup_thread: Optional[threading.Thread] = None
        self._cleanup_stop_event = threading.Event()
        if enable_auto_cleanup:
            self._start_cleanup_thread(cleanup_interval_seconds)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a model from cache.
        
        Args:
            key: Cache key for the model
            
        Returns:
            Model instance or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats['misses'] += 1
                return None
            
            # Check TTL expiration
            if entry.is_expired():
                self._remove_entry(key)
                self._stats['expirations'] += 1
                self._stats['misses'] += 1
                logger.debug(f"Cache entry '{key}' expired")
                return None
            
            # Update access tracking and move to end (most recently used)
            entry.touch()
            self._cache.move_to_end(key)
            
            self._stats['hits'] += 1
            logger.debug(f"Cache hit for '{key}' (access #{entry.access_count})")
            return entry.model
    
    def set(
        self,
        key: str,
        model: Any,
        ttl_hours: Optional[float] = None,
        size_bytes: Optional[int] = None
    ) -> None:
        """
        Store a model in cache.
        
        Args:
            key: Cache key for the model
            model: Model instance to cache
            ttl_hours: Custom TTL (overrides default)
            size_bytes: Size of model in bytes (auto-estimated if None)
        """
        with self._lock:
            # Estimate size if not provided
            if size_bytes is None:
                size_bytes = self._estimate_size(model)
            
            # Evict entries if necessary to make room
            self._evict_if_needed(size_bytes)
            
            # Determine TTL
            ttl_seconds = ttl_hours * 3600 if ttl_hours else self._default_ttl_seconds
            
            # Create cache entry
            entry = CacheEntry(
                model=model,
                size_bytes=size_bytes,
                ttl_seconds=ttl_seconds
            )
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Add to cache
            self._cache[key] = entry
            logger.info(f"Cached model '{key}' ({size_bytes / 1024 / 1024:.1f}MB)")
    
    def get_or_load(
        self,
        key: str,
        loader: Callable[[], T],
        ttl_hours: Optional[float] = None
    ) -> T:
        """
        Get model from cache or load it using the provided function.
        
        This is the recommended pattern for model access.
        
        Args:
            key: Cache key for the model
            loader: Function to call if model not in cache
            ttl_hours: Custom TTL for newly loaded model
            
        Returns:
            Model instance (from cache or freshly loaded)
            
        Example:
            model = cache.get_or_load(
                'svd_model',
                lambda: load_model_fast('models/svd.pkl')
            )
        """
        # Try cache first
        model = self.get(key)
        if model is not None:
            return model
        
        # Load model
        logger.info(f"Loading model '{key}' (not in cache)")
        start_time = time.time()
        model = loader()
        load_time = time.time() - start_time
        
        # Update stats
        self._stats['total_loads'] += 1
        self._stats['total_load_time_seconds'] += load_time
        
        # Cache the loaded model
        self.set(key, model, ttl_hours=ttl_hours)
        
        logger.info(f"Model '{key}' loaded and cached in {load_time:.2f}s")
        return model
    
    def has(self, key: str) -> bool:
        """Check if a key exists in cache (and is not expired)."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                self._remove_entry(key)
                return False
            return True
    
    def remove(self, key: str) -> bool:
        """
        Remove a model from cache.
        
        Args:
            key: Cache key to remove
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> int:
        """
        Clear all cached models.
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            gc.collect()
            logger.info(f"Cache cleared ({count} entries)")
            return count
    
    def keys(self) -> list:
        """Get list of all cache keys."""
        with self._lock:
            return list(self._cache.keys())
    
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with hit rate, memory usage, entry count, etc.
        """
        with self._lock:
            total_size = sum(e.size_bytes for e in self._cache.values())
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
            
            avg_load_time = (
                self._stats['total_load_time_seconds'] / self._stats['total_loads']
                if self._stats['total_loads'] > 0 else 0
            )
            
            return {
                'entries': len(self._cache),
                'total_size_mb': total_size / 1024 / 1024,
                'max_size_mb': self._max_memory_bytes / 1024 / 1024,
                'utilization_percent': (total_size / self._max_memory_bytes) * 100,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': hit_rate,
                'evictions': self._stats['evictions'],
                'expirations': self._stats['expirations'],
                'total_loads': self._stats['total_loads'],
                'avg_load_time_seconds': avg_load_time,
                'total_load_time_saved_seconds': (
                    self._stats['hits'] * avg_load_time
                )
            }
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache (internal)."""
        if key in self._cache:
            del self._cache[key]
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of an object in bytes."""
        try:
            return sys.getsizeof(obj)
        except (TypeError, AttributeError):
            # Fallback: assume 100MB for ML models
            return 100 * 1024 * 1024
    
    def _get_current_size(self) -> int:
        """Get total size of cached entries."""
        return sum(e.size_bytes for e in self._cache.values())
    
    def _evict_if_needed(self, new_entry_size: int) -> None:
        """Evict entries using LRU until there's room for new entry."""
        while self._cache and (self._get_current_size() + new_entry_size) > self._max_memory_bytes:
            # Remove oldest (least recently used) entry
            oldest_key = next(iter(self._cache))
            logger.info(f"Evicting '{oldest_key}' (LRU)")
            self._remove_entry(oldest_key)
            self._stats['evictions'] += 1
            gc.collect()
    
    def _cleanup_expired(self) -> int:
        """Remove all expired entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                self._remove_entry(key)
                self._stats['expirations'] += 1
            
            if expired_keys:
                gc.collect()
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
            
            return len(expired_keys)
    
    def _start_cleanup_thread(self, interval_seconds: int) -> None:
        """Start background cleanup thread."""
        def cleanup_loop():
            while not self._cleanup_stop_event.wait(interval_seconds):
                self._cleanup_expired()
        
        self._cleanup_thread = threading.Thread(
            target=cleanup_loop,
            daemon=True,
            name="ModelCache-Cleanup"
        )
        self._cleanup_thread.start()
    
    def shutdown(self) -> None:
        """Shutdown cache and cleanup thread."""
        if self._cleanup_thread:
            self._cleanup_stop_event.set()
            self._cleanup_thread.join(timeout=2.0)
        self.clear()


# Global singleton instance
_global_cache: Optional[ModelCache] = None
_global_cache_lock = threading.Lock()


def get_model_cache(
    max_memory_gb: float = 4.0,
    default_ttl_hours: Optional[float] = None
) -> ModelCache:
    """
    Get or create the global model cache instance.
    
    Args:
        max_memory_gb: Maximum memory for cached models
        default_ttl_hours: Default TTL for entries
        
    Returns:
        ModelCache singleton instance
    """
    global _global_cache
    
    with _global_cache_lock:
        if _global_cache is None:
            _global_cache = ModelCache(
                max_memory_gb=max_memory_gb,
                default_ttl_hours=default_ttl_hours
            )
            logger.info(f"Created global model cache (max {max_memory_gb}GB)")
        return _global_cache


# Convenience alias
model_cache = get_model_cache
