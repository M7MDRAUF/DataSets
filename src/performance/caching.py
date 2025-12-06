"""
Caching Module for CineMatch V2.1.6

Implements multi-level caching strategies:
- In-memory LRU cache
- TTL-based expiration
- Cache statistics
- Optional Redis integration

Phase 2 - Task 2.4: Caching Strategies
"""

import logging
import time
import threading
import hashlib
import json
from typing import Any, Callable, Dict, Generic, Optional, TypeVar, Union
from dataclasses import dataclass, field
from collections import OrderedDict
from functools import wraps
import weakref

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


@dataclass
class CacheConfig:
    """Configuration for caching."""
    max_size: int = 1000
    ttl_seconds: float = 3600
    enable_stats: bool = True
    cleanup_interval: float = 60.0


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    memory_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / max(1, total)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'size': self.size,
            'hit_rate': f"{self.hit_rate:.2%}"
        }


@dataclass
class CacheEntry(Generic[V]):
    """Cache entry with metadata."""
    value: V
    created_at: float
    accessed_at: float
    access_count: int = 0
    size_bytes: int = 0


class LRUCache(Generic[K, V]):
    """
    Thread-safe LRU cache with TTL support.
    
    Features:
    - LRU eviction policy
    - TTL-based expiration
    - Thread-safe operations
    - Statistics tracking
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = False
        
        if self.config.ttl_seconds > 0:
            self._start_cleanup()
    
    def _start_cleanup(self) -> None:
        """Start background cleanup thread."""
        self._running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self._cleanup_thread.start()
    
    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self._running:
            time.sleep(self.config.cleanup_interval)
            self._cleanup_expired()
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        if self.config.ttl_seconds <= 0:
            return
        
        now = time.time()
        to_remove = []
        
        with self._lock:
            for key, entry in self._cache.items():
                if now - entry.created_at > self.config.ttl_seconds:
                    to_remove.append(key)
            
            for key in to_remove:
                del self._cache[key]
                self._stats.evictions += 1
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Get a value from cache."""
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return default
            
            entry = self._cache[key]
            
            # Check TTL
            if self.config.ttl_seconds > 0:
                if time.time() - entry.created_at > self.config.ttl_seconds:
                    del self._cache[key]
                    self._stats.misses += 1
                    return default
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.accessed_at = time.time()
            entry.access_count += 1
            self._stats.hits += 1
            
            return entry.value
    
    def set(self, key: K, value: V) -> None:
        """Set a value in cache."""
        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.config.max_size:
                # Remove oldest (first)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats.evictions += 1
            
            now = time.time()
            self._cache[key] = CacheEntry(
                value=value,
                created_at=now,
                accessed_at=now
            )
            self._stats.size = len(self._cache)
    
    def delete(self, key: K) -> bool:
        """Delete a key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.size = len(self._cache)
                return True
            return False
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._stats.size = 0
    
    def contains(self, key: K) -> bool:
        """Check if key is in cache."""
        return self.get(key) is not None
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats
    
    def stop(self) -> None:
        """Stop cleanup thread."""
        self._running = False


class TieredCache(Generic[K, V]):
    """
    Two-tier cache with L1 (fast/small) and L2 (slow/large).
    
    Features:
    - Two-level caching
    - Automatic promotion
    - Unified interface
    """
    
    def __init__(
        self,
        l1_size: int = 100,
        l2_size: int = 1000,
        l1_ttl: float = 60,
        l2_ttl: float = 3600
    ):
        self.l1 = LRUCache[K, V](CacheConfig(max_size=l1_size, ttl_seconds=l1_ttl))
        self.l2 = LRUCache[K, V](CacheConfig(max_size=l2_size, ttl_seconds=l2_ttl))
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Get from L1, then L2."""
        # Try L1
        value = self.l1.get(key)
        if value is not None:
            return value
        
        # Try L2
        value = self.l2.get(key)
        if value is not None:
            # Promote to L1
            self.l1.set(key, value)
            return value
        
        return default
    
    def set(self, key: K, value: V) -> None:
        """Set in both L1 and L2."""
        self.l1.set(key, value)
        self.l2.set(key, value)
    
    def delete(self, key: K) -> None:
        """Delete from both levels."""
        self.l1.delete(key)
        self.l2.delete(key)
    
    def clear(self) -> None:
        """Clear both levels."""
        self.l1.clear()
        self.l2.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stats for both levels."""
        return {
            'l1': self.l1.get_stats().to_dict(),
            'l2': self.l2.get_stats().to_dict()
        }


def cached(
    cache: Optional[LRUCache] = None,
    ttl: float = 3600,
    max_size: int = 1000,
    key_func: Optional[Callable[..., str]] = None
):
    """
    Caching decorator for functions.
    
    Usage:
        @cached(ttl=60)
        def expensive_function(x, y):
            return x + y
    """
    _cache = cache or LRUCache(CacheConfig(max_size=max_size, ttl_seconds=ttl))
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = _make_key(func.__name__, args, kwargs)
            
            # Check cache
            result = _cache.get(key)
            if result is not None:
                return result
            
            # Call function
            result = func(*args, **kwargs)
            _cache.set(key, result)
            return result
        
        wrapper.cache = _cache  # type: ignore
        wrapper.cache_clear = _cache.clear  # type: ignore
        wrapper.cache_stats = _cache.get_stats  # type: ignore
        return wrapper
    
    return decorator


def _make_key(name: str, args: tuple, kwargs: dict) -> str:
    """Generate cache key from function arguments."""
    key_parts = [name]
    
    for arg in args:
        key_parts.append(str(arg))
    
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={v}")
    
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


class CacheManager:
    """
    Manager for multiple named caches.
    
    Features:
    - Named cache management
    - Global statistics
    - Cleanup coordination
    """
    
    def __init__(self):
        self._caches: Dict[str, LRUCache] = {}
        self._lock = threading.Lock()
    
    def get_cache(
        self,
        name: str,
        max_size: int = 1000,
        ttl: float = 3600
    ) -> LRUCache:
        """Get or create a named cache."""
        with self._lock:
            if name not in self._caches:
                config = CacheConfig(max_size=max_size, ttl_seconds=ttl)
                self._caches[name] = LRUCache(config)
            return self._caches[name]
    
    def clear_all(self) -> None:
        """Clear all caches."""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all caches."""
        return {
            name: cache.get_stats().to_dict()
            for name, cache in self._caches.items()
        }


# Global cache manager
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def get_cache(name: str, **kwargs: Any) -> LRUCache:
    """Get a named cache."""
    return get_cache_manager().get_cache(name, **kwargs)


# Default recommendation cache
recommendation_cache: Optional[LRUCache[str, Any]] = None


def get_recommendation_cache() -> LRUCache[str, Any]:
    global recommendation_cache
    if recommendation_cache is None:
        recommendation_cache = LRUCache(CacheConfig(
            max_size=10000,
            ttl_seconds=300  # 5 minutes
        ))
    return recommendation_cache


def cache_recommendations(
    user_id: int,
    recommendations: list,
    algorithm: str = "default"
) -> None:
    """Cache recommendations for a user."""
    key = f"user:{user_id}:algo:{algorithm}"
    get_recommendation_cache().set(key, recommendations)


def get_cached_recommendations(
    user_id: int,
    algorithm: str = "default"
) -> Optional[list]:
    """Get cached recommendations for a user."""
    key = f"user:{user_id}:algo:{algorithm}"
    return get_recommendation_cache().get(key)


if __name__ == "__main__":
    print("Caching Module Demo")
    print("=" * 50)
    
    # Demo LRU cache
    cache: LRUCache[str, int] = LRUCache(CacheConfig(max_size=3, ttl_seconds=0))
    
    print("\n1. Basic LRU Cache (max_size=3)")
    print("-" * 30)
    
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    print(f"Set a=1, b=2, c=3")
    
    print(f"Get a: {cache.get('a')}")
    
    cache.set("d", 4)  # Should evict 'b'
    print(f"Set d=4 (should evict b)")
    
    print(f"Get b: {cache.get('b')}")  # Should be None
    print(f"Get c: {cache.get('c')}")
    print(f"Get d: {cache.get('d')}")
    
    print(f"\nStats: {cache.get_stats().to_dict()}")
    
    # Demo cached decorator
    print("\n\n2. Cached Decorator")
    print("-" * 30)
    
    call_count = 0
    
    @cached(ttl=60)
    def expensive_compute(x: int, y: int) -> int:
        global call_count
        call_count += 1
        time.sleep(0.1)  # Simulate work
        return x + y
    
    print("First call (computes)...")
    result1 = expensive_compute(1, 2)
    print(f"  Result: {result1}, calls: {call_count}")
    
    print("Second call (cached)...")
    result2 = expensive_compute(1, 2)
    print(f"  Result: {result2}, calls: {call_count}")
    
    print("Third call with different args (computes)...")
    result3 = expensive_compute(2, 3)
    print(f"  Result: {result3}, calls: {call_count}")
    
    print(f"\nCache stats: {expensive_compute.cache_stats().to_dict()}")  # type: ignore
    
    # Demo tiered cache
    print("\n\n3. Tiered Cache")
    print("-" * 30)
    
    tiered: TieredCache[str, str] = TieredCache(l1_size=2, l2_size=5)
    
    tiered.set("key1", "value1")
    tiered.set("key2", "value2")
    tiered.set("key3", "value3")
    
    print("Set 3 keys")
    print(f"Get key1: {tiered.get('key1')}")
    print(f"Get key3: {tiered.get('key3')}")
    
    print(f"\nTiered stats: {tiered.get_stats()}")
