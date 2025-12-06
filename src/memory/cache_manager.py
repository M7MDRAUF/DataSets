"""
Cache Manager Module for CineMatch V2.1.6

Implements cache management with memory awareness:
- LRU/LFU caching
- Memory-bounded caches
- Cache eviction policies
- Cache statistics

Phase 6 - Task 6.4: Cache Manager
"""

import logging
import time
import threading
import sys
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar
from dataclasses import dataclass, field
from collections import OrderedDict
from datetime import datetime

logger = logging.getLogger(__name__)

K = TypeVar('K')
V = TypeVar('V')


class EvictionPolicy:
    """Cache eviction policy."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live


@dataclass
class CacheConfig:
    """Cache configuration."""
    max_size: int = 1000
    max_memory_mb: float = 100.0
    default_ttl: Optional[float] = None
    eviction_policy: str = EvictionPolicy.LRU
    eviction_batch_size: int = 10


@dataclass
class CacheEntry(Generic[V]):
    """Cache entry with metadata."""
    value: V
    size_bytes: int
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    access_count: int = 0
    
    def touch(self) -> None:
        self.last_accessed = time.time()
        self.access_count += 1
    
    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    @property
    def age(self) -> float:
        return time.time() - self.created_at


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    
    @property
    def total_requests(self) -> int:
        return self.hits + self.misses
    
    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'expirations': self.expirations,
            'hit_rate': round(self.hit_rate * 100, 2)
        }


class MemoryAwareCache(Generic[K, V]):
    """
    Memory-aware cache with multiple eviction policies.
    
    Features:
    - LRU/LFU/FIFO eviction
    - Memory limits
    - TTL support
    - Statistics tracking
    """
    
    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        name: str = "cache"
    ):
        self.config = config or CacheConfig()
        self.name = name
        self._cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
        self._current_memory_bytes = 0
    
    @property
    def size(self) -> int:
        """Number of items in cache."""
        return len(self._cache)
    
    @property
    def memory_mb(self) -> float:
        """Current memory usage in MB."""
        return self._current_memory_bytes / (1024 * 1024)
    
    def _estimate_size(self, value: V) -> int:
        """Estimate memory size of value."""
        try:
            return sys.getsizeof(value)
        except TypeError:
            return 0
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Get value from cache."""
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats.misses += 1
                return default
            
            if entry.is_expired:
                self._remove_entry(key)
                self._stats.expirations += 1
                self._stats.misses += 1
                return default
            
            entry.touch()
            self._stats.hits += 1
            
            # Move to end for LRU
            if self.config.eviction_policy == EvictionPolicy.LRU:
                self._cache.move_to_end(key)
            
            return entry.value
    
    def set(
        self,
        key: K,
        value: V,
        ttl: Optional[float] = None
    ) -> bool:
        """Set value in cache."""
        with self._lock:
            size_bytes = self._estimate_size(value)
            
            # Check if update
            if key in self._cache:
                old_entry = self._cache[key]
                self._current_memory_bytes -= old_entry.size_bytes
            
            # Create entry
            entry = CacheEntry(
                value=value,
                size_bytes=size_bytes,
                expires_at=time.time() + ttl if ttl else (
                    time.time() + self.config.default_ttl
                    if self.config.default_ttl else None
                )
            )
            
            # Evict if needed
            while self._should_evict(size_bytes):
                if not self._evict_one():
                    break
            
            self._cache[key] = entry
            self._current_memory_bytes += size_bytes
            
            return True
    
    def _should_evict(self, incoming_bytes: int = 0) -> bool:
        """Check if eviction is needed."""
        if self.size >= self.config.max_size:
            return True
        
        total_bytes = self._current_memory_bytes + incoming_bytes
        if total_bytes > self.config.max_memory_mb * 1024 * 1024:
            return True
        
        return False
    
    def _evict_one(self) -> bool:
        """Evict one entry based on policy."""
        if not self._cache:
            return False
        
        key_to_evict = None
        
        if self.config.eviction_policy == EvictionPolicy.LRU:
            # First item is least recently used
            key_to_evict = next(iter(self._cache))
        
        elif self.config.eviction_policy == EvictionPolicy.LFU:
            # Find least frequently used
            min_count = float('inf')
            for k, entry in self._cache.items():
                if entry.access_count < min_count:
                    min_count = entry.access_count
                    key_to_evict = k
        
        elif self.config.eviction_policy == EvictionPolicy.FIFO:
            # First item is oldest
            key_to_evict = next(iter(self._cache))
        
        elif self.config.eviction_policy == EvictionPolicy.TTL:
            # Find closest to expiration
            earliest = float('inf')
            for k, entry in self._cache.items():
                if entry.expires_at and entry.expires_at < earliest:
                    earliest = entry.expires_at
                    key_to_evict = k
            
            if key_to_evict is None:
                key_to_evict = next(iter(self._cache))
        
        if key_to_evict:
            self._remove_entry(key_to_evict)
            self._stats.evictions += 1
            return True
        
        return False
    
    def _remove_entry(self, key: K) -> None:
        """Remove entry from cache."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._current_memory_bytes -= entry.size_bytes
    
    def delete(self, key: K) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def exists(self, key: K) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired:
                self._remove_entry(key)
                return False
            return True
    
    def clear(self) -> int:
        """Clear all entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._current_memory_bytes = 0
            return count
    
    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        with self._lock:
            expired_keys = [
                k for k, v in self._cache.items() if v.is_expired
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
                self._stats.expirations += 1
            
            return len(expired_keys)
    
    def get_or_set(
        self,
        key: K,
        factory: Callable[[], V],
        ttl: Optional[float] = None
    ) -> V:
        """Get from cache or compute and store."""
        value = self.get(key)
        if value is not None:
            return value
        
        value = factory()
        self.set(key, value, ttl)
        return value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                **self._stats.to_dict(),
                'name': self.name,
                'size': self.size,
                'memory_mb': round(self.memory_mb, 2),
                'max_size': self.config.max_size,
                'max_memory_mb': self.config.max_memory_mb,
                'policy': self.config.eviction_policy
            }
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        with self._lock:
            self._stats = CacheStats()


class CacheManager:
    """
    Manages multiple caches with memory limits.
    
    Features:
    - Multiple named caches
    - Global memory limits
    - Cross-cache eviction
    """
    
    def __init__(self, global_memory_limit_mb: float = 500.0):
        self.global_limit_mb = global_memory_limit_mb
        self._caches: Dict[str, MemoryAwareCache] = {}
        self._lock = threading.RLock()
    
    def create_cache(
        self,
        name: str,
        config: Optional[CacheConfig] = None
    ) -> MemoryAwareCache:
        """Create a new cache."""
        with self._lock:
            cache = MemoryAwareCache(config, name)
            self._caches[name] = cache
            return cache
    
    def get_cache(self, name: str) -> Optional[MemoryAwareCache]:
        """Get cache by name."""
        return self._caches.get(name)
    
    def delete_cache(self, name: str) -> bool:
        """Delete a cache."""
        with self._lock:
            if name in self._caches:
                self._caches[name].clear()
                del self._caches[name]
                return True
            return False
    
    def get_total_memory_mb(self) -> float:
        """Get total memory used by all caches."""
        return sum(c.memory_mb for c in self._caches.values())
    
    def cleanup_all(self) -> Dict[str, int]:
        """Cleanup expired entries in all caches."""
        return {
            name: cache.cleanup_expired()
            for name, cache in self._caches.items()
        }
    
    def evict_to_limit(self) -> int:
        """Evict entries until under global limit."""
        evicted = 0
        
        while self.get_total_memory_mb() > self.global_limit_mb:
            # Find largest cache
            largest = max(
                self._caches.values(),
                key=lambda c: c.memory_mb,
                default=None
            )
            
            if largest and largest._evict_one():
                evicted += 1
            else:
                break
        
        return evicted
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics."""
        with self._lock:
            total_hits = sum(c._stats.hits for c in self._caches.values())
            total_misses = sum(c._stats.misses for c in self._caches.values())
            
            return {
                'cache_count': len(self._caches),
                'total_memory_mb': round(self.get_total_memory_mb(), 2),
                'global_limit_mb': self.global_limit_mb,
                'total_hits': total_hits,
                'total_misses': total_misses,
                'overall_hit_rate': round(
                    total_hits / (total_hits + total_misses) * 100
                    if (total_hits + total_misses) > 0 else 0, 2
                ),
                'caches': {
                    name: cache.get_stats()
                    for name, cache in self._caches.items()
                }
            }


# Global cache manager
_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager."""
    global _manager
    if _manager is None:
        _manager = CacheManager()
    return _manager


def get_cache(name: str) -> Optional[MemoryAwareCache]:
    """Get cache by name."""
    return get_cache_manager().get_cache(name)


def create_cache(
    name: str,
    config: Optional[CacheConfig] = None
) -> MemoryAwareCache:
    """Create a new cache."""
    return get_cache_manager().create_cache(name, config)


if __name__ == "__main__":
    print("Cache Manager Module Demo")
    print("=" * 50)
    
    # Create cache
    cache = MemoryAwareCache(
        config=CacheConfig(
            max_size=10,
            max_memory_mb=1.0,
            default_ttl=60.0,
            eviction_policy=EvictionPolicy.LRU
        ),
        name="demo_cache"
    )
    
    print("\n1. Basic Operations")
    print("-" * 30)
    
    cache.set("key1", "value1")
    cache.set("key2", {"data": [1, 2, 3]})
    cache.set("key3", list(range(100)))
    
    print(f"key1: {cache.get('key1')}")
    print(f"key2: {cache.get('key2')}")
    print(f"Cache size: {cache.size}")
    print(f"Memory: {cache.memory_mb:.4f} MB")
    
    print("\n\n2. TTL Expiration")
    print("-" * 30)
    
    cache.set("temp", "temporary", ttl=0.5)
    print(f"temp (immediate): {cache.get('temp')}")
    time.sleep(0.6)
    print(f"temp (after TTL): {cache.get('temp')}")
    
    print("\n\n3. LRU Eviction")
    print("-" * 30)
    
    # Fill cache
    for i in range(15):
        cache.set(f"item_{i}", f"value_{i}")
    
    print(f"Cache size: {cache.size}")
    print(f"Evictions: {cache._stats.evictions}")
    
    # Access some items to change LRU order
    cache.get("item_5")
    cache.get("item_6")
    
    # Add more items - should evict least recently used
    for i in range(15, 20):
        cache.set(f"item_{i}", f"value_{i}")
    
    print(f"item_5 exists: {cache.exists('item_5')}")
    print(f"item_0 exists: {cache.exists('item_0')}")
    
    print("\n\n4. Get or Set")
    print("-" * 30)
    
    compute_count = 0
    
    def expensive_compute():
        global compute_count
        compute_count += 1
        return f"computed_{compute_count}"
    
    result1 = cache.get_or_set("computed", expensive_compute)
    result2 = cache.get_or_set("computed", expensive_compute)
    
    print(f"First call: {result1}")
    print(f"Second call: {result2}")
    print(f"Compute count: {compute_count}")
    
    print("\n\n5. Statistics")
    print("-" * 30)
    
    stats = cache.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n\n6. Cache Manager")
    print("-" * 30)
    
    manager = CacheManager(global_memory_limit_mb=10.0)
    
    user_cache = manager.create_cache("users", CacheConfig(max_size=100))
    session_cache = manager.create_cache("sessions", CacheConfig(max_size=50))
    query_cache = manager.create_cache("queries", CacheConfig(max_size=200))
    
    # Add data
    for i in range(50):
        user_cache.set(f"user_{i}", {"id": i, "name": f"User {i}"})
        session_cache.set(f"session_{i}", {"user_id": i, "data": {}})
    
    manager_stats = manager.get_statistics()
    print(f"Total caches: {manager_stats['cache_count']}")
    print(f"Total memory: {manager_stats['total_memory_mb']} MB")
    
    print("\n\n7. Cleanup Expired")
    print("-" * 30)
    
    test_cache = manager.create_cache("test", CacheConfig(default_ttl=0.2))
    for i in range(10):
        test_cache.set(f"temp_{i}", f"value_{i}")
    
    print(f"Before cleanup: {test_cache.size}")
    time.sleep(0.3)
    cleaned = manager.cleanup_all()
    print(f"Cleaned: {cleaned}")
    print(f"After cleanup: {test_cache.size}")
    
    print("\n\n8. Different Eviction Policies")
    print("-" * 30)
    
    lfu_cache = MemoryAwareCache(
        config=CacheConfig(max_size=5, eviction_policy=EvictionPolicy.LFU),
        name="lfu_cache"
    )
    
    # Add items with different access frequencies
    lfu_cache.set("a", "A")
    lfu_cache.set("b", "B")
    lfu_cache.set("c", "C")
    
    # Access 'a' multiple times
    for _ in range(10):
        lfu_cache.get("a")
    
    # Access 'b' a few times
    for _ in range(3):
        lfu_cache.get("b")
    
    # Fill to trigger eviction
    lfu_cache.set("d", "D")
    lfu_cache.set("e", "E")
    lfu_cache.set("f", "F")
    
    print(f"'a' exists (most accessed): {lfu_cache.exists('a')}")
    print(f"'c' exists (least accessed): {lfu_cache.exists('c')}")
    
    print("\n\n✓ Cache manager demo complete")
