"""
LRU Cache Implementation with TTL Support.

This module provides a thread-safe LRU (Least Recently Used) cache with
configurable maximum size and time-to-live (TTL) for entries.

Used for caching user profiles, similarity computations, and other
expensive operations that benefit from bounded memory usage.
"""

import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Generic, Optional, TypeVar
from dataclasses import dataclass

K = TypeVar('K')
V = TypeVar('V')


@dataclass
class CacheEntry(Generic[V]):
    """Entry in the LRU cache with value and timestamp."""
    value: V
    timestamp: float
    
    def is_expired(self, ttl_seconds: Optional[float]) -> bool:
        """Check if entry has expired based on TTL."""
        if ttl_seconds is None:
            return False
        return (time.time() - self.timestamp) > ttl_seconds


class LRUCache(Generic[K, V]):
    """
    Thread-safe LRU Cache with configurable max size and TTL.
    
    Features:
    - O(1) get and set operations
    - Automatic eviction of least recently used items
    - Optional TTL-based expiration
    - Thread-safe operations
    - Memory-bounded growth
    
    Example:
        >>> cache = LRUCache[int, np.ndarray](max_size=10000, ttl_seconds=3600)
        >>> cache.set(user_id, profile_vector)
        >>> profile = cache.get(user_id)
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: Optional[float] = None,
        on_evict: Optional[Callable[[K, V], None]] = None
    ):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries to store. When exceeded,
                      least recently used entries are evicted.
            ttl_seconds: Time-to-live in seconds for each entry. None means
                        entries never expire based on time.
            on_evict: Optional callback when an entry is evicted.
                      Called with (key, value) arguments.
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._on_evict = on_evict
        self._cache: OrderedDict[K, CacheEntry[V]] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    @property
    def max_size(self) -> int:
        """Maximum cache size."""
        return self._max_size
    
    @property
    def ttl_seconds(self) -> Optional[float]:
        """TTL in seconds for cache entries."""
        return self._ttl_seconds
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        Get value from cache.
        
        Moves the accessed entry to the end (most recently used).
        Returns None if key not found or entry has expired.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return default
            
            entry = self._cache[key]
            
            # Check TTL expiration
            if entry.is_expired(self._ttl_seconds):
                self._remove(key)
                self._misses += 1
                return default
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value
    
    def set(self, key: K, value: V) -> None:
        """
        Set value in cache.
        
        If key exists, updates value and moves to end.
        If cache is full, evicts least recently used entry.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            current_time = time.time()
            
            if key in self._cache:
                # Update existing entry
                self._cache[key] = CacheEntry(value=value, timestamp=current_time)
                self._cache.move_to_end(key)
            else:
                # Evict if at capacity
                while len(self._cache) >= self._max_size:
                    self._evict_oldest()
                
                # Add new entry
                self._cache[key] = CacheEntry(value=value, timestamp=current_time)
    
    def __contains__(self, key: K) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            if key not in self._cache:
                return False
            entry = self._cache[key]
            if entry.is_expired(self._ttl_seconds):
                self._remove(key)
                return False
            return True
    
    def __getitem__(self, key: K) -> V:
        """Get item with dict-like access. Raises KeyError if not found."""
        value = self.get(key)
        if value is None and key not in self._cache:
            raise KeyError(key)
        return value
    
    def __setitem__(self, key: K, value: V) -> None:
        """Set item with dict-like access."""
        self.set(key, value)
    
    def __len__(self) -> int:
        """Return number of non-expired entries."""
        with self._lock:
            self._cleanup_expired()
            return len(self._cache)
    
    def _remove(self, key: K) -> None:
        """Remove entry by key (internal, assumes lock held)."""
        if key in self._cache:
            entry = self._cache.pop(key)
            if self._on_evict:
                try:
                    self._on_evict(key, entry.value)
                except Exception:
                    pass  # Don't let callback errors affect cache operation
    
    def _evict_oldest(self) -> None:
        """Evict the least recently used entry (internal, assumes lock held)."""
        if self._cache:
            # popitem(last=False) removes the oldest entry
            key, entry = self._cache.popitem(last=False)
            self._evictions += 1
            if self._on_evict:
                try:
                    self._on_evict(key, entry.value)
                except Exception:
                    pass
    
    def _cleanup_expired(self) -> None:
        """Remove all expired entries (internal, assumes lock held)."""
        if self._ttl_seconds is None:
            return
        
        expired_keys = [
            k for k, v in self._cache.items()
            if v.is_expired(self._ttl_seconds)
        ]
        for key in expired_keys:
            self._remove(key)
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            if self._on_evict:
                for key, entry in self._cache.items():
                    try:
                        self._on_evict(key, entry.value)
                    except Exception:
                        pass
            self._cache.clear()
    
    def keys(self) -> list[K]:
        """Return list of all non-expired keys."""
        with self._lock:
            self._cleanup_expired()
            return list(self._cache.keys())
    
    def values(self) -> list[V]:
        """Return list of all non-expired values."""
        with self._lock:
            self._cleanup_expired()
            return [entry.value for entry in self._cache.values()]
    
    def items(self) -> list[tuple[K, V]]:
        """Return list of all non-expired (key, value) pairs."""
        with self._lock:
            self._cleanup_expired()
            return [(k, entry.value) for k, entry in self._cache.items()]
    
    def stats(self) -> dict[str, Any]:
        """
        Return cache statistics.
        
        Returns:
            Dict with hits, misses, hit_ratio, size, evictions
        """
        with self._lock:
            total = self._hits + self._misses
            hit_ratio = self._hits / total if total > 0 else 0.0
            return {
                'hits': self._hits,
                'misses': self._misses,
                'hit_ratio': hit_ratio,
                'size': len(self._cache),
                'max_size': self._max_size,
                'evictions': self._evictions,
                'ttl_seconds': self._ttl_seconds
            }
    
    def reset_stats(self) -> None:
        """Reset hit/miss/eviction statistics."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
    
    def __getstate__(self):
        """
        Prepare cache for pickling.
        
        Returns cache data as a dict that can be pickled.
        Excludes threading.RLock which cannot be pickled.
        The callback function _on_evict is also excluded as it may not be picklable.
        
        Returns:
            Dict with cache configuration and current entries
        """
        with self._lock:
            # Extract cache entries as list of (key, value, timestamp) tuples
            cache_items = [(k, entry.value, entry.timestamp) 
                          for k, entry in self._cache.items()]
            
            return {
                '_max_size': self._max_size,
                '_ttl_seconds': self._ttl_seconds,
                '_cache_items': cache_items,
                '_hits': self._hits,
                '_misses': self._misses,
                '_evictions': self._evictions
            }
    
    def __setstate__(self, state):
        """
        Restore cache from pickled state.
        
        Recreates the cache structure from the serialized dict.
        The lock and OrderedDict are reconstructed, and cache entries are restored.
        
        Args:
            state: Dict containing cache configuration and entries
        """
        # Restore basic attributes
        self._max_size = state['_max_size']
        self._ttl_seconds = state.get('_ttl_seconds')
        self._hits = state.get('_hits', 0)
        self._misses = state.get('_misses', 0)
        self._evictions = state.get('_evictions', 0)
        
        # Recreate unpicklable objects
        self._on_evict = None  # Callbacks cannot be reliably pickled
        self._lock = threading.RLock()  # Locks must be recreated
        self._cache = OrderedDict()
        
        # Restore cache entries
        cache_items = state.get('_cache_items', [])
        for key, value, timestamp in cache_items:
            self._cache[key] = CacheEntry(value=value, timestamp=timestamp)


class LRUCacheWithLoader(LRUCache[K, V]):
    """
    LRU Cache with automatic loading of missing values.
    
    Extends LRUCache with a loader function that's called when a key
    is not found in the cache.
    
    Example:
        >>> def build_profile(user_id):
        ...     return compute_expensive_profile(user_id)
        >>> cache = LRUCacheWithLoader(max_size=10000, loader=build_profile)
        >>> profile = cache.get_or_load(user_id)  # Loads if not cached
    """
    
    def __init__(
        self,
        loader: Callable[[K], Optional[V]],
        max_size: int = 10000,
        ttl_seconds: Optional[float] = None,
        on_evict: Optional[Callable[[K, V], None]] = None
    ):
        """
        Initialize LRU cache with loader.
        
        Args:
            loader: Function to call when key is not in cache.
                   Takes key as argument, returns value or None.
            max_size: Maximum number of entries to store.
            ttl_seconds: Time-to-live in seconds for each entry.
            on_evict: Optional callback when an entry is evicted.
        """
        super().__init__(max_size=max_size, ttl_seconds=ttl_seconds, on_evict=on_evict)
        self._loader = loader
    
    def get_or_load(self, key: K) -> Optional[V]:
        """
        Get value from cache, loading if necessary.
        
        If key is not in cache or has expired, calls the loader function
        to compute the value and stores it in the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached or newly loaded value
        """
        # Try to get from cache first (outside lock for performance)
        value = self.get(key)
        if value is not None:
            return value
        
        # Not in cache, need to load
        with self._lock:
            # Double-check after acquiring lock
            if key in self._cache:
                entry = self._cache[key]
                if not entry.is_expired(self._ttl_seconds):
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return entry.value
            
            # Load value
            value = self._loader(key)
            
            # Only cache non-None values
            if value is not None:
                # Evict if at capacity
                while len(self._cache) >= self._max_size:
                    self._evict_oldest()
                
                self._cache[key] = CacheEntry(value=value, timestamp=time.time())
            
            return value
