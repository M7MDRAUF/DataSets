"""
CineMatch V2.1.6 - Caching Layer

Multi-level caching system with support for various backends,
automatic invalidation, and cache-aside pattern.

Author: CineMatch Development Team
"""

from abc import ABC, abstractmethod
from typing import (
    Dict, Any, Optional, TypeVar, Generic, Callable, List,
    Union, Tuple, Type
)
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import hashlib
import json
import pickle
import logging
import time
from functools import wraps
from collections import OrderedDict
import asyncio


logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# Cache Configuration
# =============================================================================

class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"      # Least Recently Used
    LFU = "lfu"      # Least Frequently Used
    FIFO = "fifo"    # First In First Out
    TTL = "ttl"      # Time To Live only


@dataclass
class CacheConfig:
    """Cache configuration"""
    max_size: int = 1000
    default_ttl: Optional[timedelta] = timedelta(minutes=30)
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    enable_stats: bool = True
    serialize: bool = False  # Whether to serialize values
    namespace: str = "default"


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    memory_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'size': self.size,
            'hit_rate': f"{self.hit_rate:.2%}",
            'memory_bytes': self.memory_bytes
        }


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata"""
    key: str
    value: T
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    
    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def touch(self) -> None:
        """Update access metadata"""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


# =============================================================================
# Cache Interface
# =============================================================================

class ICache(ABC, Generic[T]):
    """Abstract cache interface"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """Get value by key"""
        pass
    
    @abstractmethod
    def set(
        self,
        key: str,
        value: T,
        ttl: Optional[timedelta] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """Set value with optional TTL and tags"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all entries"""
        pass
    
    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        pass


# =============================================================================
# In-Memory Cache Implementation
# =============================================================================

class InMemoryCache(ICache[T]):
    """
    Thread-safe in-memory cache with LRU/LFU eviction.
    
    Supports TTL, tags for bulk invalidation, and statistics.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._entries: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._tags: Dict[str, set] = {}  # tag -> set of keys
        self._stats = CacheStats()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[T]:
        with self._lock:
            full_key = self._make_key(key)
            entry = self._entries.get(full_key)
            
            if entry is None:
                self._stats.misses += 1
                return None
            
            # Check expiration
            if entry.is_expired:
                self._remove_entry(full_key)
                self._stats.misses += 1
                return None
            
            # Update access metadata
            entry.touch()
            
            # Move to end for LRU
            if self.config.eviction_policy == EvictionPolicy.LRU:
                self._entries.move_to_end(full_key)
            
            self._stats.hits += 1
            return entry.value
    
    def set(
        self,
        key: str,
        value: T,
        ttl: Optional[timedelta] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        with self._lock:
            full_key = self._make_key(key)
            
            # Calculate expiration
            expires_at = None
            effective_ttl = ttl or self.config.default_ttl
            if effective_ttl:
                expires_at = datetime.utcnow() + effective_ttl
            
            # Create entry
            entry = CacheEntry(
                key=full_key,
                value=value,
                expires_at=expires_at,
                tags=tags or []
            )
            
            # Remove old entry if exists
            if full_key in self._entries:
                self._remove_entry(full_key)
            
            # Check capacity and evict if needed
            while len(self._entries) >= self.config.max_size:
                self._evict_one()
            
            # Add entry
            self._entries[full_key] = entry
            
            # Update tag index
            for tag in entry.tags:
                if tag not in self._tags:
                    self._tags[tag] = set()
                self._tags[tag].add(full_key)
            
            self._update_size_stats()
    
    def delete(self, key: str) -> bool:
        with self._lock:
            full_key = self._make_key(key)
            if full_key in self._entries:
                self._remove_entry(full_key)
                return True
            return False
    
    def exists(self, key: str) -> bool:
        with self._lock:
            full_key = self._make_key(key)
            entry = self._entries.get(full_key)
            
            if entry is None:
                return False
            
            if entry.is_expired:
                self._remove_entry(full_key)
                return False
            
            return True
    
    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._tags.clear()
            self._stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        with self._lock:
            self._update_size_stats()
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size=len(self._entries),
                memory_bytes=self._stats.memory_bytes
            )
    
    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with given tag"""
        with self._lock:
            keys = self._tags.get(tag, set()).copy()
            for key in keys:
                self._remove_entry(key)
            return len(keys)
    
    def invalidate_by_prefix(self, prefix: str) -> int:
        """Invalidate all entries with key prefix"""
        with self._lock:
            full_prefix = self._make_key(prefix)
            keys = [k for k in self._entries.keys() if k.startswith(full_prefix)]
            for key in keys:
                self._remove_entry(key)
            return len(keys)
    
    def get_or_set(
        self,
        key: str,
        factory: Callable[[], T],
        ttl: Optional[timedelta] = None,
        tags: Optional[List[str]] = None
    ) -> T:
        """Get value or compute and cache if not exists"""
        value = self.get(key)
        if value is not None:
            return value
        
        value = factory()
        self.set(key, value, ttl, tags)
        return value
    
    def _make_key(self, key: str) -> str:
        """Create namespaced key"""
        return f"{self.config.namespace}:{key}"
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry and update indexes"""
        entry = self._entries.pop(key, None)
        if entry:
            for tag in entry.tags:
                if tag in self._tags:
                    self._tags[tag].discard(key)
    
    def _evict_one(self) -> None:
        """Evict one entry based on policy"""
        if not self._entries:
            return
        
        if self.config.eviction_policy == EvictionPolicy.LRU:
            # Remove first item (oldest)
            key = next(iter(self._entries))
            self._remove_entry(key)
        
        elif self.config.eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used
            min_count = float('inf')
            min_key = None
            for key, entry in self._entries.items():
                if entry.access_count < min_count:
                    min_count = entry.access_count
                    min_key = key
            if min_key:
                self._remove_entry(min_key)
        
        elif self.config.eviction_policy == EvictionPolicy.FIFO:
            # Remove first item
            key = next(iter(self._entries))
            self._remove_entry(key)
        
        elif self.config.eviction_policy == EvictionPolicy.TTL:
            # Remove expired items first, then oldest
            for key, entry in list(self._entries.items()):
                if entry.is_expired:
                    self._remove_entry(key)
                    break
            else:
                # No expired, remove oldest
                key = next(iter(self._entries))
                self._remove_entry(key)
        
        self._stats.evictions += 1
    
    def _update_size_stats(self) -> None:
        """Update memory usage statistics"""
        # Approximate memory calculation
        try:
            import sys
            total_size = 0
            for entry in self._entries.values():
                total_size += sys.getsizeof(entry.value)
            self._stats.memory_bytes = total_size
        except Exception:
            pass
        
        self._stats.size = len(self._entries)


# =============================================================================
# Multi-Level Cache
# =============================================================================

class MultiLevelCache(ICache[T]):
    """
    Multi-level cache with L1 (fast/small) and L2 (slow/large).
    
    Automatically promotes items from L2 to L1 on access.
    """
    
    def __init__(
        self,
        l1_cache: ICache[T],
        l2_cache: ICache[T],
        write_through: bool = True
    ):
        self._l1 = l1_cache
        self._l2 = l2_cache
        self._write_through = write_through
        self._stats = CacheStats()
    
    def get(self, key: str) -> Optional[T]:
        # Try L1 first
        value = self._l1.get(key)
        if value is not None:
            self._stats.hits += 1
            return value
        
        # Try L2
        value = self._l2.get(key)
        if value is not None:
            # Promote to L1
            self._l1.set(key, value)
            self._stats.hits += 1
            return value
        
        self._stats.misses += 1
        return None
    
    def set(
        self,
        key: str,
        value: T,
        ttl: Optional[timedelta] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        # Always write to L1
        self._l1.set(key, value, ttl, tags)
        
        # Write-through to L2
        if self._write_through:
            self._l2.set(key, value, ttl, tags)
    
    def delete(self, key: str) -> bool:
        l1_deleted = self._l1.delete(key)
        l2_deleted = self._l2.delete(key)
        return l1_deleted or l2_deleted
    
    def exists(self, key: str) -> bool:
        return self._l1.exists(key) or self._l2.exists(key)
    
    def clear(self) -> None:
        self._l1.clear()
        self._l2.clear()
        self._stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        l1_stats = self._l1.get_stats()
        l2_stats = self._l2.get_stats()
        
        return CacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            evictions=l1_stats.evictions + l2_stats.evictions,
            size=l1_stats.size + l2_stats.size,
            memory_bytes=l1_stats.memory_bytes + l2_stats.memory_bytes
        )


# =============================================================================
# Cache Decorators
# =============================================================================

def cached(
    cache: ICache,
    key_func: Optional[Callable[..., str]] = None,
    ttl: Optional[timedelta] = None,
    tags: Optional[List[str]] = None
):
    """
    Decorator to cache function results.
    
    Usage:
        @cached(my_cache, ttl=timedelta(minutes=5))
        def expensive_function(user_id: int):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = _generate_key(func.__name__, args, kwargs)
            
            # Try cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            cache.set(key, result, ttl, tags)
            
            return result
        
        # Add cache control methods
        wrapper.cache_clear = lambda: cache.invalidate_by_prefix(func.__name__)
        wrapper.cache_info = lambda: cache.get_stats()
        
        return wrapper
    return decorator


def async_cached(
    cache: ICache,
    key_func: Optional[Callable[..., str]] = None,
    ttl: Optional[timedelta] = None,
    tags: Optional[List[str]] = None
):
    """
    Decorator to cache async function results.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = _generate_key(func.__name__, args, kwargs)
            
            # Try cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Call function
            result = await func(*args, **kwargs)
            
            # Cache result
            cache.set(key, result, ttl, tags)
            
            return result
        
        return wrapper
    return decorator


def _generate_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate cache key from function signature"""
    # Create hashable representation
    key_parts = [func_name]
    
    for arg in args:
        key_parts.append(_hash_value(arg))
    
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}={_hash_value(v)}")
    
    return ":".join(key_parts)


def _hash_value(value: Any) -> str:
    """Create hash of value for cache key"""
    try:
        # Try JSON serialization
        serialized = json.dumps(value, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode()).hexdigest()[:8]
    except (TypeError, ValueError):
        # Fallback to string representation
        return hashlib.md5(str(value).encode()).hexdigest()[:8]


# =============================================================================
# Cache Manager
# =============================================================================

class CacheManager:
    """
    Centralized cache management.
    
    Provides named caches and global operations.
    """
    
    def __init__(self):
        self._caches: Dict[str, ICache] = {}
        self._lock = threading.RLock()
    
    def get_cache(
        self,
        name: str,
        config: Optional[CacheConfig] = None
    ) -> ICache:
        """Get or create named cache"""
        with self._lock:
            if name not in self._caches:
                cache_config = config or CacheConfig(namespace=name)
                self._caches[name] = InMemoryCache(cache_config)
            return self._caches[name]
    
    def register_cache(self, name: str, cache: ICache) -> None:
        """Register existing cache instance"""
        with self._lock:
            self._caches[name] = cache
    
    def remove_cache(self, name: str) -> bool:
        """Remove named cache"""
        with self._lock:
            if name in self._caches:
                del self._caches[name]
                return True
            return False
    
    def clear_all(self) -> None:
        """Clear all caches"""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
    
    def get_all_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all caches"""
        with self._lock:
            return {
                name: cache.get_stats()
                for name, cache in self._caches.items()
            }
    
    def list_caches(self) -> List[str]:
        """List all cache names"""
        with self._lock:
            return list(self._caches.keys())


# Global cache manager
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


# =============================================================================
# Recommendation-Specific Caches
# =============================================================================

class RecommendationCache:
    """
    Specialized cache for recommendation system.
    
    Provides domain-specific caching methods with appropriate
    TTLs and invalidation strategies.
    """
    
    def __init__(self, manager: Optional[CacheManager] = None):
        self._manager = manager or get_cache_manager()
        
        # Create specialized caches
        self._recommendations = self._manager.get_cache(
            "recommendations",
            CacheConfig(max_size=10000, default_ttl=timedelta(minutes=15))
        )
        self._predictions = self._manager.get_cache(
            "predictions",
            CacheConfig(max_size=50000, default_ttl=timedelta(hours=1))
        )
        self._user_profiles = self._manager.get_cache(
            "user_profiles",
            CacheConfig(max_size=5000, default_ttl=timedelta(hours=4))
        )
        self._movie_data = self._manager.get_cache(
            "movie_data",
            CacheConfig(max_size=100000, default_ttl=timedelta(hours=24))
        )
        self._search_results = self._manager.get_cache(
            "search_results",
            CacheConfig(max_size=5000, default_ttl=timedelta(minutes=30))
        )
    
    def get_recommendations(
        self,
        user_id: int,
        algorithm: str,
        n: int
    ) -> Optional[List[Dict]]:
        """Get cached recommendations"""
        key = f"{user_id}:{algorithm}:{n}"
        return self._recommendations.get(key)
    
    def set_recommendations(
        self,
        user_id: int,
        algorithm: str,
        n: int,
        recommendations: List[Dict]
    ) -> None:
        """Cache recommendations"""
        key = f"{user_id}:{algorithm}:{n}"
        self._recommendations.set(key, recommendations, tags=[f"user:{user_id}"])
    
    def get_prediction(
        self,
        user_id: int,
        movie_id: int,
        algorithm: str
    ) -> Optional[float]:
        """Get cached prediction"""
        key = f"{user_id}:{movie_id}:{algorithm}"
        return self._predictions.get(key)
    
    def set_prediction(
        self,
        user_id: int,
        movie_id: int,
        algorithm: str,
        rating: float
    ) -> None:
        """Cache prediction"""
        key = f"{user_id}:{movie_id}:{algorithm}"
        self._predictions.set(
            key, rating,
            tags=[f"user:{user_id}", f"movie:{movie_id}"]
        )
    
    def get_user_profile(self, user_id: int) -> Optional[Dict]:
        """Get cached user profile"""
        return self._user_profiles.get(str(user_id))
    
    def set_user_profile(self, user_id: int, profile: Dict) -> None:
        """Cache user profile"""
        self._user_profiles.set(
            str(user_id), profile,
            tags=[f"user:{user_id}"]
        )
    
    def get_movie(self, movie_id: int) -> Optional[Dict]:
        """Get cached movie data"""
        return self._movie_data.get(str(movie_id))
    
    def set_movie(self, movie_id: int, movie: Dict) -> None:
        """Cache movie data"""
        self._movie_data.set(str(movie_id), movie)
    
    def get_search_results(self, query: str, filters: Dict) -> Optional[List]:
        """Get cached search results"""
        key = f"{query}:{_hash_value(filters)}"
        return self._search_results.get(key)
    
    def set_search_results(
        self,
        query: str,
        filters: Dict,
        results: List
    ) -> None:
        """Cache search results"""
        key = f"{query}:{_hash_value(filters)}"
        self._search_results.set(key, results)
    
    def invalidate_user(self, user_id: int) -> None:
        """Invalidate all caches for user"""
        tag = f"user:{user_id}"
        if hasattr(self._recommendations, 'invalidate_by_tag'):
            self._recommendations.invalidate_by_tag(tag)
        if hasattr(self._predictions, 'invalidate_by_tag'):
            self._predictions.invalidate_by_tag(tag)
        if hasattr(self._user_profiles, 'invalidate_by_tag'):
            self._user_profiles.invalidate_by_tag(tag)
    
    def invalidate_movie(self, movie_id: int) -> None:
        """Invalidate caches for movie"""
        tag = f"movie:{movie_id}"
        if hasattr(self._predictions, 'invalidate_by_tag'):
            self._predictions.invalidate_by_tag(tag)
        self._movie_data.delete(str(movie_id))
    
    def clear_all(self) -> None:
        """Clear all recommendation caches"""
        self._recommendations.clear()
        self._predictions.clear()
        self._user_profiles.clear()
        self._movie_data.clear()
        self._search_results.clear()
    
    def get_stats(self) -> Dict[str, CacheStats]:
        """Get all cache statistics"""
        return {
            'recommendations': self._recommendations.get_stats(),
            'predictions': self._predictions.get_stats(),
            'user_profiles': self._user_profiles.get_stats(),
            'movie_data': self._movie_data.get_stats(),
            'search_results': self._search_results.get_stats()
        }
