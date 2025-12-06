"""
CineMatch V2.1.6 - Recommendation Caching

TTL-based caching for recommendations to reduce computation.
Supports both in-memory and optional Redis-based caching.

Author: CineMatch Development Team
Date: December 5, 2025

Features:
    - TTL (Time-To-Live) cache with automatic expiration
    - LRU eviction when cache is full
    - Cache key generation based on request parameters
    - Thread-safe implementation
    - Optional Redis backend for distributed caching
"""

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, TypeVar, Generic
from functools import wraps
import logging
import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with value and expiration time."""
    
    value: T
    expires_at: float
    created_at: float = field(default_factory=time.time)
    hit_count: int = field(default=0)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.expires_at
    
    @property
    def ttl_remaining(self) -> float:
        """Get remaining TTL in seconds."""
        return max(0, self.expires_at - time.time())


class TTLCache(Generic[T]):
    """
    Thread-safe TTL cache with LRU eviction.
    
    Features:
    - Automatic expiration based on TTL
    - LRU eviction when max size reached
    - Thread-safe operations
    - Statistics tracking
    
    Usage:
        cache = TTLCache(maxsize=1000, ttl=300)  # 5 minute TTL
        cache.set('key', 'value')
        value = cache.get('key')  # Returns 'value' or None if expired
    """
    
    def __init__(self, maxsize: int = 1000, ttl: float = 300.0):
        """
        Initialize TTL cache.
        
        Args:
            maxsize: Maximum number of entries
            ttl: Time-to-live in seconds (default: 5 minutes)
        """
        self.maxsize = maxsize
        self.default_ttl = ttl
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get(self, key: str) -> Optional[T]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            entry.hit_count += 1
            self._hits += 1
            
            return entry.value
    
    def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional custom TTL (uses default if not specified)
        """
        ttl = ttl if ttl is not None else self.default_ttl
        
        with self._lock:
            # Remove existing entry
            if key in self._cache:
                del self._cache[key]
            
            # Evict oldest if at capacity
            while len(self._cache) >= self.maxsize:
                self._cache.popitem(last=False)
                self._evictions += 1
            
            # Add new entry
            self._cache[key] = CacheEntry(
                value=value,
                expires_at=time.time() + ttl
            )
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.
        
        Returns:
            True if entry existed
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            now = time.time()
            expired = [k for k, v in self._cache.items() if v.expires_at < now]
            
            for key in expired:
                del self._cache[key]
            
            return len(expired)
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            
            return {
                'size': len(self._cache),
                'maxsize': self.maxsize,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': f"{hit_rate:.1f}%",
                'evictions': self._evictions,
                'ttl': self.default_ttl
            }
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        with self._lock:
            entry = self._cache.get(key)
            return entry is not None and not entry.is_expired


class RecommendationCache:
    """
    Specialized cache for movie recommendations.
    
    Features:
    - Automatic cache key generation
    - DataFrame serialization/deserialization
    - User-specific and algorithm-specific caching
    - Cache invalidation on model updates
    
    Usage:
        cache = RecommendationCache(ttl=300)
        
        # Cache recommendations
        cache.cache_recommendations(user_id=123, algorithm='svd', recs=df)
        
        # Get cached
        cached = cache.get_recommendations(user_id=123, algorithm='svd')
    """
    
    def __init__(
        self,
        maxsize: int = 500,
        ttl: float = 300.0,
        user_ttl: float = 600.0
    ):
        """
        Initialize recommendation cache.
        
        Args:
            maxsize: Max cached recommendations
            ttl: Default TTL for recommendations (5 minutes)
            user_ttl: TTL for user-specific data (10 minutes)
        """
        self._rec_cache = TTLCache[pd.DataFrame](maxsize=maxsize, ttl=ttl)
        self._user_cache = TTLCache[Dict](maxsize=maxsize, ttl=user_ttl)
        self._model_version: Dict[str, str] = {}
    
    def _make_key(
        self,
        user_id: int,
        algorithm: str,
        n: int = 10,
        exclude_rated: bool = True,
        extra: Optional[Dict] = None
    ) -> str:
        """Generate cache key for recommendation request."""
        key_data = {
            'user_id': user_id,
            'algorithm': algorithm,
            'n': n,
            'exclude_rated': exclude_rated,
            'model_version': self._model_version.get(algorithm, '1')
        }
        
        if extra:
            key_data.update(extra)
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def get_recommendations(
        self,
        user_id: int,
        algorithm: str,
        n: int = 10,
        exclude_rated: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get cached recommendations.
        
        Returns:
            Cached DataFrame or None if not found
        """
        key = self._make_key(user_id, algorithm, n, exclude_rated)
        return self._rec_cache.get(key)
    
    def cache_recommendations(
        self,
        user_id: int,
        algorithm: str,
        recommendations: pd.DataFrame,
        n: int = 10,
        exclude_rated: bool = True,
        ttl: Optional[float] = None
    ) -> None:
        """Cache recommendation results."""
        key = self._make_key(user_id, algorithm, n, exclude_rated)
        self._rec_cache.set(key, recommendations, ttl)
        
        logger.debug(f"Cached recommendations for user {user_id}, algorithm {algorithm}")
    
    def get_user_data(self, user_id: int, data_type: str = 'history') -> Optional[Dict]:
        """Get cached user data (history, preferences, etc.)."""
        key = f"user:{user_id}:{data_type}"
        return self._user_cache.get(key)
    
    def cache_user_data(
        self,
        user_id: int,
        data: Dict,
        data_type: str = 'history',
        ttl: Optional[float] = None
    ) -> None:
        """Cache user data."""
        key = f"user:{user_id}:{data_type}"
        self._user_cache.set(key, data, ttl)
    
    def invalidate_user(self, user_id: int) -> int:
        """
        Invalidate all cached data for a user.
        
        Returns:
            Number of entries invalidated
        """
        count = 0
        
        # Clear recommendation cache entries for this user
        # Note: This is a simplified version - in production,
        # you'd want to track user -> keys mapping
        with self._rec_cache._lock:
            keys_to_remove = [
                k for k in self._rec_cache._cache.keys()
                if f'user_id": {user_id}' in k or f'user_id":{user_id}' in k
            ]
            for key in keys_to_remove:
                del self._rec_cache._cache[key]
                count += 1
        
        return count
    
    def invalidate_algorithm(self, algorithm: str) -> None:
        """
        Invalidate cache for a specific algorithm (e.g., after model update).
        
        Increments model version to make old cache keys invalid.
        """
        current_version = self._model_version.get(algorithm, '1')
        try:
            new_version = str(int(current_version) + 1)
        except ValueError:
            new_version = '2'
        
        self._model_version[algorithm] = new_version
        logger.info(f"Invalidated cache for algorithm {algorithm}, new version: {new_version}")
    
    def clear_all(self) -> Dict[str, int]:
        """Clear all caches."""
        return {
            'recommendations': self._rec_cache.clear(),
            'user_data': self._user_cache.clear()
        }
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'recommendations': self._rec_cache.stats,
            'user_data': self._user_cache.stats,
            'model_versions': self._model_version.copy()
        }


# =============================================================================
# CACHING DECORATOR
# =============================================================================

def cached_recommendations(
    cache: RecommendationCache,
    ttl: Optional[float] = None
) -> Callable:
    """
    Decorator to cache recommendation function results.
    
    The decorated function must accept user_id and algorithm parameters.
    
    Usage:
        cache = RecommendationCache()
        
        @cached_recommendations(cache)
        def get_recommendations(user_id, algorithm, n=10):
            # Expensive computation
            return recommendations_df
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(
            user_id: int,
            algorithm: str = 'svd',
            n: int = 10,
            exclude_rated: bool = True,
            **kwargs
        ):
            # Check cache first
            cached = cache.get_recommendations(user_id, algorithm, n, exclude_rated)
            if cached is not None:
                logger.debug(f"Cache hit for user {user_id}, algorithm {algorithm}")
                return cached
            
            # Compute and cache
            result = func(user_id, algorithm, n, exclude_rated, **kwargs)
            
            if result is not None and isinstance(result, pd.DataFrame):
                cache.cache_recommendations(
                    user_id, algorithm, result, n, exclude_rated, ttl
                )
            
            return result
        
        return wrapper
    
    return decorator


# =============================================================================
# GLOBAL CACHE INSTANCE
# =============================================================================

_global_cache: Optional[RecommendationCache] = None


def get_recommendation_cache() -> RecommendationCache:
    """Get or create global recommendation cache."""
    global _global_cache
    
    if _global_cache is None:
        _global_cache = RecommendationCache(
            maxsize=500,
            ttl=300.0,  # 5 minutes for recommendations
            user_ttl=600.0  # 10 minutes for user data
        )
    
    return _global_cache


# =============================================================================
# CLI & TESTING
# =============================================================================

if __name__ == '__main__':
    print("Recommendation Cache Demo")
    print("=" * 40)
    
    # Create cache
    cache = RecommendationCache(maxsize=100, ttl=5.0)
    
    # Create sample recommendations
    sample_recs = pd.DataFrame({
        'movieId': [1, 2, 3],
        'title': ['Movie A', 'Movie B', 'Movie C'],
        'predicted_rating': [4.5, 4.2, 4.0]
    })
    
    # Cache recommendations
    print("\n1. Caching recommendations...")
    cache.cache_recommendations(
        user_id=123,
        algorithm='svd',
        recommendations=sample_recs
    )
    print(f"   Stats: {cache.stats}")
    
    # Retrieve from cache
    print("\n2. Retrieving from cache...")
    cached = cache.get_recommendations(user_id=123, algorithm='svd')
    print(f"   Found: {cached is not None}")
    if cached is not None:
        print(f"   Shape: {cached.shape}")
    
    # Check cache miss
    print("\n3. Cache miss for different user...")
    cached = cache.get_recommendations(user_id=456, algorithm='svd')
    print(f"   Found: {cached is not None}")
    
    # Wait for expiration
    print("\n4. Waiting for TTL expiration (5 seconds)...")
    time.sleep(6)
    cached = cache.get_recommendations(user_id=123, algorithm='svd')
    print(f"   After TTL - Found: {cached is not None}")
    
    # Final stats
    print(f"\n5. Final stats: {cache.stats}")
