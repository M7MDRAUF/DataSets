"""
Lazy Loader for CineMatch V2.1.6

Implements lazy loading patterns for models and data:
- Deferred model loading
- Memory-efficient access
- Thread-safe loading
- LRU eviction

Phase 2 - Task 2.2: Lazy Loading Implementation
"""

import logging
import threading
import time
import gc
from typing import Any, Callable, Dict, Generic, Optional, TypeVar
from dataclasses import dataclass, field
from pathlib import Path
from functools import wraps
import weakref

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class LazyLoadConfig:
    """Configuration for lazy loading."""
    max_cached: int = 5           # Maximum cached objects
    ttl_seconds: float = 3600     # Time to live (1 hour)
    preload: bool = False         # Whether to preload
    enable_weak_refs: bool = True # Use weak references when possible


@dataclass
class CacheEntry(Generic[T]):
    """Entry in the lazy cache."""
    value: T
    loaded_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0


class LazyLoader(Generic[T]):
    """
    Lazy loader with LRU eviction and TTL support.
    
    Features:
    - Deferred loading until first access
    - Thread-safe operations
    - LRU eviction policy
    - TTL-based expiration
    - Memory tracking
    """
    
    def __init__(
        self,
        loader: Callable[[], T],
        config: Optional[LazyLoadConfig] = None,
        name: str = "unnamed"
    ):
        self.loader = loader
        self.config = config or LazyLoadConfig()
        self.name = name
        self._value: Optional[T] = None
        self._loaded = False
        self._lock = threading.RLock()
        self._loaded_at: float = 0
        self._access_count: int = 0
        
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def get(self) -> T:
        """Get the value, loading if necessary."""
        if self._loaded:
            # Check TTL
            if self.config.ttl_seconds > 0:
                age = time.time() - self._loaded_at
                if age > self.config.ttl_seconds:
                    logger.debug(f"TTL expired for {self.name}, reloading")
                    self._loaded = False
        
        if not self._loaded:
            with self._lock:
                if not self._loaded:  # Double-check
                    logger.info(f"Lazy loading: {self.name}")
                    start = time.time()
                    self._value = self.loader()
                    self._loaded = True
                    self._loaded_at = time.time()
                    logger.info(f"Loaded {self.name} in {time.time() - start:.2f}s")
        
        self._access_count += 1
        return self._value  # type: ignore
    
    def unload(self) -> None:
        """Unload the value to free memory."""
        with self._lock:
            if self._loaded:
                logger.info(f"Unloading: {self.name}")
                self._value = None
                self._loaded = False
                gc.collect()
    
    def reload(self) -> T:
        """Force reload."""
        self.unload()
        return self.get()


class LazyModelCache:
    """
    Cache for lazily loaded models with LRU eviction.
    
    Features:
    - Multiple model management
    - LRU eviction when full
    - Memory tracking
    - Statistics
    """
    
    def __init__(self, max_models: int = 5, ttl_seconds: float = 3600):
        self.max_models = max_models
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, CacheEntry[Any]] = {}
        self._loaders: Dict[str, Callable[[], Any]] = {}
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    def register(self, name: str, loader: Callable[[], Any]) -> None:
        """Register a lazy loader."""
        self._loaders[name] = loader
        logger.debug(f"Registered lazy loader: {name}")
    
    def get(self, name: str) -> Any:
        """Get a model, loading if necessary."""
        with self._lock:
            # Check cache
            if name in self._cache:
                entry = self._cache[name]
                
                # Check TTL
                if self.ttl_seconds > 0:
                    age = time.time() - entry.loaded_at
                    if age > self.ttl_seconds:
                        logger.debug(f"TTL expired for {name}")
                        del self._cache[name]
                    else:
                        # Cache hit
                        entry.last_accessed = time.time()
                        entry.access_count += 1
                        self._stats['hits'] += 1
                        return entry.value
                else:
                    # No TTL, cache hit
                    entry.last_accessed = time.time()
                    entry.access_count += 1
                    self._stats['hits'] += 1
                    return entry.value
            
            # Cache miss
            self._stats['misses'] += 1
            
            if name not in self._loaders:
                raise KeyError(f"No loader registered for: {name}")
            
            # Evict if necessary
            if len(self._cache) >= self.max_models:
                self._evict_lru()
            
            # Load
            logger.info(f"Loading model: {name}")
            start = time.time()
            value = self._loaders[name]()
            load_time = time.time() - start
            logger.info(f"Loaded {name} in {load_time:.2f}s")
            
            # Cache
            now = time.time()
            self._cache[name] = CacheEntry(
                value=value,
                loaded_at=now,
                last_accessed=now,
                access_count=1
            )
            
            return value
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        # Find LRU
        lru_name = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed
        )
        
        logger.info(f"Evicting LRU model: {lru_name}")
        del self._cache[lru_name]
        self._stats['evictions'] += 1
        gc.collect()
    
    def unload(self, name: str) -> None:
        """Unload a specific model."""
        with self._lock:
            if name in self._cache:
                del self._cache[name]
                gc.collect()
    
    def unload_all(self) -> None:
        """Unload all models."""
        with self._lock:
            self._cache.clear()
            gc.collect()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cached_models': list(self._cache.keys()),
            'cache_size': len(self._cache),
            'max_models': self.max_models,
            **self._stats,
            'hit_rate': self._stats['hits'] / max(1, self._stats['hits'] + self._stats['misses'])
        }
    
    @property
    def loaded_models(self) -> list[str]:
        return list(self._cache.keys())


class LazyProperty(Generic[T]):
    """
    Lazy property descriptor.
    
    Usage:
        class MyClass:
            @lazy_property
            def expensive_value(self) -> int:
                return compute_expensive()
    """
    
    def __init__(self, func: Callable[[Any], T]):
        self.func = func
        self.name = func.__name__
        self._cache: Dict[int, T] = {}
        self._lock = threading.RLock()
    
    def __get__(self, obj: Any, objtype: Optional[type] = None) -> T:
        if obj is None:
            return self  # type: ignore
        
        obj_id = id(obj)
        
        if obj_id not in self._cache:
            with self._lock:
                if obj_id not in self._cache:
                    self._cache[obj_id] = self.func(obj)
        
        return self._cache[obj_id]
    
    def __delete__(self, obj: Any) -> None:
        obj_id = id(obj)
        if obj_id in self._cache:
            del self._cache[obj_id]


def lazy_property(func: Callable[[Any], T]) -> LazyProperty[T]:
    """Decorator for lazy properties."""
    return LazyProperty(func)


def lazy_load(loader: Callable[[], T], name: str = "unnamed") -> LazyLoader[T]:
    """Create a lazy loader."""
    return LazyLoader(loader, name=name)


# Global model cache
_model_cache: Optional[LazyModelCache] = None


def get_model_cache(max_models: int = 5) -> LazyModelCache:
    global _model_cache
    if _model_cache is None:
        _model_cache = LazyModelCache(max_models=max_models)
    return _model_cache


def register_model(name: str, loader: Callable[[], Any]) -> None:
    """Register a model for lazy loading."""
    get_model_cache().register(name, loader)


def get_model(name: str) -> Any:
    """Get a lazily loaded model."""
    return get_model_cache().get(name)


def unload_model(name: str) -> None:
    """Unload a model."""
    get_model_cache().unload(name)


if __name__ == "__main__":
    import random
    
    print("Lazy Loader Demo")
    print("=" * 50)
    
    # Demo lazy loader
    def expensive_computation():
        print("Computing expensive value...")
        time.sleep(0.1)  # Simulate work
        return random.randint(1, 100)
    
    lazy_value = LazyLoader(expensive_computation, name="random_value")
    
    print(f"\n1. Lazy value created, loaded: {lazy_value.is_loaded}")
    
    print(f"2. Accessing value...")
    value = lazy_value.get()
    print(f"   Value: {value}, loaded: {lazy_value.is_loaded}")
    
    print(f"3. Accessing again (should be cached)...")
    value2 = lazy_value.get()
    print(f"   Value: {value2}")
    
    print(f"4. Unloading...")
    lazy_value.unload()
    print(f"   Loaded: {lazy_value.is_loaded}")
    
    # Demo model cache
    print("\n\nModel Cache Demo")
    print("-" * 50)
    
    cache = LazyModelCache(max_models=2)
    
    # Register models
    for i in range(3):
        cache.register(f"model_{i}", lambda i=i: {"id": i, "data": [0] * 1000})
    
    print("Registered 3 models, max cache = 2")
    
    print("\nLoading model_0...")
    cache.get("model_0")
    print(f"  Loaded: {cache.loaded_models}")
    
    print("\nLoading model_1...")
    cache.get("model_1")
    print(f"  Loaded: {cache.loaded_models}")
    
    print("\nLoading model_2 (should evict model_0)...")
    cache.get("model_2")
    print(f"  Loaded: {cache.loaded_models}")
    
    print("\nCache stats:")
    stats = cache.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
