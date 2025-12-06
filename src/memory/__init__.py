"""
CineMatch Memory Management Module

Provides memory management utilities for CineMatch including:
- Memory monitoring and tracking
- Garbage collection optimization
- Object pooling for reuse
- Memory-aware caching
- Memory leak detection
- Memory limit enforcement

Author: CineMatch Development Team
Date: November 2025
"""

from .memory_monitor import (
    MemoryMonitor,
    MemorySnapshot,
    MemoryConfig,
    MemoryLevel,
    MemoryAlert,
    get_memory_monitor,
    get_memory_usage,
    check_memory,
    start_memory_monitoring,
    stop_memory_monitoring,
)

from .gc_optimizer import (
    GCOptimizer,
    GCConfig,
    GCStats,
    GCMode,
    GCContext,
    get_gc_optimizer,
    gc_collect,
    set_gc_mode,
    gc_disabled,
)

from .object_pool import (
    ObjectPool,
    PoolManager,
    PoolConfig,
    PooledObject,
    PoolContext,
    get_pool_manager,
    create_pool,
    get_pool,
)

from .cache_manager import (
    MemoryAwareCache,
    CacheManager,
    CacheConfig,
    CacheEntry,
    CacheStats,
    EvictionPolicy,
    get_cache_manager,
    get_cache,
    create_cache,
)

from .leak_detector import (
    MemoryLeakDetector,
    LeakConfig,
    LeakReport,
    ObjectSnapshot,
    ObjectTracker,
    get_leak_detector,
    take_memory_snapshot,
    detect_memory_leaks,
    track_for_leaks,
)

from .memory_limits import (
    MemoryLimiter,
    MemoryLimitConfig,
    MemoryBudget,
    LimitViolation,
    LimitAction,
    ThrottleLevel,
    MemoryGuard,
    get_memory_limiter,
    check_memory_limits,
    request_memory,
    memory_guard,
)

from .model_cache import (
    ModelCache,
    CacheEntry as ModelCacheEntry,
    get_model_cache,
    model_cache,
)

from .model_preloader import (
    ModelPreloader,
    UsageStats,
    PreloadTask,
    get_model_preloader,
    preload_on_startup,
    record_model_usage,
)

__all__ = [
    # Memory monitor
    'MemoryMonitor',
    'MemorySnapshot',
    'MemoryConfig',
    'MemoryLevel',
    'MemoryAlert',
    'get_memory_monitor',
    'get_memory_usage',
    'check_memory',
    'start_memory_monitoring',
    'stop_memory_monitoring',
    # GC optimizer
    'GCOptimizer',
    'GCConfig',
    'GCStats',
    'GCMode',
    'GCContext',
    'get_gc_optimizer',
    'gc_collect',
    'set_gc_mode',
    'gc_disabled',
    # Object pool
    'ObjectPool',
    'PoolManager',
    'PoolConfig',
    'PooledObject',
    'PoolContext',
    'get_pool_manager',
    'create_pool',
    'get_pool',
    # Cache manager
    'MemoryAwareCache',
    'CacheManager',
    'CacheConfig',
    'CacheEntry',
    'CacheStats',
    'EvictionPolicy',
    'get_cache_manager',
    'get_cache',
    'create_cache',
    # Leak detector
    'MemoryLeakDetector',
    'LeakConfig',
    'LeakReport',
    'ObjectSnapshot',
    'ObjectTracker',
    'get_leak_detector',
    'take_memory_snapshot',
    'detect_memory_leaks',
    'track_for_leaks',
    # Memory limits
    'MemoryLimiter',
    'MemoryLimitConfig',
    'MemoryBudget',
    'LimitViolation',
    'LimitAction',
    'ThrottleLevel',
    'MemoryGuard',
    'get_memory_limiter',
    'check_memory_limits',
    'request_memory',
    'memory_guard',
    # Model cache (V2.1.6)
    'ModelCache',
    'ModelCacheEntry',
    'get_model_cache',
    'model_cache',
    # Model preloader (V2.1.6)
    'ModelPreloader',
    'UsageStats',
    'PreloadTask',
    'get_model_preloader',
    'preload_on_startup',
    'record_model_usage',
]
