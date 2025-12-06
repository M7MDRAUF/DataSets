"""
CineMatch Performance Module

Provides performance optimization utilities for CineMatch including:
- DataFrame optimization
- Lazy loading
- Batch processing
- Caching strategies
- Memory profiling
- Data type optimization
- Pagination
- Popular items caching
- Performance benchmarking

Author: CineMatch Development Team
Date: November 2025
"""

from .dataframe_optimizer import (
    DataFrameOptimizer,
    OptimizationConfig,
    OptimizationReport,
    optimize_dataframe,
    reduce_memory_usage,
    get_memory_usage,
)

from .lazy_loader import (
    LazyLoader,
    LazyModelCache,
    LazyProperty,
    LazyLoadConfig,
    lazy_load,
    lazy_property,
    get_model_cache,
    register_model,
    get_model,
    unload_model,
)

from .batch_processor import (
    BatchProcessor,
    AsyncBatchProcessor,
    BatchQueue,
    BatchConfig,
    BatchResult,
    BatchProgress,
    BatchStatus,
    process_in_batches,
)

from .caching import (
    CacheConfig,
    CacheStats,
    CacheEntry,
    LRUCache,
    TieredCache,
    CacheManager,
    cached,
    get_cache,
    get_cache_manager,
    get_recommendation_cache,
    cache_recommendations,
    get_cached_recommendations,
)

from .memory_profiler import (
    MemoryProfiler,
    MemoryTracker,
    MemorySnapshot,
    MemoryProfile,
    profile_memory,
    get_memory_usage,
    get_object_size,
    get_object_size_mb,
    force_gc,
    start_memory_monitoring,
    stop_memory_monitoring,
    get_memory_trend,
)

from .dtype_optimizer import (
    DTypeOptimizer,
    DTypeStrategy,
    DTypeReport,
    DTypeRecommendation,
    optimize_dtypes,
    recommend_dtype,
    downcast_array,
)

from .pagination import (
    Paginator,
    CursorPaginator,
    RecommendationPaginator,
    Page,
    PageInfo,
    PageRequest,
    PaginationType,
    paginate,
    paginate_recommendations,
)

from .popular_items import (
    PopularItemsCache,
    MoviePopularityCache,
    PopularItem,
    PopularityConfig,
    PopularityWindow,
    get_movie_popularity_cache,
    get_popular_movies,
    get_trending_movies,
    record_movie_interaction,
)

from .benchmarks import (
    Benchmark,
    RecommendationBenchmark,
    BenchmarkConfig,
    TimingResult,
    MemoryResult,
    timed,
    profile_throughput,
    time_function,
    compare_functions,
)

__all__ = [
    # DataFrame optimization
    'DataFrameOptimizer',
    'OptimizationConfig',
    'OptimizationReport',
    'optimize_dataframe',
    'reduce_memory_usage',
    'get_memory_usage',
    # Lazy loading
    'LazyLoader',
    'LazyModelCache',
    'LazyProperty',
    'LazyLoadConfig',
    'lazy_load',
    'lazy_property',
    'get_model_cache',
    'register_model',
    'get_model',
    'unload_model',
    # Batch processing
    'BatchProcessor',
    'AsyncBatchProcessor',
    'BatchQueue',
    'BatchConfig',
    'BatchResult',
    'BatchProgress',
    'BatchStatus',
    'process_in_batches',
    # Caching
    'CacheConfig',
    'CacheStats',
    'CacheEntry',
    'LRUCache',
    'TieredCache',
    'CacheManager',
    'cached',
    'get_cache',
    'get_cache_manager',
    'get_recommendation_cache',
    'cache_recommendations',
    'get_cached_recommendations',
    # Memory profiling
    'MemoryProfiler',
    'MemoryTracker',
    'MemorySnapshot',
    'MemoryProfile',
    'profile_memory',
    'get_object_size',
    'get_object_size_mb',
    'force_gc',
    'start_memory_monitoring',
    'stop_memory_monitoring',
    'get_memory_trend',
    # DType optimization
    'DTypeOptimizer',
    'DTypeStrategy',
    'DTypeReport',
    'DTypeRecommendation',
    'optimize_dtypes',
    'recommend_dtype',
    'downcast_array',
    # Pagination
    'Paginator',
    'CursorPaginator',
    'RecommendationPaginator',
    'Page',
    'PageInfo',
    'PageRequest',
    'PaginationType',
    'paginate',
    'paginate_recommendations',
    # Popular items
    'PopularItemsCache',
    'MoviePopularityCache',
    'PopularItem',
    'PopularityConfig',
    'PopularityWindow',
    'get_movie_popularity_cache',
    'get_popular_movies',
    'get_trending_movies',
    'record_movie_interaction',
    # Benchmarks
    'Benchmark',
    'RecommendationBenchmark',
    'BenchmarkConfig',
    'TimingResult',
    'MemoryResult',
    'timed',
    'profile_throughput',
    'time_function',
    'compare_functions',
]
