"""
CineMatch V2.1.6 - Algorithm Plugin System

Dynamic algorithm registration, discovery, and factory pattern implementation.
Allows third-party algorithms to be registered and used seamlessly.

Author: CineMatch Development Team
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Type, Optional, Any, Callable, TypeVar
from dataclasses import dataclass, field
from enum import Enum
import importlib
import pkgutil
import inspect
import logging
from pathlib import Path
from functools import wraps
import time

from .base_recommender import BaseRecommender, AlgorithmMetrics


logger = logging.getLogger(__name__)


class AlgorithmCategory(Enum):
    """Categories for recommendation algorithms"""
    COLLABORATIVE = "collaborative"
    CONTENT_BASED = "content_based"
    HYBRID = "hybrid"
    DEEP_LEARNING = "deep_learning"
    GRAPH_BASED = "graph_based"
    CONTEXT_AWARE = "context_aware"
    OTHER = "other"


@dataclass
class AlgorithmMetadata:
    """Metadata for registered algorithms"""
    name: str
    version: str
    category: AlgorithmCategory
    author: str
    description: str
    algorithm_class: Type[BaseRecommender]
    default_params: Dict[str, Any] = field(default_factory=dict)
    required_features: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    is_experimental: bool = False
    min_ratings_required: int = 0
    supports_incremental_training: bool = False
    supports_cold_start: bool = False
    memory_complexity: str = "O(n)"
    time_complexity: str = "O(n)"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            'name': self.name,
            'version': self.version,
            'category': self.category.value,
            'author': self.author,
            'description': self.description,
            'class': self.algorithm_class.__name__,
            'default_params': self.default_params,
            'required_features': self.required_features,
            'tags': self.tags,
            'is_experimental': self.is_experimental,
            'min_ratings_required': self.min_ratings_required,
            'supports_incremental_training': self.supports_incremental_training,
            'supports_cold_start': self.supports_cold_start,
            'memory_complexity': self.memory_complexity,
            'time_complexity': self.time_complexity
        }


class AlgorithmRegistry:
    """
    Central registry for recommendation algorithms.
    
    Implements singleton pattern to ensure single source of truth
    for all registered algorithms.
    """
    
    _instance: Optional['AlgorithmRegistry'] = None
    _algorithms: Dict[str, AlgorithmMetadata] = {}
    _hooks: Dict[str, List[Callable]] = {
        'pre_register': [],
        'post_register': [],
        'pre_create': [],
        'post_create': []
    }
    
    def __new__(cls) -> 'AlgorithmRegistry':
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._algorithms = {}
            cls._instance._hooks = {
                'pre_register': [],
                'post_register': [],
                'pre_create': [],
                'post_create': []
            }
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> 'AlgorithmRegistry':
        """Get singleton instance"""
        return cls()
    
    @classmethod
    def reset(cls) -> None:
        """Reset registry (useful for testing)"""
        cls._instance = None
        cls._algorithms = {}
        cls._hooks = {
            'pre_register': [],
            'post_register': [],
            'pre_create': [],
            'post_create': []
        }
    
    def register(
        self,
        name: str,
        algorithm_class: Type[BaseRecommender],
        version: str = "1.0.0",
        category: AlgorithmCategory = AlgorithmCategory.OTHER,
        author: str = "Unknown",
        description: str = "",
        default_params: Optional[Dict[str, Any]] = None,
        required_features: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        is_experimental: bool = False,
        min_ratings_required: int = 0,
        supports_incremental_training: bool = False,
        supports_cold_start: bool = False,
        memory_complexity: str = "O(n)",
        time_complexity: str = "O(n)"
    ) -> None:
        """
        Register an algorithm with the registry.
        
        Args:
            name: Unique algorithm name
            algorithm_class: Class implementing BaseRecommender
            version: Algorithm version
            category: Algorithm category
            author: Author name
            description: Human-readable description
            default_params: Default initialization parameters
            required_features: Features required in dataset
            tags: Searchable tags
            is_experimental: Whether algorithm is experimental
            min_ratings_required: Minimum ratings needed
            supports_incremental_training: Can be updated incrementally
            supports_cold_start: Handles new users/items
            memory_complexity: Big-O memory complexity
            time_complexity: Big-O time complexity
        """
        # Validate algorithm class
        if not inspect.isclass(algorithm_class):
            raise TypeError(f"algorithm_class must be a class, got {type(algorithm_class)}")
        
        if not issubclass(algorithm_class, BaseRecommender):
            raise TypeError(
                f"algorithm_class must be subclass of BaseRecommender, "
                f"got {algorithm_class.__bases__}"
            )
        
        # Run pre-register hooks
        for hook in self._hooks.get('pre_register', []):
            hook(name, algorithm_class)
        
        # Check for duplicate
        if name in self._algorithms:
            logger.warning(f"Algorithm '{name}' already registered, overwriting...")
        
        # Create metadata
        metadata = AlgorithmMetadata(
            name=name,
            version=version,
            category=category,
            author=author,
            description=description or algorithm_class.__doc__ or "",
            algorithm_class=algorithm_class,
            default_params=default_params or {},
            required_features=required_features or [],
            tags=tags or [],
            is_experimental=is_experimental,
            min_ratings_required=min_ratings_required,
            supports_incremental_training=supports_incremental_training,
            supports_cold_start=supports_cold_start,
            memory_complexity=memory_complexity,
            time_complexity=time_complexity
        )
        
        self._algorithms[name] = metadata
        logger.info(f"Registered algorithm: {name} v{version} ({category.value})")
        
        # Run post-register hooks
        for hook in self._hooks.get('post_register', []):
            hook(name, metadata)
    
    def unregister(self, name: str) -> bool:
        """
        Remove an algorithm from the registry.
        
        Args:
            name: Algorithm name to remove
            
        Returns:
            True if removed, False if not found
        """
        if name in self._algorithms:
            del self._algorithms[name]
            logger.info(f"Unregistered algorithm: {name}")
            return True
        return False
    
    def get(self, name: str) -> Optional[AlgorithmMetadata]:
        """Get algorithm metadata by name"""
        return self._algorithms.get(name)
    
    def get_all(self) -> Dict[str, AlgorithmMetadata]:
        """Get all registered algorithms"""
        return self._algorithms.copy()
    
    def list_names(self) -> List[str]:
        """List all registered algorithm names"""
        return list(self._algorithms.keys())
    
    def list_by_category(self, category: AlgorithmCategory) -> List[AlgorithmMetadata]:
        """Get algorithms by category"""
        return [
            meta for meta in self._algorithms.values()
            if meta.category == category
        ]
    
    def search(
        self,
        query: str = "",
        category: Optional[AlgorithmCategory] = None,
        tags: Optional[List[str]] = None,
        exclude_experimental: bool = False
    ) -> List[AlgorithmMetadata]:
        """
        Search algorithms by various criteria.
        
        Args:
            query: Text search in name and description
            category: Filter by category
            tags: Filter by tags (any match)
            exclude_experimental: Exclude experimental algorithms
            
        Returns:
            List of matching algorithm metadata
        """
        results = []
        
        for meta in self._algorithms.values():
            # Filter by experimental
            if exclude_experimental and meta.is_experimental:
                continue
            
            # Filter by category
            if category and meta.category != category:
                continue
            
            # Filter by tags
            if tags:
                if not any(tag in meta.tags for tag in tags):
                    continue
            
            # Text search
            if query:
                query_lower = query.lower()
                if (query_lower not in meta.name.lower() and
                    query_lower not in meta.description.lower()):
                    continue
            
            results.append(meta)
        
        return results
    
    def add_hook(self, event: str, callback: Callable) -> None:
        """
        Add a lifecycle hook.
        
        Events: pre_register, post_register, pre_create, post_create
        """
        if event not in self._hooks:
            raise ValueError(f"Unknown event: {event}")
        self._hooks[event].append(callback)
    
    def remove_hook(self, event: str, callback: Callable) -> bool:
        """Remove a lifecycle hook"""
        if event in self._hooks and callback in self._hooks[event]:
            self._hooks[event].remove(callback)
            return True
        return False


class AlgorithmFactory:
    """
    Factory for creating algorithm instances.
    
    Provides a clean interface for instantiating algorithms
    with proper parameter handling and validation.
    """
    
    def __init__(self, registry: Optional[AlgorithmRegistry] = None):
        """
        Initialize factory with optional registry.
        
        Args:
            registry: Algorithm registry (uses singleton if not provided)
        """
        self.registry = registry or AlgorithmRegistry.get_instance()
    
    def create(
        self,
        name: str,
        **kwargs: Any
    ) -> BaseRecommender:
        """
        Create an algorithm instance.
        
        Args:
            name: Registered algorithm name
            **kwargs: Override default parameters
            
        Returns:
            Configured algorithm instance
        """
        metadata = self.registry.get(name)
        
        if metadata is None:
            available = ", ".join(self.registry.list_names())
            raise ValueError(
                f"Unknown algorithm: '{name}'. Available: {available}"
            )
        
        # Run pre-create hooks
        for hook in self.registry._hooks.get('pre_create', []):
            hook(name, kwargs)
        
        # Merge default params with overrides
        params = {**metadata.default_params, **kwargs}
        
        # Create instance
        try:
            instance = metadata.algorithm_class(name=metadata.name, **params)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create algorithm '{name}': {e}"
            ) from e
        
        # Run post-create hooks
        for hook in self.registry._hooks.get('post_create', []):
            hook(name, instance)
        
        logger.debug(f"Created algorithm instance: {name}")
        return instance
    
    def create_multiple(
        self,
        configs: List[Dict[str, Any]]
    ) -> List[BaseRecommender]:
        """
        Create multiple algorithm instances.
        
        Args:
            configs: List of dicts with 'name' and optional params
            
        Returns:
            List of algorithm instances
        """
        instances = []
        for config in configs:
            name = config.pop('name')
            instance = self.create(name, **config)
            instances.append(instance)
        return instances
    
    def get_available_params(self, name: str) -> Dict[str, Any]:
        """Get available parameters for an algorithm"""
        metadata = self.registry.get(name)
        if metadata is None:
            raise ValueError(f"Unknown algorithm: {name}")
        
        # Get init signature
        sig = inspect.signature(metadata.algorithm_class.__init__)
        params = {}
        
        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'name'):
                continue
            
            params[param_name] = {
                'default': param.default if param.default != inspect.Parameter.empty else None,
                'required': param.default == inspect.Parameter.empty,
                'annotation': str(param.annotation) if param.annotation != inspect.Parameter.empty else None
            }
        
        return params


# Decorator for easy algorithm registration
def register_algorithm(
    name: str,
    version: str = "1.0.0",
    category: AlgorithmCategory = AlgorithmCategory.OTHER,
    author: str = "Unknown",
    description: str = "",
    default_params: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    is_experimental: bool = False,
    **kwargs: Any
):
    """
    Decorator to register an algorithm class.
    
    Usage:
        @register_algorithm(
            name="my_algorithm",
            version="1.0.0",
            category=AlgorithmCategory.COLLABORATIVE,
            author="Dev Team"
        )
        class MyAlgorithm(BaseRecommender):
            ...
    """
    def decorator(cls: Type[BaseRecommender]) -> Type[BaseRecommender]:
        registry = AlgorithmRegistry.get_instance()
        registry.register(
            name=name,
            algorithm_class=cls,
            version=version,
            category=category,
            author=author,
            description=description or cls.__doc__ or "",
            default_params=default_params,
            tags=tags,
            is_experimental=is_experimental,
            **kwargs
        )
        return cls
    return decorator


class PluginLoader:
    """
    Dynamic plugin loader for algorithms.
    
    Discovers and loads algorithm plugins from specified paths.
    """
    
    def __init__(
        self,
        plugin_paths: Optional[List[Path]] = None,
        registry: Optional[AlgorithmRegistry] = None
    ):
        """
        Initialize plugin loader.
        
        Args:
            plugin_paths: List of paths to search for plugins
            registry: Algorithm registry to register plugins with
        """
        self.plugin_paths = plugin_paths or []
        self.registry = registry or AlgorithmRegistry.get_instance()
        self.loaded_plugins: Dict[str, Any] = {}
    
    def add_plugin_path(self, path: Path) -> None:
        """Add a path to search for plugins"""
        if path not in self.plugin_paths:
            self.plugin_paths.append(path)
    
    def discover_plugins(self) -> List[str]:
        """
        Discover available plugins without loading.
        
        Returns:
            List of discovered plugin module names
        """
        discovered = []
        
        for plugin_path in self.plugin_paths:
            if not plugin_path.exists():
                logger.warning(f"Plugin path does not exist: {plugin_path}")
                continue
            
            for file in plugin_path.glob("*.py"):
                if file.name.startswith("_"):
                    continue
                module_name = file.stem
                discovered.append(module_name)
        
        return discovered
    
    def load_plugin(self, module_path: str) -> bool:
        """
        Load a specific plugin module.
        
        Args:
            module_path: Dot-separated module path
            
        Returns:
            True if loaded successfully
        """
        try:
            module = importlib.import_module(module_path)
            self.loaded_plugins[module_path] = module
            logger.info(f"Loaded plugin: {module_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load plugin {module_path}: {e}")
            return False
    
    def load_all_plugins(self) -> Dict[str, bool]:
        """
        Load all discovered plugins.
        
        Returns:
            Dict mapping plugin names to load success
        """
        results = {}
        
        for plugin_name in self.discover_plugins():
            # Try to find the module path
            for plugin_path in self.plugin_paths:
                module_path = f"{plugin_path.name}.{plugin_name}"
                results[plugin_name] = self.load_plugin(module_path)
                if results[plugin_name]:
                    break
        
        return results
    
    def reload_plugin(self, module_path: str) -> bool:
        """
        Reload a plugin module.
        
        Args:
            module_path: Module path to reload
            
        Returns:
            True if reloaded successfully
        """
        if module_path not in self.loaded_plugins:
            return self.load_plugin(module_path)
        
        try:
            module = self.loaded_plugins[module_path]
            importlib.reload(module)
            logger.info(f"Reloaded plugin: {module_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to reload plugin {module_path}: {e}")
            return False


def register_builtin_algorithms() -> None:
    """Register all built-in CineMatch algorithms"""
    from .svd_recommender import SVDRecommender
    from .user_knn_recommender import UserKNNRecommender
    from .item_knn_recommender import ItemKNNRecommender
    from .hybrid_recommender import HybridRecommender
    from .content_based_recommender import ContentBasedRecommender
    
    registry = AlgorithmRegistry.get_instance()
    
    # SVD Algorithm
    registry.register(
        name="svd",
        algorithm_class=SVDRecommender,
        version="2.1.6",
        category=AlgorithmCategory.COLLABORATIVE,
        author="CineMatch Team",
        description="Matrix factorization using Singular Value Decomposition",
        default_params={'n_factors': 100, 'n_epochs': 20, 'lr_all': 0.005, 'reg_all': 0.02},
        tags=["matrix-factorization", "latent-factors", "scalable"],
        supports_cold_start=False,
        memory_complexity="O(n*k)",
        time_complexity="O(n*k*epochs)"
    )
    
    # User-KNN Algorithm
    registry.register(
        name="user_knn",
        algorithm_class=UserKNNRecommender,
        version="2.1.6",
        category=AlgorithmCategory.COLLABORATIVE,
        author="CineMatch Team",
        description="User-based collaborative filtering with k-nearest neighbors",
        default_params={'k': 40, 'min_k': 1, 'sim_type': 'cosine'},
        tags=["neighborhood", "user-based", "interpretable"],
        supports_cold_start=False,
        memory_complexity="O(n²)",
        time_complexity="O(n*k)"
    )
    
    # Item-KNN Algorithm
    registry.register(
        name="item_knn",
        algorithm_class=ItemKNNRecommender,
        version="2.1.6",
        category=AlgorithmCategory.COLLABORATIVE,
        author="CineMatch Team",
        description="Item-based collaborative filtering with k-nearest neighbors",
        default_params={'k': 40, 'min_k': 1, 'sim_type': 'cosine'},
        tags=["neighborhood", "item-based", "interpretable"],
        supports_cold_start=False,
        memory_complexity="O(m²)",
        time_complexity="O(m*k)"
    )
    
    # Hybrid Algorithm
    registry.register(
        name="hybrid",
        algorithm_class=HybridRecommender,
        version="2.1.6",
        category=AlgorithmCategory.HYBRID,
        author="CineMatch Team",
        description="Weighted hybrid combining multiple algorithms",
        default_params={'weights': {'svd': 0.4, 'user_knn': 0.3, 'item_knn': 0.3}},
        tags=["ensemble", "hybrid", "flexible"],
        supports_cold_start=False,
        memory_complexity="O(sum of components)",
        time_complexity="O(sum of components)"
    )
    
    # Content-Based Algorithm
    registry.register(
        name="content_based",
        algorithm_class=ContentBasedRecommender,
        version="2.1.6",
        category=AlgorithmCategory.CONTENT_BASED,
        author="CineMatch Team",
        description="Content-based filtering using movie features",
        default_params={'feature_weights': {'genres': 1.0, 'year': 0.3}},
        required_features=["genres"],
        tags=["content", "feature-based", "cold-start"],
        supports_cold_start=True,
        memory_complexity="O(m*f)",
        time_complexity="O(m*f)"
    )
    
    logger.info("Registered 5 built-in algorithms")


# Initialize built-in algorithms when module is imported
try:
    register_builtin_algorithms()
except ImportError as e:
    logger.warning(f"Could not register all built-in algorithms: {e}")
