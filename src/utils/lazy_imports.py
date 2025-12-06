"""
CineMatch V2.1.6 - Lazy Import Utilities

Defers heavy module imports until first use to reduce startup time.
Particularly useful for sklearn, scipy, numpy, and pandas.

Author: CineMatch Development Team
Date: December 5, 2025

Usage:
    from src.utils.lazy_imports import LazyModule, lazy_import
    
    # Option 1: Module wrapper
    np = LazyModule('numpy')
    np.array([1, 2, 3])  # numpy imported here
    
    # Option 2: Decorator for functions
    @lazy_import('sklearn.neighbors', 'NearestNeighbors')
    def create_knn_model():
        return NearestNeighbors(n_neighbors=10)
"""

import importlib
import sys
from typing import Any, Callable, Optional, TypeVar
from functools import wraps

F = TypeVar('F', bound=Callable[..., Any])


class LazyModule:
    """
    Lazy module loader that defers import until first attribute access.
    
    Reduces startup time by not loading heavy modules until needed.
    
    Example:
        np = LazyModule('numpy')
        # numpy not loaded yet
        
        arr = np.array([1, 2, 3])
        # numpy loaded on first use
    """
    
    __slots__ = ('_module_name', '_module', '_submodule')
    
    def __init__(self, module_name: str, submodule: Optional[str] = None):
        """
        Initialize lazy module.
        
        Args:
            module_name: Full module name (e.g., 'numpy', 'sklearn.neighbors')
            submodule: Optional submodule/class to import (e.g., 'NearestNeighbors')
        """
        object.__setattr__(self, '_module_name', module_name)
        object.__setattr__(self, '_module', None)
        object.__setattr__(self, '_submodule', submodule)
    
    def _load(self) -> Any:
        """Load the module on first access."""
        module = object.__getattribute__(self, '_module')
        if module is None:
            module_name = object.__getattribute__(self, '_module_name')
            submodule = object.__getattribute__(self, '_submodule')
            
            module = importlib.import_module(module_name)
            
            if submodule:
                module = getattr(module, submodule)
            
            object.__setattr__(self, '_module', module)
        
        return module
    
    def __getattr__(self, name: str) -> Any:
        """Get attribute from loaded module."""
        return getattr(self._load(), name)
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute on loaded module."""
        setattr(self._load(), name, value)
    
    def __repr__(self) -> str:
        module_name = object.__getattribute__(self, '_module_name')
        submodule = object.__getattribute__(self, '_submodule')
        module = object.__getattribute__(self, '_module')
        
        loaded = "loaded" if module is not None else "not loaded"
        if submodule:
            return f"<LazyModule '{module_name}.{submodule}' ({loaded})>"
        return f"<LazyModule '{module_name}' ({loaded})>"
    
    def __call__(self, *args, **kwargs):
        """Support calling if the module/submodule is callable."""
        return self._load()(*args, **kwargs)


class LazyImportGroup:
    """
    Group of lazy imports for related modules.
    
    Example:
        ml = LazyImportGroup()
        ml.add('numpy', 'np')
        ml.add('pandas', 'pd')
        ml.add('sklearn.neighbors', 'NearestNeighbors', 'NearestNeighbors')
        
        # Access lazily
        arr = ml.np.array([1, 2, 3])
        df = ml.pd.DataFrame({'a': [1, 2, 3]})
    """
    
    def __init__(self):
        self._modules = {}
    
    def add(self, module_name: str, alias: str, submodule: Optional[str] = None) -> None:
        """Add a lazy module to the group."""
        self._modules[alias] = LazyModule(module_name, submodule)
    
    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(name)
        
        if name in self._modules:
            return self._modules[name]
        
        raise AttributeError(f"No lazy module named '{name}'")


def lazy_import(module_name: str, *names: str) -> Callable[[F], F]:
    """
    Decorator to lazily import modules/classes before function execution.
    
    Args:
        module_name: Module to import
        *names: Names to import from module (injected into function globals)
        
    Returns:
        Decorated function
        
    Example:
        @lazy_import('sklearn.neighbors', 'NearestNeighbors')
        def create_model():
            return NearestNeighbors(n_neighbors=10)
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Import module
            module = importlib.import_module(module_name)
            
            # Inject names into function's global namespace
            for name in names:
                func.__globals__[name] = getattr(module, name)
            
            return func(*args, **kwargs)
        
        return wrapper  # type: ignore
    
    return decorator


# =============================================================================
# PRE-CONFIGURED LAZY MODULES
# =============================================================================

# Heavy scientific computing modules
numpy = LazyModule('numpy')
pandas = LazyModule('pandas')
scipy = LazyModule('scipy')

# sklearn components (import on demand)
sklearn_neighbors = LazyModule('sklearn.neighbors')
sklearn_metrics = LazyModule('sklearn.metrics.pairwise')
sklearn_decomposition = LazyModule('sklearn.decomposition')

# scipy.sparse
scipy_sparse = LazyModule('scipy.sparse')


def get_lazy_numpy():
    """Get lazy numpy module."""
    return numpy


def get_lazy_pandas():
    """Get lazy pandas module."""
    return pandas


def get_lazy_sklearn_neighbors():
    """Get lazy sklearn.neighbors module."""
    return sklearn_neighbors


# =============================================================================
# CONDITIONAL IMPORT HELPERS
# =============================================================================

def import_if_available(module_name: str) -> Optional[Any]:
    """
    Import a module if available, return None otherwise.
    
    Useful for optional dependencies.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def require_module(module_name: str, feature_name: str) -> Any:
    """
    Import a module, raising helpful error if not available.
    
    Args:
        module_name: Module to import
        feature_name: Feature name for error message
        
    Raises:
        ImportError: With helpful message if module not found
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        raise ImportError(
            f"Module '{module_name}' is required for {feature_name}. "
            f"Install it with: pip install {module_name}"
        )


# =============================================================================
# STARTUP OPTIMIZATION
# =============================================================================

def defer_heavy_imports():
    """
    Defer heavy imports in sys.modules.
    
    Call this at application startup to replace heavy modules with lazy versions.
    WARNING: Use with caution - may cause issues with some libraries.
    """
    heavy_modules = [
        'numpy',
        'pandas', 
        'scipy',
        'sklearn',
        'matplotlib',
        'plotly'
    ]
    
    for module in heavy_modules:
        if module not in sys.modules:
            sys.modules[module] = LazyModule(module)


# =============================================================================
# CLI & TESTING
# =============================================================================

if __name__ == '__main__':
    import time
    
    print("Lazy Import Demo")
    print("=" * 40)
    
    # Test lazy loading
    print("\n1. Create lazy numpy wrapper...")
    np_lazy = LazyModule('numpy')
    print(f"   Created: {np_lazy}")
    
    print("\n2. Access numpy array (triggers import)...")
    start = time.time()
    arr = np_lazy.array([1, 2, 3, 4, 5])
    elapsed = time.time() - start
    print(f"   Result: {arr}")
    print(f"   Time: {elapsed:.3f}s")
    print(f"   After: {np_lazy}")
    
    print("\n3. Second access (already loaded)...")
    start = time.time()
    arr2 = np_lazy.sum(arr)
    elapsed = time.time() - start
    print(f"   Sum: {arr2}")
    print(f"   Time: {elapsed:.6f}s (much faster)")
    
    # Test decorator
    print("\n4. Test lazy_import decorator...")
    
    @lazy_import('collections', 'Counter')
    def count_items(items):
        Counter = globals()['Counter']  # Retrieved from globals by decorator
        return Counter(items)
    
    result = count_items(['a', 'b', 'a', 'c', 'a'])
    print(f"   Counter result: {result}")
