"""
CineMatch V2.1.6 - Deprecation Warning Utilities

Provides decorators and utilities for marking deprecated code with:
- Clear deprecation messages
- Version information (deprecated in, removal planned)
- Suggested alternatives
- Stack trace context

Usage:
    from src.utils.deprecation import deprecated, deprecation_warning
    
    @deprecated(
        version="2.1.6",
        reason="Use new_function() instead",
        removal_version="3.0.0"
    )
    def old_function():
        pass

Author: CineMatch Development Team
Date: December 2025
"""

import functools
import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type, TypeVar

T = TypeVar('T')


@dataclass
class DeprecationInfo:
    """Information about a deprecation."""
    name: str
    version: str  # Version when deprecated
    reason: str
    removal_version: Optional[str] = None
    alternative: Optional[str] = None
    
    def to_message(self) -> str:
        """Generate deprecation warning message."""
        msg_parts = [f"'{self.name}' is deprecated since version {self.version}"]
        
        if self.reason:
            msg_parts.append(f": {self.reason}")
        
        if self.alternative:
            msg_parts.append(f" Use '{self.alternative}' instead.")
        
        if self.removal_version:
            msg_parts.append(f" Will be removed in version {self.removal_version}.")
        
        return "".join(msg_parts)


# Registry of deprecated items for documentation
_deprecation_registry: dict[str, DeprecationInfo] = {}


def get_deprecation_registry() -> dict[str, DeprecationInfo]:
    """Get the registry of all deprecated items."""
    return _deprecation_registry.copy()


def deprecated(
    version: str,
    reason: str = "",
    removal_version: Optional[str] = None,
    alternative: Optional[str] = None,
    category: Type[Warning] = DeprecationWarning
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to mark functions/methods as deprecated.
    
    Emits a deprecation warning when the decorated function is called.
    The warning includes version info, reason, and suggested alternative.
    
    Args:
        version: Version in which the function was deprecated
        reason: Why the function is deprecated
        removal_version: Version in which the function will be removed
        alternative: Name of the function to use instead
        category: Warning category (default: DeprecationWarning)
        
    Returns:
        Decorated function that emits warning on call
        
    Example:
        @deprecated(
            version="2.1.0",
            reason="Slow implementation",
            removal_version="3.0.0",
            alternative="fast_predict()"
        )
        def slow_predict(user_id):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Get qualified name
        module = func.__module__ or ''
        qualname = func.__qualname__ or func.__name__
        full_name = f"{module}.{qualname}" if module else qualname
        
        # Create deprecation info
        info = DeprecationInfo(
            name=full_name,
            version=version,
            reason=reason,
            removal_version=removal_version,
            alternative=alternative
        )
        
        # Register deprecation
        _deprecation_registry[full_name] = info
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Emit warning with proper stack level
            warnings.warn(
                info.to_message(),
                category=category,
                stacklevel=2
            )
            return func(*args, **kwargs)
        
        # Add deprecation info to docstring
        original_doc = func.__doc__ or ""
        deprecation_note = f"""
.. deprecated:: {version}
   {info.to_message()}

"""
        wrapper.__doc__ = deprecation_note + original_doc
        
        return wrapper
    
    return decorator


def deprecated_parameter(
    parameter: str,
    version: str,
    reason: str = "",
    removal_version: Optional[str] = None,
    default_value: Any = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to mark specific function parameters as deprecated.
    
    Emits a warning when the deprecated parameter is used.
    
    Args:
        parameter: Name of the deprecated parameter
        version: Version in which the parameter was deprecated
        reason: Why the parameter is deprecated
        removal_version: Version when parameter will be removed
        default_value: Default value to use if parameter is provided
        
    Example:
        @deprecated_parameter(
            parameter="use_cache",
            version="2.1.0",
            reason="Caching is now automatic"
        )
        def get_recommendations(user_id, use_cache=None):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        sig = inspect.signature(func)
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Check if deprecated parameter is provided
            if parameter in kwargs:
                msg_parts = [
                    f"Parameter '{parameter}' of '{func.__name__}' "
                    f"is deprecated since version {version}"
                ]
                if reason:
                    msg_parts.append(f": {reason}")
                if removal_version:
                    msg_parts.append(f". Will be removed in version {removal_version}")
                
                warnings.warn(
                    "".join(msg_parts),
                    DeprecationWarning,
                    stacklevel=2
                )
                
                # Optionally replace with default value
                if default_value is not None:
                    kwargs[parameter] = default_value
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def deprecated_class(
    version: str,
    reason: str = "",
    removal_version: Optional[str] = None,
    alternative: Optional[str] = None
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to mark classes as deprecated.
    
    Emits a warning when the class is instantiated.
    
    Args:
        version: Version in which the class was deprecated
        reason: Why the class is deprecated
        removal_version: Version when class will be removed
        alternative: Name of the class to use instead
        
    Example:
        @deprecated_class(
            version="2.0.0",
            alternative="NewRecommender"
        )
        class OldRecommender:
            ...
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Get full class name
        full_name = f"{cls.__module__}.{cls.__qualname__}"
        
        # Create deprecation info
        info = DeprecationInfo(
            name=full_name,
            version=version,
            reason=reason,
            removal_version=removal_version,
            alternative=alternative
        )
        
        # Register deprecation
        _deprecation_registry[full_name] = info
        
        original_init = cls.__init__
        
        @functools.wraps(original_init)
        def new_init(self, *args: Any, **kwargs: Any) -> None:
            warnings.warn(
                info.to_message(),
                DeprecationWarning,
                stacklevel=2
            )
            original_init(self, *args, **kwargs)
        
        cls.__init__ = new_init
        
        # Add deprecation to docstring
        original_doc = cls.__doc__ or ""
        deprecation_note = f"""
.. deprecated:: {version}
   {info.to_message()}

"""
        cls.__doc__ = deprecation_note + original_doc
        
        return cls
    
    return decorator


def deprecation_warning(
    message: str,
    version: str,
    removal_version: Optional[str] = None,
    stacklevel: int = 2
) -> None:
    """
    Emit a deprecation warning with consistent formatting.
    
    Use this for one-off deprecation warnings that don't fit the decorator pattern.
    
    Args:
        message: Warning message
        version: Version when feature was deprecated
        removal_version: Version when feature will be removed
        stacklevel: Stack level for warning (default: 2)
        
    Example:
        if old_parameter is not None:
            deprecation_warning(
                "The 'old_parameter' argument is deprecated",
                version="2.1.0",
                removal_version="3.0.0"
            )
    """
    full_message = f"[Deprecated since v{version}] {message}"
    if removal_version:
        full_message += f" Will be removed in v{removal_version}."
    
    warnings.warn(full_message, DeprecationWarning, stacklevel=stacklevel)


class DeprecatedAlias:
    """
    Descriptor for creating deprecated attribute aliases.
    
    Use when renaming attributes/methods but maintaining backward compatibility.
    
    Example:
        class Model:
            new_name = "value"
            old_name = DeprecatedAlias("new_name", "2.0.0")
            
        m = Model()
        m.old_name  # Warns and returns m.new_name
    """
    
    def __init__(
        self,
        new_name: str,
        version: str,
        removal_version: Optional[str] = None
    ):
        self.new_name = new_name
        self.version = version
        self.removal_version = removal_version
    
    def __set_name__(self, owner: Type, name: str) -> None:
        self.old_name = name
        self.owner = owner
    
    def __get__(self, obj: Any, objtype: Optional[Type] = None) -> Any:
        if obj is None:
            return self
        
        msg = (
            f"'{self.old_name}' is deprecated since v{self.version}, "
            f"use '{self.new_name}' instead"
        )
        if self.removal_version:
            msg += f". Will be removed in v{self.removal_version}"
        
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return getattr(obj, self.new_name)
    
    def __set__(self, obj: Any, value: Any) -> None:
        msg = (
            f"'{self.old_name}' is deprecated since v{self.version}, "
            f"use '{self.new_name}' instead"
        )
        if self.removal_version:
            msg += f". Will be removed in v{self.removal_version}"
        
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        setattr(obj, self.new_name, value)


# =============================================================================
# CineMatch-Specific Deprecations
# =============================================================================

# List of deprecated items in CineMatch V2.1.6
# These will emit warnings when used

CINEMATCH_DEPRECATIONS = {
    # Legacy algorithm names
    "svd": {
        "deprecated_in": "2.1.0",
        "removal_in": "3.0.0",
        "alternative": "svd_sklearn",
        "reason": "Legacy algorithm, use svd_sklearn for better performance"
    },
    # Old function names
    "get_recommendations_v1": {
        "deprecated_in": "2.0.0",
        "removal_in": "2.2.0",
        "alternative": "get_recommendations",
        "reason": "V1 API replaced by unified recommendation interface"
    },
    # Old session state pattern
    "st.session_state['model']": {
        "deprecated_in": "2.1.6",
        "removal_in": "3.0.0",
        "alternative": "ModelStateManager",
        "reason": "Direct session_state access doesn't scale horizontally"
    }
}


def check_for_deprecated_usage(identifier: str) -> None:
    """
    Check if an identifier is deprecated and emit warning.
    
    Call this when encountering potentially deprecated identifiers.
    
    Args:
        identifier: The identifier to check (e.g., algorithm name, function name)
    """
    if identifier in CINEMATCH_DEPRECATIONS:
        info = CINEMATCH_DEPRECATIONS[identifier]
        msg = f"'{identifier}' is deprecated since v{info['deprecated_in']}"
        if info.get('reason'):
            msg += f": {info['reason']}"
        if info.get('alternative'):
            msg += f". Use '{info['alternative']}' instead"
        if info.get('removal_in'):
            msg += f". Will be removed in v{info['removal_in']}"
        
        warnings.warn(msg, DeprecationWarning, stacklevel=3)
