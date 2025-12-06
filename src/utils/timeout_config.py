"""
CineMatch V2.1.6 - Request Timeout Configuration

Provides centralized timeout configuration for all external service calls
to prevent hanging requests and resource exhaustion.

Features:
- Configurable timeouts via environment variables
- Sensible defaults for different service types
- Timeout decorator for easy application
- Context manager for timeout enforcement

Author: CineMatch Development Team
Date: December 2025
"""

import functools
import logging
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class TimeoutConfig:
    """
    Centralized timeout configuration for CineMatch.
    
    All timeouts are in seconds. Configure via environment variables
    with CINEMATCH_ prefix.
    """
    
    # HTTP/API timeouts
    http_connect_timeout: float = 5.0  # Time to establish connection
    http_read_timeout: float = 30.0    # Time to receive response
    api_request_timeout: float = 60.0  # Total API request timeout
    
    # Database timeouts
    db_connect_timeout: float = 10.0   # Database connection timeout
    db_query_timeout: float = 30.0     # Query execution timeout
    db_pool_timeout: float = 30.0      # Wait for pool connection
    
    # Cache (Redis) timeouts
    redis_connect_timeout: float = 5.0
    redis_socket_timeout: float = 5.0
    redis_operation_timeout: float = 10.0
    
    # External services
    webhook_timeout: float = 10.0      # Webhook delivery timeout
    tracing_export_timeout: float = 10.0  # Trace export timeout
    metrics_push_timeout: float = 5.0  # Metrics push timeout
    
    # Model/ML operations
    model_load_timeout: float = 300.0  # Model loading (large models)
    model_predict_timeout: float = 30.0  # Single prediction timeout
    batch_recommendation_timeout: float = 60.0  # Batch processing
    
    # Data loading
    data_load_timeout: float = 120.0   # Large data file loading
    
    @classmethod
    def from_env(cls) -> 'TimeoutConfig':
        """
        Create configuration from environment variables.
        
        All environment variables should be prefixed with CINEMATCH_.
        Example: CINEMATCH_HTTP_CONNECT_TIMEOUT=10
        """
        def get_float(key: str, default: float) -> float:
            value = os.getenv(f'CINEMATCH_{key.upper()}', '')
            try:
                return float(value) if value else default
            except ValueError:
                logger.warning(f"Invalid timeout value for {key}: {value}, using default {default}")
                return default
        
        return cls(
            http_connect_timeout=get_float('HTTP_CONNECT_TIMEOUT', 5.0),
            http_read_timeout=get_float('HTTP_READ_TIMEOUT', 30.0),
            api_request_timeout=get_float('API_REQUEST_TIMEOUT', 60.0),
            db_connect_timeout=get_float('DB_CONNECT_TIMEOUT', 10.0),
            db_query_timeout=get_float('DB_QUERY_TIMEOUT', 30.0),
            db_pool_timeout=get_float('DB_POOL_TIMEOUT', 30.0),
            redis_connect_timeout=get_float('REDIS_CONNECT_TIMEOUT', 5.0),
            redis_socket_timeout=get_float('REDIS_SOCKET_TIMEOUT', 5.0),
            redis_operation_timeout=get_float('REDIS_OPERATION_TIMEOUT', 10.0),
            webhook_timeout=get_float('WEBHOOK_TIMEOUT', 10.0),
            tracing_export_timeout=get_float('TRACING_EXPORT_TIMEOUT', 10.0),
            metrics_push_timeout=get_float('METRICS_PUSH_TIMEOUT', 5.0),
            model_load_timeout=get_float('MODEL_LOAD_TIMEOUT', 300.0),
            model_predict_timeout=get_float('MODEL_PREDICT_TIMEOUT', 30.0),
            batch_recommendation_timeout=get_float('BATCH_RECOMMENDATION_TIMEOUT', 60.0),
            data_load_timeout=get_float('DATA_LOAD_TIMEOUT', 120.0),
        )
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'http_connect_timeout': self.http_connect_timeout,
            'http_read_timeout': self.http_read_timeout,
            'api_request_timeout': self.api_request_timeout,
            'db_connect_timeout': self.db_connect_timeout,
            'db_query_timeout': self.db_query_timeout,
            'db_pool_timeout': self.db_pool_timeout,
            'redis_connect_timeout': self.redis_connect_timeout,
            'redis_socket_timeout': self.redis_socket_timeout,
            'redis_operation_timeout': self.redis_operation_timeout,
            'webhook_timeout': self.webhook_timeout,
            'tracing_export_timeout': self.tracing_export_timeout,
            'metrics_push_timeout': self.metrics_push_timeout,
            'model_load_timeout': self.model_load_timeout,
            'model_predict_timeout': self.model_predict_timeout,
            'batch_recommendation_timeout': self.batch_recommendation_timeout,
            'data_load_timeout': self.data_load_timeout,
        }


# Global timeout configuration (singleton)
_timeout_config: Optional[TimeoutConfig] = None


def get_timeout_config() -> TimeoutConfig:
    """Get the global timeout configuration."""
    global _timeout_config
    if _timeout_config is None:
        _timeout_config = TimeoutConfig.from_env()
    return _timeout_config


def set_timeout_config(config: TimeoutConfig) -> None:
    """Set the global timeout configuration."""
    global _timeout_config
    _timeout_config = config


class TimeoutError(Exception):
    """Raised when an operation times out."""
    def __init__(self, message: str, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        super().__init__(f"{message} (timeout: {timeout_seconds}s)")


def timeout(
    seconds: Optional[float] = None,
    timeout_type: Optional[str] = None,
    error_message: str = "Operation timed out"
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to enforce timeout on a function.
    
    Uses threading-based timeout (works on Windows and Linux).
    
    Args:
        seconds: Timeout in seconds. If None, uses timeout_type.
        timeout_type: Type of timeout from TimeoutConfig (e.g., 'api_request_timeout').
        error_message: Custom error message on timeout.
        
    Returns:
        Decorated function that raises TimeoutError if execution exceeds timeout.
        
    Example:
        @timeout(seconds=30)
        def slow_function():
            ...
            
        @timeout(timeout_type='model_predict_timeout')
        def predict():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Determine timeout value
            timeout_seconds = seconds
            if timeout_seconds is None and timeout_type:
                config = get_timeout_config()
                timeout_seconds = getattr(config, timeout_type, 30.0)
            elif timeout_seconds is None:
                timeout_seconds = 30.0  # Default
            
            # Use thread pool for timeout enforcement
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=timeout_seconds)
                except FuturesTimeoutError:
                    raise TimeoutError(
                        f"{error_message}: {func.__name__}",
                        timeout_seconds
                    )
        
        return wrapper
    return decorator


@contextmanager
def timeout_context(
    seconds: float,
    error_message: str = "Operation timed out"
):
    """
    Context manager for enforcing timeout on a block of code.
    
    Note: Only works on Unix systems with signal support.
    On Windows, consider using timeout decorator instead.
    
    Args:
        seconds: Timeout in seconds
        error_message: Error message on timeout
        
    Yields:
        None
        
    Example:
        with timeout_context(30, "Data loading timed out"):
            data = load_large_file()
    """
    # Check if signals are supported (Unix only)
    if hasattr(signal, 'SIGALRM'):
        def handler(signum, frame):
            raise TimeoutError(error_message, seconds)
        
        # Set the signal handler
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(int(seconds))
        
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # On Windows, we can't use signals - just yield
        # User should use the timeout decorator instead
        logger.debug("timeout_context not supported on Windows, proceeding without timeout")
        yield


class TimeoutEnforcer:
    """
    Utility class for enforcing timeouts on operations.
    
    Provides both synchronous and asynchronous timeout enforcement
    using threading.
    
    Example:
        enforcer = TimeoutEnforcer(timeout_seconds=30)
        result = enforcer.run(slow_function, arg1, arg2)
    """
    
    def __init__(self, timeout_seconds: float = 30.0):
        """
        Initialize timeout enforcer.
        
        Args:
            timeout_seconds: Default timeout in seconds
        """
        self.timeout_seconds = timeout_seconds
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    def run(
        self,
        func: Callable[..., T],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> T:
        """
        Run function with timeout enforcement.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            timeout: Override default timeout
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            TimeoutError: If execution exceeds timeout
        """
        effective_timeout = timeout or self.timeout_seconds
        future = self._executor.submit(func, *args, **kwargs)
        
        try:
            return future.result(timeout=effective_timeout)
        except FuturesTimeoutError:
            raise TimeoutError(
                f"Function {func.__name__} timed out",
                effective_timeout
            )
    
    def run_with_fallback(
        self,
        func: Callable[..., T],
        fallback: Callable[[], T],
        *args: Any,
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> T:
        """
        Run function with timeout and fallback on timeout.
        
        Args:
            func: Primary function to execute
            fallback: Fallback function if timeout occurs
            *args: Positional arguments for primary function
            timeout: Override default timeout
            **kwargs: Keyword arguments for primary function
            
        Returns:
            Result from primary function or fallback
        """
        try:
            return self.run(func, *args, timeout=timeout, **kwargs)
        except TimeoutError:
            logger.warning(f"Timeout calling {func.__name__}, using fallback")
            return fallback()
    
    def shutdown(self) -> None:
        """Shutdown the executor."""
        self._executor.shutdown(wait=False)
    
    def __enter__(self) -> 'TimeoutEnforcer':
        return self
    
    def __exit__(self, *args) -> None:
        self.shutdown()


# Pre-configured timeout enforcers for common use cases
def get_http_enforcer() -> TimeoutEnforcer:
    """Get timeout enforcer configured for HTTP operations."""
    config = get_timeout_config()
    return TimeoutEnforcer(timeout_seconds=config.http_read_timeout)


def get_db_enforcer() -> TimeoutEnforcer:
    """Get timeout enforcer configured for database operations."""
    config = get_timeout_config()
    return TimeoutEnforcer(timeout_seconds=config.db_query_timeout)


def get_model_enforcer() -> TimeoutEnforcer:
    """Get timeout enforcer configured for model operations."""
    config = get_timeout_config()
    return TimeoutEnforcer(timeout_seconds=config.model_predict_timeout)


# HTTP tuple format (connect_timeout, read_timeout) for requests library
def get_http_timeout_tuple() -> tuple[float, float]:
    """
    Get timeout tuple for use with requests library.
    
    Returns:
        Tuple of (connect_timeout, read_timeout)
        
    Example:
        response = requests.get(url, timeout=get_http_timeout_tuple())
    """
    config = get_timeout_config()
    return (config.http_connect_timeout, config.http_read_timeout)
