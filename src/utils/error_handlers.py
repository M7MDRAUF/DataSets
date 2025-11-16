"""
Unified Error Handling Module for CineMatch V2.1

This module provides consistent error handling, validation, and debugging
utilities across all recommendation algorithms and UI components.
"""

import logging
import traceback
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union
from functools import wraps
import pandas as pd
import numpy as np
from contextlib import contextmanager
import time

# Type variable for generic function returns
T = TypeVar('T')

# Configure logger
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for data validation failures"""
    pass


class ModelError(Exception):
    """Custom exception for model-related errors"""
    pass


def safe_execute(
    func: Callable[..., T],
    *args,
    fallback_value: Optional[T] = None,
    error_message: str = "Operation failed",
    raise_on_error: bool = False,
    log_level: str = "error",
    **kwargs
) -> Tuple[bool, Optional[T], Optional[str]]:
    """
    Safely execute a function with comprehensive error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments for function
        fallback_value: Value to return if function fails
        error_message: Custom error message prefix
        raise_on_error: Whether to re-raise exceptions
        log_level: Logging level for errors ('debug', 'info', 'warning', 'error')
        **kwargs: Keyword arguments for function
        
    Returns:
        Tuple of (success: bool, result: T, error_msg: str)
        
    Example:
        >>> success, result, error = safe_execute(
        ...     risky_function,
        ...     arg1, arg2,
        ...     fallback_value=[],
        ...     error_message="Failed to process data"
        ... )
        >>> if success:
        ...     print(f"Result: {result}")
        ... else:
        ...     print(f"Error: {error}")
    """
    try:
        result = func(*args, **kwargs)
        return True, result, None
        
    except Exception as e:
        error_msg = f"{error_message}: {str(e)}"
        error_trace = traceback.format_exc()
        
        # Log at appropriate level
        log_func = getattr(logger, log_level.lower(), logger.error)
        log_func(f"{error_msg}\n{error_trace}")
        
        if raise_on_error:
            raise
            
        return False, fallback_value, error_msg


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[list] = None,
    min_rows: int = 0,
    max_rows: Optional[int] = None,
    allow_empty: bool = False,
    column_types: Optional[Dict[str, type]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        max_rows: Maximum number of rows allowed
        allow_empty: Whether to allow empty DataFrame
        column_types: Dict mapping column names to expected types
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
        
    Example:
        >>> is_valid, error = validate_dataframe(
        ...     ratings_df,
        ...     required_columns=['userId', 'movieId', 'rating'],
        ...     min_rows=1,
        ...     column_types={'userId': int, 'rating': float}
        ... )
    """
    # Check if DataFrame exists
    if df is None:
        return False, "DataFrame is None"
    
    if not isinstance(df, pd.DataFrame):
        return False, f"Expected DataFrame, got {type(df).__name__}"
    
    # Check if empty
    if len(df) == 0 and not allow_empty:
        return False, "DataFrame is empty"
    
    # Check row count
    if len(df) < min_rows:
        return False, f"DataFrame has {len(df)} rows, minimum {min_rows} required"
    
    if max_rows is not None and len(df) > max_rows:
        return False, f"DataFrame has {len(df)} rows, maximum {max_rows} allowed"
    
    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
    
    # Check column types
    if column_types:
        type_errors = []
        for col, expected_type in column_types.items():
            if col not in df.columns:
                type_errors.append(f"Column '{col}' not found")
                continue
            
            # Check if column can be converted to expected type
            try:
                if expected_type == int:
                    pd.to_numeric(df[col], errors='coerce').astype('Int64')
                elif expected_type == float:
                    pd.to_numeric(df[col], errors='coerce')
                elif expected_type == str:
                    df[col].astype(str)
            except Exception as e:
                type_errors.append(f"Column '{col}' cannot be converted to {expected_type.__name__}: {e}")
        
        if type_errors:
            return False, "; ".join(type_errors)
    
    return True, None


def handle_model_error(
    error: Exception,
    context: str = "",
    user_id: Optional[int] = None,
    movie_id: Optional[int] = None,
    algorithm: Optional[str] = None
) -> Dict[str, Any]:
    """
    Handle model prediction/recommendation errors with context.
    
    Args:
        error: The exception that occurred
        context: Description of what was being done
        user_id: User ID if applicable
        movie_id: Movie ID if applicable
        algorithm: Algorithm name if applicable
        
    Returns:
        Dict with error details and suggested fallback
        
    Example:
        >>> try:
        ...     prediction = model.predict(user_id, movie_id)
        ... except Exception as e:
        ...     error_info = handle_model_error(
        ...         e, 
        ...         context="Rating prediction",
        ...         user_id=123,
        ...         movie_id=456,
        ...         algorithm="SVD"
        ...     )
        ...     print(error_info['message'])
    """
    error_details = {
        'error_type': type(error).__name__,
        'message': str(error),
        'context': context,
        'user_id': user_id,
        'movie_id': movie_id,
        'algorithm': algorithm,
        'traceback': traceback.format_exc(),
        'suggested_fallback': None
    }
    
    # Determine suggested fallback based on error type
    if isinstance(error, KeyError):
        error_details['suggested_fallback'] = 'use_global_average'
        error_details['user_message'] = "User or movie not found in training data"
    elif isinstance(error, ValueError):
        error_details['suggested_fallback'] = 'return_default'
        error_details['user_message'] = "Invalid input parameters"
    elif isinstance(error, MemoryError):
        error_details['suggested_fallback'] = 'reduce_sample_size'
        error_details['user_message'] = "Insufficient memory for operation"
    else:
        error_details['suggested_fallback'] = 'retry_with_fallback_algorithm'
        error_details['user_message'] = "Unexpected error occurred"
    
    # Log the error
    logger.error(
        f"Model Error in {context} | "
        f"Algorithm: {algorithm} | "
        f"User: {user_id} | Movie: {movie_id} | "
        f"Error: {error_details['error_type']} - {error_details['message']}"
    )
    
    return error_details


@contextmanager
def track_performance(operation_name: str, log_threshold_ms: float = 1000.0):
    """
    Context manager to track and log performance of operations.
    
    Args:
        operation_name: Name of the operation being tracked
        log_threshold_ms: Only log if operation takes longer than this (milliseconds)
        
    Example:
        >>> with track_performance("Load recommendations", log_threshold_ms=500):
        ...     recommendations = algorithm.recommend(user_id, n=10)
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms >= log_threshold_ms:
            logger.warning(
                f"Performance Warning: {operation_name} took {elapsed_ms:.2f}ms "
                f"(threshold: {log_threshold_ms}ms)"
            )
        else:
            logger.debug(f"{operation_name} completed in {elapsed_ms:.2f}ms")


def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """
    Safely divide two numbers, returning fallback on division by zero.
    
    Args:
        numerator: Number to divide
        denominator: Number to divide by
        fallback: Value to return if denominator is zero
        
    Returns:
        Result of division or fallback value
    """
    try:
        if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            return fallback
        result = numerator / denominator
        if np.isnan(result) or np.isinf(result):
            return fallback
        return result
    except:
        return fallback


def sanitize_prediction(
    prediction: float,
    min_value: float = 0.5,
    max_value: float = 5.0,
    default_value: float = 3.0
) -> float:
    """
    Sanitize prediction values to valid range.
    
    Args:
        prediction: Raw prediction value
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        default_value: Default value if prediction is invalid
        
    Returns:
        Sanitized prediction value
    """
    try:
        if prediction is None or np.isnan(prediction) or np.isinf(prediction):
            return default_value
        
        # Clip to valid range
        return float(np.clip(prediction, min_value, max_value))
    except:
        return default_value


def validate_user_id(user_id: Any, valid_users: Optional[set] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate user ID format and existence.
    
    Args:
        user_id: User ID to validate
        valid_users: Set of valid user IDs (optional)
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    # Check type
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return False, f"Invalid user_id type: {type(user_id).__name__}"
    
    # Check positive
    if user_id <= 0:
        return False, f"User ID must be positive, got {user_id}"
    
    # Check existence
    if valid_users is not None and user_id not in valid_users:
        return False, f"User ID {user_id} not found in dataset"
    
    return True, None


def validate_movie_id(movie_id: Any, valid_movies: Optional[set] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate movie ID format and existence.
    
    Args:
        movie_id: Movie ID to validate
        valid_movies: Set of valid movie IDs (optional)
        
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    # Check type
    try:
        movie_id = int(movie_id)
    except (TypeError, ValueError):
        return False, f"Invalid movie_id type: {type(movie_id).__name__}"
    
    # Check positive
    if movie_id <= 0:
        return False, f"Movie ID must be positive, got {movie_id}"
    
    # Check existence
    if valid_movies is not None and movie_id not in valid_movies:
        return False, f"Movie ID {movie_id} not found in dataset"
    
    return True, None


def robust_decorator(fallback_value: Any = None):
    """
    Decorator to add automatic error handling to functions.
    
    Args:
        fallback_value: Value to return on error
        
    Example:
        >>> @robust_decorator(fallback_value=[])
        ... def get_recommendations(user_id):
        ...     # Function that might fail
        ...     return algorithm.recommend(user_id)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}"
                )
                return fallback_value
        return wrapper
    return decorator


# Export main functions
__all__ = [
    'safe_execute',
    'validate_dataframe',
    'handle_model_error',
    'track_performance',
    'safe_divide',
    'sanitize_prediction',
    'validate_user_id',
    'validate_movie_id',
    'robust_decorator',
    'ValidationError',
    'ModelError'
]
