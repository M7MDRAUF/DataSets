"""
CineMatch V2.1.6 - Input Validation and Sanitization

Security module for validating and sanitizing all user inputs.
Prevents injection attacks and ensures data integrity.

Author: CineMatch Development Team
Date: December 5, 2025

Security Features:
    - User ID validation (integer bounds checking)
    - Movie ID validation (integer bounds checking)  
    - Search query sanitization (XSS prevention)
    - Path traversal prevention
    - SQL-like injection pattern detection
"""

import re
import html
from typing import Union, Optional, Tuple, List
import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================

# Maximum allowed lengths
MAX_SEARCH_QUERY_LENGTH = 500
MAX_USER_ID = 10_000_000  # 10 million - reasonable upper bound
MAX_MOVIE_ID = 10_000_000
MIN_ID = 1

# Dangerous patterns that could indicate injection attempts
DANGEROUS_PATTERNS = [
    r'<script',           # XSS script tags
    r'javascript:',       # JavaScript URI
    r'on\w+\s*=',        # Event handlers (onclick, onerror, etc.)
    r'data:text/html',    # Data URI XSS
    r'vbscript:',         # VBScript URI
    r'\x00',              # Null bytes
    r'\.\./',             # Path traversal
    r'\.\.\\',            # Windows path traversal
    r';\s*DROP\s+TABLE',  # SQL injection
    r';\s*DELETE\s+FROM', # SQL injection
    r"'\s*OR\s+'1'\s*=",  # SQL injection
    r'UNION\s+SELECT',    # SQL injection
    r'<iframe',           # Iframe injection
    r'<embed',            # Embed injection
    r'<object',           # Object injection
]

# Compiled regex for efficiency
DANGEROUS_REGEX = re.compile(
    '|'.join(DANGEROUS_PATTERNS), 
    re.IGNORECASE
)


# =============================================================================
# INPUT VALIDATION FUNCTIONS
# =============================================================================

class InputValidationError(ValueError):
    """Raised when input validation fails."""
    pass


def validate_user_id(
    user_id: Union[int, float, str],
    ratings_df: Optional[pd.DataFrame] = None,
    strict: bool = True
) -> Tuple[bool, int, str]:
    """
    Validate a user ID input.
    
    Args:
        user_id: The user ID to validate
        ratings_df: Optional DataFrame to check if user exists
        strict: If True, require user to exist in ratings_df
        
    Returns:
        Tuple of (is_valid, sanitized_id, message)
    """
    # Type conversion
    try:
        if isinstance(user_id, str):
            user_id = user_id.strip()
            if not user_id:
                return False, 0, "User ID cannot be empty"
            user_id = int(float(user_id))
        elif isinstance(user_id, float):
            user_id = int(user_id)
        elif not isinstance(user_id, int):
            return False, 0, f"User ID must be a number, got {type(user_id).__name__}"
    except (ValueError, TypeError):
        return False, 0, "User ID must be a valid integer"
    
    # Bounds checking
    if user_id < MIN_ID:
        return False, 0, f"User ID must be at least {MIN_ID}"
    if user_id > MAX_USER_ID:
        return False, 0, f"User ID cannot exceed {MAX_USER_ID:,}"
    
    # Check existence in dataset if provided
    if ratings_df is not None and strict:
        if 'userId' in ratings_df.columns:
            if user_id not in ratings_df['userId'].values:
                return False, user_id, f"User ID {user_id} not found in dataset"
    
    return True, user_id, "Valid"


def validate_movie_id(
    movie_id: Union[int, float, str],
    movies_df: Optional[pd.DataFrame] = None,
    strict: bool = True
) -> Tuple[bool, int, str]:
    """
    Validate a movie ID input.
    
    Args:
        movie_id: The movie ID to validate
        movies_df: Optional DataFrame to check if movie exists
        strict: If True, require movie to exist in movies_df
        
    Returns:
        Tuple of (is_valid, sanitized_id, message)
    """
    # Type conversion
    try:
        if isinstance(movie_id, str):
            movie_id = movie_id.strip()
            if not movie_id:
                return False, 0, "Movie ID cannot be empty"
            movie_id = int(float(movie_id))
        elif isinstance(movie_id, float):
            movie_id = int(movie_id)
        elif not isinstance(movie_id, int):
            return False, 0, f"Movie ID must be a number, got {type(movie_id).__name__}"
    except (ValueError, TypeError):
        return False, 0, "Movie ID must be a valid integer"
    
    # Bounds checking
    if movie_id < MIN_ID:
        return False, 0, f"Movie ID must be at least {MIN_ID}"
    if movie_id > MAX_MOVIE_ID:
        return False, 0, f"Movie ID cannot exceed {MAX_MOVIE_ID:,}"
    
    # Check existence in dataset if provided
    if movies_df is not None and strict:
        if 'movieId' in movies_df.columns:
            if movie_id not in movies_df['movieId'].values:
                return False, movie_id, f"Movie ID {movie_id} not found in dataset"
    
    return True, movie_id, "Valid"


def sanitize_search_query(
    query: str,
    max_length: int = MAX_SEARCH_QUERY_LENGTH,
    allow_special_chars: bool = False
) -> Tuple[bool, str, str]:
    """
    Sanitize a search query string.
    
    Args:
        query: The search query to sanitize
        max_length: Maximum allowed length
        allow_special_chars: If False, remove special characters
        
    Returns:
        Tuple of (is_safe, sanitized_query, message)
    """
    if not isinstance(query, str):
        return False, "", "Search query must be a string"
    
    # Basic cleanup
    query = query.strip()
    
    if not query:
        return False, "", "Search query cannot be empty"
    
    # Length check
    if len(query) > max_length:
        return False, "", f"Search query too long (max {max_length} characters)"
    
    # Check for dangerous patterns
    if DANGEROUS_REGEX.search(query):
        return False, "", "Search query contains potentially dangerous content"
    
    # HTML escape to prevent XSS
    sanitized = html.escape(query)
    
    # Optionally strip special characters
    if not allow_special_chars:
        # Keep alphanumeric, spaces, and common punctuation
        sanitized = re.sub(r'[^\w\s\-\.\,\!\?\'\"\(\)]', '', sanitized)
    
    return True, sanitized, "Valid"


def validate_rating(rating: Union[int, float, str]) -> Tuple[bool, float, str]:
    """
    Validate a movie rating value.
    
    Args:
        rating: The rating value to validate (0.5-5.0 scale)
        
    Returns:
        Tuple of (is_valid, sanitized_rating, message)
    """
    try:
        if isinstance(rating, str):
            rating = float(rating.strip())
        elif isinstance(rating, int):
            rating = float(rating)
        elif not isinstance(rating, float):
            return False, 0.0, f"Rating must be a number, got {type(rating).__name__}"
    except (ValueError, TypeError):
        return False, 0.0, "Rating must be a valid number"
    
    # MovieLens uses 0.5-5.0 scale in 0.5 increments
    if rating < 0.5 or rating > 5.0:
        return False, 0.0, "Rating must be between 0.5 and 5.0"
    
    # Snap to nearest 0.5
    sanitized = round(rating * 2) / 2
    
    return True, sanitized, "Valid"


def validate_num_recommendations(n: Union[int, str], max_n: int = 100) -> Tuple[bool, int, str]:
    """
    Validate number of recommendations requested.
    
    Args:
        n: Number of recommendations
        max_n: Maximum allowed
        
    Returns:
        Tuple of (is_valid, sanitized_n, message)
    """
    try:
        if isinstance(n, str):
            n = int(n.strip())
        elif isinstance(n, float):
            n = int(n)
        elif not isinstance(n, int):
            return False, 0, f"Number must be an integer, got {type(n).__name__}"
    except (ValueError, TypeError):
        return False, 0, "Number of recommendations must be a valid integer"
    
    if n < 1:
        return False, 0, "Must request at least 1 recommendation"
    if n > max_n:
        return False, max_n, f"Cannot request more than {max_n} recommendations"
    
    return True, n, "Valid"


# =============================================================================
# STREAMLIT INTEGRATION HELPERS
# =============================================================================

def safe_user_id_input(user_id_raw, ratings_df=None):
    """
    Validate user ID from Streamlit input.
    
    Returns validated user_id or raises InputValidationError.
    """
    is_valid, user_id, message = validate_user_id(user_id_raw, ratings_df)
    if not is_valid:
        raise InputValidationError(message)
    return user_id


def safe_movie_id_input(movie_id_raw, movies_df=None):
    """
    Validate movie ID from Streamlit input.
    
    Returns validated movie_id or raises InputValidationError.
    """
    is_valid, movie_id, message = validate_movie_id(movie_id_raw, movies_df)
    if not is_valid:
        raise InputValidationError(message)
    return movie_id


def safe_search_query(query_raw):
    """
    Sanitize search query from Streamlit input.
    
    Returns sanitized query or raises InputValidationError.
    """
    is_safe, query, message = sanitize_search_query(query_raw)
    if not is_safe:
        raise InputValidationError(message)
    return query


# =============================================================================
# BATCH VALIDATION
# =============================================================================

def validate_user_ids_batch(
    user_ids: List[Union[int, float, str]],
    ratings_df: Optional[pd.DataFrame] = None
) -> Tuple[List[int], List[Tuple[int, str]]]:
    """
    Validate a batch of user IDs.
    
    Returns:
        Tuple of (valid_ids, invalid_ids_with_reasons)
    """
    valid = []
    invalid = []
    
    for uid in user_ids:
        is_valid, sanitized, message = validate_user_id(uid, ratings_df)
        if is_valid:
            valid.append(sanitized)
        else:
            invalid.append((uid, message))
    
    return valid, invalid


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    # Self-test
    print("Testing input validation module...")
    
    # User ID tests
    assert validate_user_id(1)[0] == True
    assert validate_user_id(-1)[0] == False
    assert validate_user_id("123")[0] == True
    assert validate_user_id("abc")[0] == False
    assert validate_user_id(999999999999)[0] == False
    
    # Search query tests
    assert sanitize_search_query("Toy Story")[0] == True
    assert sanitize_search_query("<script>alert('xss')</script>")[0] == False
    assert sanitize_search_query("'; DROP TABLE movies;--")[0] == False
    assert sanitize_search_query("The Matrix (1999)")[0] == True
    
    # Rating tests
    assert validate_rating(4.5)[0] == True
    assert validate_rating(0)[0] == False
    assert validate_rating(6)[0] == False
    
    print("✓ All tests passed!")
