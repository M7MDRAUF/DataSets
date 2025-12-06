"""
CineMatch V2.1.6 - Data Processing Module

This module handles data loading, preprocessing, and integrity checks.
Implements NF-01: Data Integrity & Developer Alerting requirements.

Supports both CSV and Parquet formats with automatic detection.
Parquet is preferred for 70% storage reduction and 5x faster loads.

Circuit breaker pattern applied for graceful degradation on I/O failures.
Retry with exponential backoff for transient failures.

Author: CineMatch Team
Date: December 2025
"""

import logging
import os
import sys
from typing import Tuple, List, Optional, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path

# Configure module logger
logger = logging.getLogger(__name__)

# Import circuit breaker for resilient data loading
try:
    from src.reliability.circuit_breaker import circuit_breaker, CircuitOpenError
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False
    logger.warning("Circuit breaker not available - proceeding without fault tolerance")
    
    # Fallback no-op decorator
    def circuit_breaker(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class CircuitOpenError(Exception):
        pass

# Import retry with exponential backoff
try:
    from src.reliability.retry_mechanisms import retry, BackoffStrategy
    RETRY_AVAILABLE = True
except ImportError:
    RETRY_AVAILABLE = False
    logger.warning("Retry mechanism not available - proceeding without retry")
    
    # Fallback no-op decorator
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class BackoffStrategy:
        EXPONENTIAL = "exponential"


# Data paths
DATA_DIR = Path("data/ml-32m")
PROCESSED_DIR = Path("data/processed")

# Required files - supports both Parquet (preferred) and CSV (fallback)
REQUIRED_FILES_PARQUET = [
    "ratings.parquet",
    "movies.parquet",
    "links.parquet",
    "tags.parquet"
]
REQUIRED_FILES_CSV = [
    "ratings.csv",
    "movies.csv",
    "links.csv",
    "tags.csv"
]

# Optimized dtypes for memory efficiency
RATINGS_DTYPES: Dict[str, str] = {
    'userId': 'int32',
    'movieId': 'int32',
    'rating': 'float32',
    'timestamp': 'int32'  # Unix timestamps fit in int32 until 2038
}

MOVIES_DTYPES: Dict[str, str] = {
    'movieId': 'int32',
    'title': 'str',
    'genres': 'str'
}

LINKS_DTYPES: Dict[str, str] = {
    'movieId': 'int32',
    'imdbId': 'str',
    'tmdbId': 'float32'  # Float because of NaN values
}

TAGS_DTYPES: Dict[str, str] = {
    'userId': 'int32',
    'movieId': 'int32',
    'tag': 'str',
    'timestamp': 'int32'
}


def _detect_data_format() -> str:
    """
    Detect available data format (Parquet preferred over CSV).
    
    Returns:
        'parquet' if Parquet files exist, 'csv' otherwise
    """
    parquet_path = DATA_DIR / "ratings.parquet"
    if parquet_path.exists():
        return 'parquet'
    return 'csv'


def _get_required_files() -> List[str]:
    """Get list of required files based on detected format."""
    fmt = _detect_data_format()
    return REQUIRED_FILES_PARQUET if fmt == 'parquet' else REQUIRED_FILES_CSV


# For backward compatibility
REQUIRED_FILES = REQUIRED_FILES_CSV  # Will be updated at runtime


def check_data_integrity() -> Tuple[bool, List[str], Optional[str]]:
    """
    Validates the presence of all required dataset files.
    
    Implements NF-01: Data Integrity & Developer Alerting
    
    Checks for Parquet files first (preferred), falls back to CSV.
    
    Returns:
        Tuple of (success: bool, missing_files: List[str], error_message: Optional[str])
        - success: True if all files found, False otherwise
        - missing_files: List of missing filenames
        - error_message: Detailed error message with instructions if files missing
    
    Example:
        >>> success, missing, error = check_data_integrity()
        >>> if not success:
        ...     print(error)
        ...     sys.exit(1)
    """
    missing_files = []
    data_format = 'unknown'
    
    # Check if data directory exists
    if not DATA_DIR.exists():
        missing_files = REQUIRED_FILES_CSV
    else:
        # Check for Parquet files first (preferred)
        parquet_missing = []
        for filename in REQUIRED_FILES_PARQUET:
            file_path = DATA_DIR / filename
            if not file_path.exists():
                parquet_missing.append(filename)
        
        if not parquet_missing:
            # All Parquet files found
            data_format = 'parquet'
            logger.info("✓ Using Parquet format (optimized)")
        else:
            # Fall back to CSV check
            csv_missing = []
            for filename in REQUIRED_FILES_CSV:
                file_path = DATA_DIR / filename
                if not file_path.exists():
                    csv_missing.append(filename)
            
            if not csv_missing:
                data_format = 'csv'
                logger.info("ℹ Using CSV format (consider converting to Parquet)")
                logger.info("  Run: python scripts/convert_csv_to_parquet.py")
            else:
                missing_files = csv_missing
    
    # If files are missing, generate detailed error message
    if missing_files:
        absolute_path = DATA_DIR.absolute()
        error_message = f"""
╔══════════════════════════════════════════════════════════════════╗
║                   ❌ DATA INTEGRITY CHECK FAILED                 ║
╚══════════════════════════════════════════════════════════════════╝

Missing Dataset Files:
{chr(10).join(f"  • {file}" for file in missing_files)}

Expected Location:
  {absolute_path}

╔══════════════════════════════════════════════════════════════════╗
║                    🔧 ACTION REQUIRED                             ║
╚══════════════════════════════════════════════════════════════════╝

To fix this issue, please follow these steps:

1. Download the MovieLens 32M dataset:
   🔗 http://files.grouplens.org/datasets/movielens/ml-32m.zip
   
   Alternative link:
   🔗 https://grouplens.org/datasets/movielens/latest/

2. Extract the downloaded ZIP file

3. Copy ALL CSV files to the following directory:
   📁 {absolute_path}

4. Verify the following files are present:
   ✓ ratings.csv  (32 million ratings)
   ✓ movies.csv   (Movie catalog with titles and genres)
   ✓ links.csv    (IMDb and TMDb IDs)
   ✓ tags.csv     (User-generated tags)

5. Restart the application

╔══════════════════════════════════════════════════════════════════╗
║                      ℹ️ IMPORTANT NOTES                          ║
╚══════════════════════════════════════════════════════════════════╝

• Dataset size: ~600 MB (compressed), ~1.5 GB (extracted)
• Download time: 5-15 minutes (depending on connection)
• This is a one-time setup requirement

For detailed instructions, see README.md

        """
        return False, missing_files, error_message
    
    # All files found - success!
    return True, [], None


def _ratings_fallback(sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Fallback function when ratings cannot be loaded.
    Returns empty DataFrame with correct schema.
    """
    logger.warning("Using fallback empty ratings DataFrame")
    return pd.DataFrame({
        'userId': pd.Series(dtype='int32'),
        'movieId': pd.Series(dtype='int32'),
        'rating': pd.Series(dtype='float32'),
        'timestamp': pd.Series(dtype='int32')
    })


@retry(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    backoff=BackoffStrategy.EXPONENTIAL,
    exceptions={IOError, OSError, pd.errors.ParserError},
    jitter=True
)
def _load_ratings_with_retry(sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Internal function for loading ratings with retry on transient failures.
    
    Retries up to 3 times with exponential backoff (1s, 2s, 4s) for:
    - IOError: Disk/network issues
    - OSError: File system issues  
    - ParserError: Corrupted file reads
    """
    # Prefer Parquet, fall back to CSV
    parquet_path = DATA_DIR / "ratings.parquet"
    csv_path = DATA_DIR / "ratings.csv"
    
    use_parquet = parquet_path.exists()
    file_path = parquet_path if use_parquet else csv_path
    
    logger.info(f"Loading ratings from {file_path}...")
    
    if sample_size:
        logger.info(f"  → Sampling {sample_size:,} ratings for faster processing")
    
    if use_parquet:
        # Parquet: Fast columnar loading
        df = pd.read_parquet(file_path)
        
        # Ensure correct dtypes (Parquet preserves them, but verify)
        for col, dtype in RATINGS_DTYPES.items():
            if col in df.columns and df[col].dtype.name != dtype:
                df[col] = df[col].astype(dtype)
    else:
        # CSV: Load with dtype optimization
        df = pd.read_csv(file_path, dtype=RATINGS_DTYPES)
    
    # Sample if requested
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    logger.info(f"  [OK] Loaded {len(df):,} ratings")
    return df


@circuit_breaker(
    name="load_ratings",
    failure_threshold=3,
    timeout=60.0,
    fallback=_ratings_fallback,
    excluded_exceptions={FileNotFoundError}  # Don't count config issues
)
def load_ratings(sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load the ratings dataset from Parquet (preferred) or CSV.
    
    Protected by:
    1. Retry with exponential backoff (3 attempts for transient failures)
    2. Circuit breaker for graceful degradation (opens after 3 failures)
    
    After circuit breaker opens, returns empty DataFrame for 60s.
    
    Args:
        sample_size: If provided, randomly sample this many rows (for faster development)
    
    Returns:
        DataFrame with columns: userId, movieId, rating, timestamp
    
    Raises:
        FileNotFoundError: If ratings file is missing (should be caught by integrity check)
        CircuitOpenError: If circuit is open (handled by fallback)
    """
    # Delegate to retry-protected inner function
    return _load_ratings_with_retry(sample_size)


def _movies_fallback() -> pd.DataFrame:
    """
    Fallback function when movies cannot be loaded.
    Returns empty DataFrame with correct schema.
    """
    logger.warning("Using fallback empty movies DataFrame")
    return pd.DataFrame({
        'movieId': pd.Series(dtype='int32'),
        'title': pd.Series(dtype='str'),
        'genres': pd.Series(dtype='str'),
        'backdrop_path': pd.Series(dtype='object'),
        'poster_path': pd.Series(dtype='object'),
        'genres_list': pd.Series(dtype='object')
    })


@retry(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    backoff=BackoffStrategy.EXPONENTIAL,
    exceptions={IOError, OSError, pd.errors.ParserError},
    jitter=True
)
def _load_movies_with_retry() -> pd.DataFrame:
    """
    Internal function for loading movies with retry on transient failures.
    
    Retries up to 3 times with exponential backoff for I/O issues.
    """
    # Try Parquet versions first, then CSV
    # Priority: TMDB Parquet > Original Parquet > TMDB CSV > Original CSV
    paths_to_try = [
        DATA_DIR / "movies_with_TMDB_image_links.parquet",
        DATA_DIR / "movies.parquet",
        DATA_DIR / "movies_with_TMDB_image_links.csv",
        DATA_DIR / "movies.csv"
    ]
    
    movies_path = None
    use_parquet = False
    
    for path in paths_to_try:
        if path.exists():
            movies_path = path
            use_parquet = path.suffix == '.parquet'
            break
    
    if movies_path is None:
        raise FileNotFoundError("No movies file found in data directory")
    
    logger.info(f"Loading movies from {movies_path}...")
    
    if use_parquet:
        df = pd.read_parquet(movies_path)
    else:
        df = pd.read_csv(movies_path, dtype=MOVIES_DTYPES)
    
    # Ensure movieId is int32
    if df['movieId'].dtype.name != 'int32':
        df['movieId'] = df['movieId'].astype('int32')
    
    # Ensure image columns exist (for backward compatibility)
    if 'backdrop_path' not in df.columns:
        df['backdrop_path'] = None
    if 'poster_path' not in df.columns:
        df['poster_path'] = None
    
    # Parse genres (pipe-separated) into list
    df['genres_list'] = df['genres'].str.split('|')
    
    logger.info(f"  [OK] Loaded {len(df):,} movies")
    return df


@circuit_breaker(
    name="load_movies",
    failure_threshold=3,
    timeout=60.0,
    fallback=_movies_fallback,
    excluded_exceptions={FileNotFoundError}
)
def load_movies() -> pd.DataFrame:
    """
    Load the movies dataset with TMDB image links from Parquet (preferred) or CSV.
    
    Protected by:
    1. Retry with exponential backoff (3 attempts for transient failures)
    2. Circuit breaker for graceful degradation (opens after 3 failures)
    
    Returns:
        DataFrame with columns: movieId, title, genres, backdrop_path, poster_path
    
    Raises:
        FileNotFoundError: If movies file is missing
        CircuitOpenError: If circuit is open (handled by fallback)
    """
    # Delegate to retry-protected inner function
    return _load_movies_with_retry()


def load_links() -> pd.DataFrame:
    """
    Load the links dataset (IMDb and TMDb IDs) from Parquet (preferred) or CSV.
    
    Returns:
        DataFrame with columns: movieId, imdbId, tmdbId
    """
    parquet_path = DATA_DIR / "links.parquet"
    csv_path = DATA_DIR / "links.csv"
    
    use_parquet = parquet_path.exists()
    file_path = parquet_path if use_parquet else csv_path
    
    logger.info(f"Loading links from {file_path}...")
    
    if use_parquet:
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path, dtype=LINKS_DTYPES)
    
    logger.info(f"  [OK] Loaded {len(df):,} links")
    return df


def load_tags() -> pd.DataFrame:
    """
    Load the tags dataset (user-generated tags) from Parquet (preferred) or CSV.
    
    Returns:
        DataFrame with columns: userId, movieId, tag, timestamp
    """
    parquet_path = DATA_DIR / "tags.parquet"
    csv_path = DATA_DIR / "tags.csv"
    
    use_parquet = parquet_path.exists()
    file_path = parquet_path if use_parquet else csv_path
    
    logger.info(f"Loading tags from {file_path}...")
    
    if use_parquet:
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path, dtype=TAGS_DTYPES)
    
    logger.info(f"  [OK] Loaded {len(df):,} tags")
    return df


def preprocess_data(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess and clean the data.
    
    Args:
        ratings_df: Raw ratings DataFrame
        movies_df: Raw movies DataFrame
    
    Returns:
        Tuple of (cleaned_ratings, cleaned_movies)
    """
    logger.info("Preprocessing data...")
    
    # Remove duplicates
    initial_ratings = len(ratings_df)
    ratings_df = ratings_df.drop_duplicates(subset=['userId', 'movieId'])
    removed_ratings = initial_ratings - len(ratings_df)
    if removed_ratings > 0:
        logger.info(f"  • Removed {removed_ratings:,} duplicate ratings")
    
    # Remove movies with no ratings
    rated_movies = set(ratings_df['movieId'].unique())
    initial_movies = len(movies_df)
    movies_df = movies_df[movies_df['movieId'].isin(rated_movies)]
    removed_movies = initial_movies - len(movies_df)
    if removed_movies > 0:
        logger.info(f"  • Removed {removed_movies:,} movies with no ratings")
    
    # Handle missing genres
    movies_df['genres'] = movies_df['genres'].fillna('(no genres listed)')
    movies_df['genres_list'] = movies_df['genres'].str.split('|')
    
    logger.info("  ✓ Preprocessing complete")
    logger.info(f"    - {len(ratings_df):,} ratings")
    logger.info(f"    - {len(movies_df):,} movies")
    logger.info(f"    - {ratings_df['userId'].nunique():,} users")
    
    return ratings_df, movies_df


def create_user_genre_matrix(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a user-genre preference matrix for taste profiling.
    
    Args:
        ratings_df: Ratings DataFrame
        movies_df: Movies DataFrame
    
    Returns:
        DataFrame with userId as index and genres as columns
    """
    logger.info("Creating user-genre preference matrix...")
    
    # Merge ratings with movie genres
    merged = ratings_df.merge(movies_df[['movieId', 'genres_list']], on='movieId', how='left')
    
    # Explode genres (one row per rating-genre pair)
    merged = merged.explode('genres_list')
    
    # Calculate average rating per user per genre
    user_genre_ratings = merged.groupby(['userId', 'genres_list'])['rating'].agg(['mean', 'count']).reset_index()
    
    # Pivot to create user-genre matrix
    user_genre_matrix = user_genre_ratings.pivot(
        index='userId',
        columns='genres_list',
        values='mean'
    ).fillna(0)
    
    logger.info(f"  ✓ Created matrix: {user_genre_matrix.shape[0]:,} users × {user_genre_matrix.shape[1]} genres")
    
    return user_genre_matrix


def save_processed_data(user_genre_matrix: pd.DataFrame):
    """
    Save preprocessed data to disk for faster loading.
    
    Args:
        user_genre_matrix: User-genre preference matrix to save
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    output_path = PROCESSED_DIR / "user_genre_matrix.pkl"
    user_genre_matrix.to_pickle(output_path)
    
    logger.info(f"✓ Saved processed data to {output_path}")


def load_processed_data() -> Optional[pd.DataFrame]:
    """
    Load preprocessed data if available.
    
    Returns:
        User-genre matrix if exists, None otherwise
    """
    matrix_path = PROCESSED_DIR / "user_genre_matrix.pkl"
    
    if matrix_path.exists():
        logger.info(f"Loading cached user-genre matrix from {matrix_path}...")
        return pd.read_pickle(matrix_path)
    
    return None


if __name__ == "__main__":
    """
    Test the data integrity check.
    """
    # Configure logging for CLI execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    logger.info("CineMatch V2.1.6 - Data Integrity Test\n")
    logger.info("=" * 70)
    
    success, missing, error = check_data_integrity()
    
    if success:
        logger.info("\n✅ SUCCESS: All required dataset files found!")
        logger.info("\nAttempting to load data...")
        
        try:
            ratings = load_ratings(sample_size=100000)  # Sample for testing
            movies = load_movies()
            logger.info("\n✓ Data loading test successful!")
            
        except Exception as e:
            logger.error(f"\n❌ Error loading data: {e}")
            sys.exit(1)
    else:
        logger.error(error)
        sys.exit(1)
