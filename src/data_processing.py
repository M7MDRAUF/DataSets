"""
CineMatch V1.0.0 - Data Processing Module

This module handles data loading, preprocessing, and integrity checks.
Implements NF-01: Data Integrity & Developer Alerting requirements.

Author: CineMatch Team
Date: October 24, 2025
"""

import os
import sys
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path


# Data paths
DATA_DIR = Path("data/ml-32m")
PROCESSED_DIR = Path("data/processed")
REQUIRED_FILES = [
    "ratings.csv",
    "movies.csv",
    "links.csv",
    "tags.csv"
]


def check_data_integrity() -> Tuple[bool, List[str], Optional[str]]:
    """
    Validates the presence of all required dataset files.
    
    Implements NF-01: Data Integrity & Developer Alerting
    
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
    
    # Check if data directory exists
    if not DATA_DIR.exists():
        missing_files = REQUIRED_FILES
    else:
        # Check each required file
        for filename in REQUIRED_FILES:
            file_path = DATA_DIR / filename
            if not file_path.exists():
                missing_files.append(filename)
    
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


def load_ratings(sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load the ratings dataset.
    
    Args:
        sample_size: If provided, randomly sample this many rows (for faster development)
    
    Returns:
        DataFrame with columns: userId, movieId, rating, timestamp
    
    Raises:
        FileNotFoundError: If ratings.csv is missing (should be caught by integrity check)
    """
    ratings_path = DATA_DIR / "ratings.csv"
    
    print(f"Loading ratings from {ratings_path}...")
    
    # Load with dtype optimization for memory efficiency
    dtypes = {
        'userId': 'int32',
        'movieId': 'int32',
        'rating': 'float32',
        'timestamp': 'int64'
    }
    
    if sample_size:
        # For development: load a sample
        print(f"  → Sampling {sample_size:,} ratings for faster processing")
        # Read in chunks and sample
        df = pd.read_csv(ratings_path, dtype=dtypes)
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    else:
        # Production: load full dataset
        df = pd.read_csv(ratings_path, dtype=dtypes)
    
    print(f"  [OK] Loaded {len(df):,} ratings")
    return df


def load_movies() -> pd.DataFrame:
    """
    Load the movies dataset with TMDB image links.
    
    Returns:
        DataFrame with columns: movieId, title, genres, backdrop_path, poster_path
    
    Raises:
        FileNotFoundError: If movies CSV is missing
    """
    # Try TMDB version first, fall back to original
    tmdb_path = DATA_DIR / "movies_with_TMDB_image_links.csv"
    original_path = DATA_DIR / "movies.csv"
    
    if tmdb_path.exists():
        movies_path = tmdb_path
        print(f"Loading movies with TMDB images from {movies_path}...")
    else:
        movies_path = original_path
        print(f"Loading movies from {movies_path}...")
    
    df = pd.read_csv(movies_path, dtype={'movieId': 'int32'})
    
    # Ensure image columns exist (for backward compatibility)
    if 'backdrop_path' not in df.columns:
        df['backdrop_path'] = None
    if 'poster_path' not in df.columns:
        df['poster_path'] = None
    
    # Parse genres (pipe-separated) into list
    df['genres_list'] = df['genres'].str.split('|')
    
    print(f"  [OK] Loaded {len(df):,} movies")
    return df


def load_links() -> pd.DataFrame:
    """
    Load the links dataset (IMDb and TMDb IDs).
    
    Returns:
        DataFrame with columns: movieId, imdbId, tmdbId
    """
    links_path = DATA_DIR / "links.csv"
    
    print(f"Loading links from {links_path}...")
    
    dtypes = {
        'movieId': 'int32',
        'imdbId': 'str',
        'tmdbId': 'float32'  # Float because of NaN values
    }
    
    df = pd.read_csv(links_path, dtype=dtypes)
    
    print(f"  [OK] Loaded {len(df):,} links")
    return df


def load_tags() -> pd.DataFrame:
    """
    Load the tags dataset (user-generated tags).
    
    Returns:
        DataFrame with columns: userId, movieId, tag, timestamp
    """
    tags_path = DATA_DIR / "tags.csv"
    
    print(f"Loading tags from {tags_path}...")
    
    dtypes = {
        'userId': 'int32',
        'movieId': 'int32',
        'tag': 'str',
        'timestamp': 'int64'
    }
    
    df = pd.read_csv(tags_path, dtype=dtypes)
    
    print(f"  [OK] Loaded {len(df):,} tags")
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
    print("\nPreprocessing data...")
    
    # Remove duplicates
    initial_ratings = len(ratings_df)
    ratings_df = ratings_df.drop_duplicates(subset=['userId', 'movieId'])
    removed_ratings = initial_ratings - len(ratings_df)
    if removed_ratings > 0:
        print(f"  • Removed {removed_ratings:,} duplicate ratings")
    
    # Remove movies with no ratings
    rated_movies = set(ratings_df['movieId'].unique())
    initial_movies = len(movies_df)
    movies_df = movies_df[movies_df['movieId'].isin(rated_movies)]
    removed_movies = initial_movies - len(movies_df)
    if removed_movies > 0:
        print(f"  • Removed {removed_movies:,} movies with no ratings")
    
    # Handle missing genres
    movies_df['genres'] = movies_df['genres'].fillna('(no genres listed)')
    movies_df['genres_list'] = movies_df['genres'].str.split('|')
    
    print(f"  ✓ Preprocessing complete")
    print(f"    - {len(ratings_df):,} ratings")
    print(f"    - {len(movies_df):,} movies")
    print(f"    - {ratings_df['userId'].nunique():,} users")
    
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
    print("\nCreating user-genre preference matrix...")
    
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
    
    print(f"  ✓ Created matrix: {user_genre_matrix.shape[0]:,} users × {user_genre_matrix.shape[1]} genres")
    
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
    
    print(f"\n✓ Saved processed data to {output_path}")


def load_processed_data() -> Optional[pd.DataFrame]:
    """
    Load preprocessed data if available.
    
    Returns:
        User-genre matrix if exists, None otherwise
    """
    matrix_path = PROCESSED_DIR / "user_genre_matrix.pkl"
    
    if matrix_path.exists():
        print(f"Loading cached user-genre matrix from {matrix_path}...")
        return pd.read_pickle(matrix_path)
    
    return None


if __name__ == "__main__":
    """
    Test the data integrity check.
    """
    print("CineMatch V1.0.0 - Data Integrity Test\n")
    print("=" * 70)
    
    success, missing, error = check_data_integrity()
    
    if success:
        print("\n✅ SUCCESS: All required dataset files found!")
        print("\nAttempting to load data...")
        
        try:
            ratings = load_ratings(sample_size=100000)  # Sample for testing
            movies = load_movies()
            print("\n✓ Data loading test successful!")
            
        except Exception as e:
            print(f"\n❌ Error loading data: {e}")
            sys.exit(1)
    else:
        print(error)
        sys.exit(1)
