"""
CineMatch V2.1.2 - Search Engine Module

Backend logic for user rating history lookup and advanced search functionality.
Implements professor's request to view all movies rated by a specific user.

Author: CineMatch Development Team
Date: November 20, 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_user_existence(user_id: int, ratings_df: pd.DataFrame) -> bool:
    """
    Check if a user ID exists in the ratings dataset.
    
    Args:
        user_id: User identifier to validate
        ratings_df: Ratings DataFrame with 'userId' column
        
    Returns:
        bool: True if user exists, False otherwise
    """
    if ratings_df is None or ratings_df.empty:
        return False
    
    return int(user_id) in ratings_df['userId'].values


def get_user_ratings(
    user_id: int, 
    ratings_df: pd.DataFrame, 
    movies_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Retrieve all movies rated by a specific user with metadata.
    
    Args:
        user_id: User identifier
        ratings_df: Full ratings DataFrame (userId, movieId, rating, timestamp)
        movies_df: Movies metadata (movieId, title, genres)
        
    Returns:
        DataFrame with columns: movieId, title, genres, rating, timestamp
        Sorted by timestamp (most recent first)
    """
    try:
        # Filter ratings for this user
        user_ratings = ratings_df[ratings_df['userId'] == user_id].copy()
        
        if user_ratings.empty:
            logger.warning(f"No ratings found for User ID {user_id}")
            return pd.DataFrame()
        
        # Merge with movie metadata (including poster_path for TMDB images)
        user_ratings_with_metadata = user_ratings.merge(
            movies_df[['movieId', 'title', 'genres', 'poster_path']], 
            on='movieId', 
            how='left'
        )
        
        # Sort by timestamp (most recent first)
        user_ratings_with_metadata = user_ratings_with_metadata.sort_values(
            'timestamp', 
            ascending=False
        )
        
        # Convert timestamp to readable format
        user_ratings_with_metadata['timestamp'] = pd.to_datetime(
            user_ratings_with_metadata['timestamp'], 
            unit='s'
        )
        
        logger.info(f"Retrieved {len(user_ratings_with_metadata)} ratings for User ID {user_id}")
        return user_ratings_with_metadata
        
    except Exception as e:
        logger.error(f"Error retrieving ratings for User ID {user_id}: {e}")
        return pd.DataFrame()


def get_user_statistics(user_ratings: pd.DataFrame) -> Dict[str, any]:
    """
    Calculate comprehensive statistics for a user's rating history.
    
    Args:
        user_ratings: DataFrame of user's ratings (from get_user_ratings)
        
    Returns:
        Dict with statistics:
        - total_ratings: int
        - avg_rating: float
        - median_rating: float
        - std_rating: float
        - min_rating: float
        - max_rating: float
        - top_genres: List[Tuple[str, int]]
        - rating_distribution: Dict[float, int]
        - first_rating_date: datetime
        - last_rating_date: datetime
    """
    if user_ratings.empty:
        return {
            'total_ratings': 0,
            'avg_rating': 0.0,
            'median_rating': 0.0,
            'std_rating': 0.0,
            'min_rating': 0.0,
            'max_rating': 0.0,
            'top_genres': [],
            'rating_distribution': {},
            'first_rating_date': None,
            'last_rating_date': None
        }
    
    # Basic statistics
    stats = {
        'total_ratings': len(user_ratings),
        'avg_rating': user_ratings['rating'].mean(),
        'median_rating': user_ratings['rating'].median(),
        'std_rating': user_ratings['rating'].std(),
        'min_rating': user_ratings['rating'].min(),
        'max_rating': user_ratings['rating'].max(),
    }
    
    # Rating distribution
    rating_dist = user_ratings['rating'].value_counts().to_dict()
    stats['rating_distribution'] = {float(k): int(v) for k, v in rating_dist.items()}
    
    # Genre analysis
    genre_counts = {}
    for genres_str in user_ratings['genres'].dropna():
        for genre in str(genres_str).split('|'):
            genre = genre.strip()
            if genre and genre != '(no genres listed)':
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    # Sort genres by frequency
    top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    stats['top_genres'] = top_genres
    
    # Temporal statistics
    stats['first_rating_date'] = user_ratings['timestamp'].min()
    stats['last_rating_date'] = user_ratings['timestamp'].max()
    
    return stats


def search_movies_by_criteria(
    movies_df: pd.DataFrame,
    title_query: Optional[str] = None,
    genre_filter: Optional[str] = None,
    limit: int = 100
) -> pd.DataFrame:
    """
    Search movies by title and/or genre.
    
    Args:
        movies_df: Movies metadata DataFrame
        title_query: Partial movie title to search
        genre_filter: Genre to filter by
        limit: Maximum number of results to return
        
    Returns:
        Filtered DataFrame of matching movies
    """
    result = movies_df.copy()
    
    # Filter by title (case-insensitive partial match)
    if title_query:
        result = result[
            result['title'].str.contains(title_query, case=False, na=False)
        ]
    
    # Filter by genre
    if genre_filter:
        result = result[
            result['genres'].str.contains(genre_filter, case=False, na=False)
        ]
    
    return result.head(limit)


def get_user_genre_preferences(user_ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate normalized genre preference scores for a user.
    
    Args:
        user_ratings: DataFrame of user's ratings with genres
        
    Returns:
        DataFrame with columns: genre, count, avg_rating, preference_score
        Sorted by preference_score (descending)
    """
    if user_ratings.empty:
        return pd.DataFrame(columns=['genre', 'count', 'avg_rating', 'preference_score'])
    
    genre_data = []
    
    # Explode genres and calculate statistics
    for _, row in user_ratings.iterrows():
        if pd.notna(row['genres']):
            genres = str(row['genres']).split('|')
            for genre in genres:
                genre = genre.strip()
                if genre and genre != '(no genres listed)':
                    genre_data.append({
                        'genre': genre,
                        'rating': row['rating']
                    })
    
    if not genre_data:
        return pd.DataFrame(columns=['genre', 'count', 'avg_rating', 'preference_score'])
    
    genre_df = pd.DataFrame(genre_data)
    
    # Calculate statistics per genre
    genre_stats = genre_df.groupby('genre').agg({
        'rating': ['count', 'mean']
    }).reset_index()
    
    genre_stats.columns = ['genre', 'count', 'avg_rating']
    
    # Calculate preference score (weighted by frequency and avg rating)
    # Normalize count to 0-1 range
    max_count = genre_stats['count'].max()
    genre_stats['count_norm'] = genre_stats['count'] / max_count
    
    # Preference score: 60% avg_rating + 40% frequency
    genre_stats['preference_score'] = (
        0.6 * (genre_stats['avg_rating'] / 5.0) + 
        0.4 * genre_stats['count_norm']
    )
    
    # Sort by preference score
    genre_stats = genre_stats.sort_values('preference_score', ascending=False)
    
    return genre_stats[['genre', 'count', 'avg_rating', 'preference_score']]


def get_rating_timeline(user_ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Create a timeline of user's rating activity.
    
    Args:
        user_ratings: DataFrame of user's ratings with timestamps
        
    Returns:
        DataFrame with columns: date, rating_count, cumulative_count
    """
    if user_ratings.empty or 'timestamp' not in user_ratings.columns:
        return pd.DataFrame(columns=['date', 'rating_count', 'cumulative_count'])
    
    # Extract date from timestamp
    user_ratings_copy = user_ratings.copy()
    user_ratings_copy['date'] = user_ratings_copy['timestamp'].dt.date
    
    # Count ratings per date
    timeline = user_ratings_copy.groupby('date').size().reset_index(name='rating_count')
    timeline = timeline.sort_values('date')
    
    # Calculate cumulative count
    timeline['cumulative_count'] = timeline['rating_count'].cumsum()
    
    return timeline
