"""
CineMatch V2.1.6 - Precomputed Items

Precompute and cache popular/trending movies at startup.
Reduces recommendation latency for cold start users.

Author: CineMatch Development Team
Date: December 5, 2025

Features:
    - Precomputed popular movies (by rating count)
    - Precomputed highly-rated movies (by average rating)
    - Genre-specific popular movies
    - Time-weighted trending movies
    - Cached at startup for instant access
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from functools import lru_cache
import time

logger = logging.getLogger(__name__)


@dataclass
class PopularMovies:
    """Container for precomputed popular movies."""
    
    # Overall popular movies (by rating count)
    by_rating_count: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Highly rated movies (by average rating, min ratings threshold)
    by_average_rating: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Popular by genre
    by_genre: Dict[str, pd.DataFrame] = field(default_factory=dict)
    
    # Time-weighted trending (recent ratings weighted higher)
    trending: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Metadata
    computed_at: float = field(default_factory=time.time)
    rating_count_threshold: int = 50
    
    @property
    def age_seconds(self) -> float:
        """Time since precomputation."""
        return time.time() - self.computed_at
    
    @property
    def is_stale(self) -> bool:
        """Check if precomputed data is stale (older than 24 hours)."""
        return self.age_seconds > 86400  # 24 hours


class PopularItemsCache:
    """
    Precomputed popular movies cache.
    
    Computes popular items at startup for instant cold-start recommendations.
    
    Usage:
        cache = PopularItemsCache()
        cache.precompute(ratings_df, movies_df)
        
        # Get top 10 popular movies
        popular = cache.get_popular(n=10)
        
        # Get popular in a genre
        action = cache.get_popular_by_genre('Action', n=10)
    """
    
    def __init__(
        self,
        min_ratings: int = 50,
        top_n: int = 100,
        trending_window_days: int = 365
    ):
        """
        Initialize popular items cache.
        
        Args:
            min_ratings: Minimum ratings for a movie to be considered
            top_n: Number of items to precompute per category
            trending_window_days: Window for trending calculation
        """
        self.min_ratings = min_ratings
        self.top_n = top_n
        self.trending_window_days = trending_window_days
        self._cache: Optional[PopularMovies] = None
        self._is_initialized = False
    
    def precompute(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        verbose: bool = True
    ) -> None:
        """
        Precompute popular items from ratings data.
        
        Args:
            ratings_df: Ratings DataFrame with userId, movieId, rating, timestamp
            movies_df: Movies DataFrame with movieId, title, genres
            verbose: Print progress
        """
        start_time = time.time()
        
        if verbose:
            print("🔥 Precomputing popular items...")
        
        # Calculate rating statistics per movie
        movie_stats = ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean', 'std'],
            'timestamp': 'max'
        }).reset_index()
        
        movie_stats.columns = ['movieId', 'rating_count', 'avg_rating', 'std_rating', 'last_rated']
        
        # Filter by minimum ratings
        popular_movies = movie_stats[movie_stats['rating_count'] >= self.min_ratings]
        
        # Merge with movie metadata
        popular_movies = popular_movies.merge(
            movies_df[['movieId', 'title', 'genres']],
            on='movieId',
            how='left'
        )
        
        # 1. Popular by rating count
        by_count = popular_movies.nlargest(self.top_n, 'rating_count').copy()
        by_count['popularity_score'] = by_count['rating_count'] / by_count['rating_count'].max()
        
        # 2. Popular by average rating (with Bayesian adjustment)
        # Bayesian average: (n * avg + m * C) / (n + m)
        # where m = minimum votes, C = overall mean
        overall_mean = popular_movies['avg_rating'].mean()
        m = self.min_ratings
        
        popular_movies['bayesian_rating'] = (
            (popular_movies['rating_count'] * popular_movies['avg_rating'] + m * overall_mean) /
            (popular_movies['rating_count'] + m)
        )
        
        by_rating = popular_movies.nlargest(self.top_n, 'bayesian_rating').copy()
        
        # 3. Trending (recent ratings weighted higher)
        now = ratings_df['timestamp'].max()
        cutoff = now - (self.trending_window_days * 86400)
        
        recent_ratings = ratings_df[ratings_df['timestamp'] >= cutoff]
        recent_stats = recent_ratings.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        recent_stats.columns = ['movieId', 'recent_count', 'recent_avg']
        
        trending = popular_movies.merge(recent_stats, on='movieId', how='left')
        trending['recent_count'] = trending['recent_count'].fillna(0)
        trending['recent_avg'] = trending['recent_avg'].fillna(trending['avg_rating'])
        
        # Trending score: recent activity + quality
        trending['trending_score'] = (
            trending['recent_count'] / trending['recent_count'].max() * 0.7 +
            trending['recent_avg'] / 5.0 * 0.3
        )
        trending = trending.nlargest(self.top_n, 'trending_score')
        
        # 4. Popular by genre
        by_genre = {}
        all_genres = set()
        
        for genres in movies_df['genres'].dropna():
            all_genres.update(genres.split('|'))
        
        for genre in all_genres:
            if genre == '(no genres listed)':
                continue
                
            genre_movies = popular_movies[
                popular_movies['genres'].str.contains(genre, na=False)
            ]
            
            if len(genre_movies) >= 10:
                by_genre[genre] = genre_movies.nlargest(
                    min(self.top_n, len(genre_movies)),
                    'bayesian_rating'
                ).copy()
        
        # Store in cache
        self._cache = PopularMovies(
            by_rating_count=by_count,
            by_average_rating=by_rating,
            by_genre=by_genre,
            trending=trending,
            rating_count_threshold=self.min_ratings
        )
        self._is_initialized = True
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"  ✓ Precomputed in {elapsed:.2f}s")
            print(f"  • Popular by count: {len(by_count)} movies")
            print(f"  • Popular by rating: {len(by_rating)} movies")
            print(f"  • Trending: {len(trending)} movies")
            print(f"  • Genres covered: {len(by_genre)} genres")
    
    def get_popular(
        self,
        n: int = 10,
        method: str = 'count'
    ) -> pd.DataFrame:
        """
        Get top N popular movies.
        
        Args:
            n: Number of movies
            method: 'count' (most rated) or 'rating' (highest rated)
            
        Returns:
            DataFrame with popular movies
        """
        if not self._is_initialized or self._cache is None:
            return pd.DataFrame()
        
        if method == 'count':
            return self._cache.by_rating_count.head(n)
        else:
            return self._cache.by_average_rating.head(n)
    
    def get_trending(self, n: int = 10) -> pd.DataFrame:
        """Get top N trending movies."""
        if not self._is_initialized or self._cache is None:
            return pd.DataFrame()
        
        return self._cache.trending.head(n)
    
    def get_popular_by_genre(self, genre: str, n: int = 10) -> pd.DataFrame:
        """Get top N popular movies in a genre."""
        if not self._is_initialized or self._cache is None:
            return pd.DataFrame()
        
        if genre in self._cache.by_genre:
            return self._cache.by_genre[genre].head(n)
        
        return pd.DataFrame()
    
    def get_cold_start_recommendations(
        self,
        n: int = 10,
        diversity: bool = True
    ) -> pd.DataFrame:
        """
        Get recommendations for cold-start users (no rating history).
        
        Args:
            n: Number of recommendations
            diversity: Include genre diversity
            
        Returns:
            DataFrame with recommendations
        """
        if not self._is_initialized or self._cache is None:
            return pd.DataFrame()
        
        if not diversity:
            return self.get_popular(n, method='rating')
        
        # Mix popular from different genres for diversity
        recommendations = []
        seen_ids = set()
        
        # Start with top trending
        for _, row in self._cache.trending.head(n // 3).iterrows():
            if row['movieId'] not in seen_ids:
                recommendations.append(row)
                seen_ids.add(row['movieId'])
        
        # Add top rated
        for _, row in self._cache.by_average_rating.head(n // 3).iterrows():
            if row['movieId'] not in seen_ids:
                recommendations.append(row)
                seen_ids.add(row['movieId'])
        
        # Add genre variety
        genres = list(self._cache.by_genre.keys())
        np.random.shuffle(genres)
        
        for genre in genres:
            if len(recommendations) >= n:
                break
            
            genre_df = self._cache.by_genre[genre]
            for _, row in genre_df.head(2).iterrows():
                if row['movieId'] not in seen_ids:
                    recommendations.append(row)
                    seen_ids.add(row['movieId'])
                    break
        
        # Fill remaining with popular
        for _, row in self._cache.by_rating_count.iterrows():
            if len(recommendations) >= n:
                break
            if row['movieId'] not in seen_ids:
                recommendations.append(row)
                seen_ids.add(row['movieId'])
        
        return pd.DataFrame(recommendations[:n])
    
    @property
    def is_initialized(self) -> bool:
        """Check if cache is initialized."""
        return self._is_initialized
    
    @property
    def stats(self) -> Dict:
        """Get cache statistics."""
        if not self._is_initialized or self._cache is None:
            return {'initialized': False}
        
        return {
            'initialized': True,
            'age_seconds': self._cache.age_seconds,
            'is_stale': self._cache.is_stale,
            'popular_count': len(self._cache.by_rating_count),
            'top_rated_count': len(self._cache.by_average_rating),
            'trending_count': len(self._cache.trending),
            'genres_count': len(self._cache.by_genre)
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_popular_cache: Optional[PopularItemsCache] = None


def get_popular_items_cache() -> PopularItemsCache:
    """Get or create global popular items cache."""
    global _popular_cache
    
    if _popular_cache is None:
        _popular_cache = PopularItemsCache()
    
    return _popular_cache


def initialize_popular_items(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame
) -> PopularItemsCache:
    """
    Initialize global popular items cache.
    
    Call this at application startup.
    """
    cache = get_popular_items_cache()
    cache.precompute(ratings_df, movies_df)
    return cache


# =============================================================================
# CLI & TESTING
# =============================================================================

if __name__ == '__main__':
    print("Popular Items Cache Demo")
    print("=" * 40)
    
    # Create sample data
    np.random.seed(42)
    n_users = 1000
    n_movies = 500
    n_ratings = 50000
    
    ratings_df = pd.DataFrame({
        'userId': np.random.randint(1, n_users + 1, n_ratings),
        'movieId': np.random.randint(1, n_movies + 1, n_ratings),
        'rating': np.random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], n_ratings),
        'timestamp': np.random.randint(1500000000, 1700000000, n_ratings)
    })
    
    genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
    movies_df = pd.DataFrame({
        'movieId': range(1, n_movies + 1),
        'title': [f'Movie {i}' for i in range(1, n_movies + 1)],
        'genres': [
            '|'.join(np.random.choice(genres, np.random.randint(1, 4), replace=False))
            for _ in range(n_movies)
        ]
    })
    
    # Initialize cache
    cache = PopularItemsCache(min_ratings=20, top_n=50)
    cache.precompute(ratings_df, movies_df)
    
    # Get popular
    print("\nTop 5 Popular (by count):")
    print(cache.get_popular(5, method='count')[['movieId', 'title', 'rating_count', 'avg_rating']])
    
    print("\nTop 5 Popular (by rating):")
    print(cache.get_popular(5, method='rating')[['movieId', 'title', 'rating_count', 'bayesian_rating']])
    
    print("\nTop 5 Trending:")
    print(cache.get_trending(5)[['movieId', 'title', 'trending_score']])
    
    print("\nTop 5 Action movies:")
    print(cache.get_popular_by_genre('Action', 5)[['movieId', 'title', 'avg_rating']])
    
    print("\nCold Start Recommendations (diverse):")
    print(cache.get_cold_start_recommendations(5)[['movieId', 'title', 'genres']])
    
    print(f"\nStats: {cache.stats}")
