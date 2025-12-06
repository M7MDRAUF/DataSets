"""
CineMatch V2.1.6 - Repository Pattern Implementation

Data access layer with clean separation from business logic.
Implements repository pattern for movies, ratings, and users.

Author: CineMatch Development Team
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Iterator, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from contextlib import contextmanager
import threading


logger = logging.getLogger(__name__)


# =============================================================================
# Data Transfer Objects (DTOs)
# =============================================================================

@dataclass
class MovieDTO:
    """Movie data transfer object"""
    movie_id: int
    title: str
    genres: List[str]
    year: Optional[int] = None
    imdb_id: Optional[str] = None
    tmdb_id: Optional[int] = None
    poster_url: Optional[str] = None
    overview: Optional[str] = None
    avg_rating: Optional[float] = None
    rating_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'movieId': self.movie_id,
            'title': self.title,
            'genres': '|'.join(self.genres),
            'year': self.year,
            'imdbId': self.imdb_id,
            'tmdbId': self.tmdb_id,
            'posterUrl': self.poster_url,
            'overview': self.overview,
            'avgRating': self.avg_rating,
            'ratingCount': self.rating_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MovieDTO':
        genres = data.get('genres', '')
        if isinstance(genres, str):
            genres = genres.split('|') if genres else []
        return cls(
            movie_id=data.get('movieId', data.get('movie_id')),
            title=data.get('title', ''),
            genres=genres,
            year=data.get('year'),
            imdb_id=data.get('imdbId', data.get('imdb_id')),
            tmdb_id=data.get('tmdbId', data.get('tmdb_id')),
            poster_url=data.get('posterUrl', data.get('poster_url')),
            overview=data.get('overview'),
            avg_rating=data.get('avgRating', data.get('avg_rating')),
            rating_count=data.get('ratingCount', data.get('rating_count'))
        )


@dataclass
class RatingDTO:
    """Rating data transfer object"""
    user_id: int
    movie_id: int
    rating: float
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'userId': self.user_id,
            'movieId': self.movie_id,
            'rating': self.rating,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RatingDTO':
        timestamp = data.get('timestamp')
        if isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        return cls(
            user_id=data.get('userId', data.get('user_id')),
            movie_id=data.get('movieId', data.get('movie_id')),
            rating=data.get('rating'),
            timestamp=timestamp
        )


@dataclass
class UserDTO:
    """User data transfer object"""
    user_id: int
    rating_count: int = 0
    avg_rating: float = 0.0
    favorite_genres: List[str] = field(default_factory=list)
    first_rating_date: Optional[datetime] = None
    last_rating_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'userId': self.user_id,
            'ratingCount': self.rating_count,
            'avgRating': self.avg_rating,
            'favoriteGenres': self.favorite_genres,
            'firstRatingDate': self.first_rating_date.isoformat() if self.first_rating_date else None,
            'lastRatingDate': self.last_rating_date.isoformat() if self.last_rating_date else None
        }


@dataclass
class RecommendationDTO:
    """Recommendation data transfer object"""
    movie: MovieDTO
    predicted_rating: float
    confidence: float = 0.0
    explanation: Optional[str] = None
    algorithm: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'movie': self.movie.to_dict(),
            'predictedRating': self.predicted_rating,
            'confidence': self.confidence,
            'explanation': self.explanation,
            'algorithm': self.algorithm
        }


# =============================================================================
# Repository Interfaces
# =============================================================================

class IMovieRepository(ABC):
    """Abstract interface for movie data access"""
    
    @abstractmethod
    def get_by_id(self, movie_id: int) -> Optional[MovieDTO]:
        """Get movie by ID"""
        pass
    
    @abstractmethod
    def get_by_ids(self, movie_ids: List[int]) -> List[MovieDTO]:
        """Get multiple movies by IDs"""
        pass
    
    @abstractmethod
    def search(
        self,
        query: str,
        genres: Optional[List[str]] = None,
        year_range: Optional[Tuple[int, int]] = None,
        min_rating: Optional[float] = None,
        limit: int = 10
    ) -> List[MovieDTO]:
        """Search movies with filters"""
        pass
    
    @abstractmethod
    def get_all_genres(self) -> List[str]:
        """Get all unique genres"""
        pass
    
    @abstractmethod
    def get_popular(self, limit: int = 10) -> List[MovieDTO]:
        """Get popular movies"""
        pass
    
    @abstractmethod
    def get_count(self) -> int:
        """Get total movie count"""
        pass
    
    @abstractmethod
    def get_by_genre(self, genre: str, limit: int = 10) -> List[MovieDTO]:
        """Get movies by genre"""
        pass


class IRatingRepository(ABC):
    """Abstract interface for rating data access"""
    
    @abstractmethod
    def get_by_user(self, user_id: int) -> List[RatingDTO]:
        """Get all ratings by user"""
        pass
    
    @abstractmethod
    def get_by_movie(self, movie_id: int) -> List[RatingDTO]:
        """Get all ratings for movie"""
        pass
    
    @abstractmethod
    def get_rating(self, user_id: int, movie_id: int) -> Optional[RatingDTO]:
        """Get specific rating"""
        pass
    
    @abstractmethod
    def add_rating(self, rating: RatingDTO) -> bool:
        """Add new rating"""
        pass
    
    @abstractmethod
    def update_rating(self, rating: RatingDTO) -> bool:
        """Update existing rating"""
        pass
    
    @abstractmethod
    def delete_rating(self, user_id: int, movie_id: int) -> bool:
        """Delete rating"""
        pass
    
    @abstractmethod
    def get_count(self) -> int:
        """Get total rating count"""
        pass
    
    @abstractmethod
    def get_average_by_movie(self, movie_id: int) -> Optional[float]:
        """Get average rating for movie"""
        pass
    
    @abstractmethod
    def get_all_dataframe(self) -> pd.DataFrame:
        """Get all ratings as DataFrame (for ML training)"""
        pass


class IUserRepository(ABC):
    """Abstract interface for user data access"""
    
    @abstractmethod
    def get_by_id(self, user_id: int) -> Optional[UserDTO]:
        """Get user by ID"""
        pass
    
    @abstractmethod
    def exists(self, user_id: int) -> bool:
        """Check if user exists"""
        pass
    
    @abstractmethod
    def get_all_ids(self) -> List[int]:
        """Get all user IDs"""
        pass
    
    @abstractmethod
    def get_count(self) -> int:
        """Get total user count"""
        pass
    
    @abstractmethod
    def get_active_users(self, min_ratings: int = 10) -> List[int]:
        """Get users with minimum ratings"""
        pass


# =============================================================================
# Concrete Implementations (Pandas-based)
# =============================================================================

class PandasMovieRepository(IMovieRepository):
    """Pandas DataFrame-based movie repository"""
    
    def __init__(self, movies_df: pd.DataFrame, ratings_df: Optional[pd.DataFrame] = None):
        self._movies = movies_df.copy()
        self._ratings = ratings_df
        self._lock = threading.RLock()
        
        # Pre-compute movie statistics
        self._compute_statistics()
    
    def _compute_statistics(self) -> None:
        """Compute movie statistics from ratings"""
        if self._ratings is not None and not self._ratings.empty:
            stats = self._ratings.groupby('movieId').agg({
                'rating': ['mean', 'count']
            }).reset_index()
            stats.columns = ['movieId', 'avgRating', 'ratingCount']
            self._movies = self._movies.merge(stats, on='movieId', how='left')
            self._movies['avgRating'] = self._movies['avgRating'].fillna(0)
            self._movies['ratingCount'] = self._movies['ratingCount'].fillna(0).astype(int)
        else:
            self._movies['avgRating'] = 0.0
            self._movies['ratingCount'] = 0
    
    def _row_to_dto(self, row: pd.Series) -> MovieDTO:
        """Convert DataFrame row to DTO"""
        genres = row.get('genres', '')
        if isinstance(genres, str):
            genres = genres.split('|') if genres else []
        
        # Extract year from title if available
        title = row.get('title', '')
        year = None
        if '(' in title and ')' in title:
            try:
                year_str = title[title.rfind('(')+1:title.rfind(')')]
                year = int(year_str)
            except (ValueError, IndexError):
                pass
        
        return MovieDTO(
            movie_id=int(row.get('movieId', 0)),
            title=title,
            genres=genres,
            year=year,
            imdb_id=row.get('imdbId'),
            tmdb_id=row.get('tmdbId'),
            avg_rating=row.get('avgRating', 0.0),
            rating_count=int(row.get('ratingCount', 0))
        )
    
    def get_by_id(self, movie_id: int) -> Optional[MovieDTO]:
        with self._lock:
            movie = self._movies[self._movies['movieId'] == movie_id]
            if movie.empty:
                return None
            return self._row_to_dto(movie.iloc[0])
    
    def get_by_ids(self, movie_ids: List[int]) -> List[MovieDTO]:
        with self._lock:
            movies = self._movies[self._movies['movieId'].isin(movie_ids)]
            return [self._row_to_dto(row) for _, row in movies.iterrows()]
    
    def search(
        self,
        query: str,
        genres: Optional[List[str]] = None,
        year_range: Optional[Tuple[int, int]] = None,
        min_rating: Optional[float] = None,
        limit: int = 10
    ) -> List[MovieDTO]:
        with self._lock:
            mask = pd.Series([True] * len(self._movies))
            
            # Text search
            if query:
                mask &= self._movies['title'].str.contains(query, case=False, na=False)
            
            # Genre filter
            if genres:
                genre_mask = pd.Series([False] * len(self._movies))
                for genre in genres:
                    genre_mask |= self._movies['genres'].str.contains(genre, case=False, na=False)
                mask &= genre_mask
            
            # Rating filter
            if min_rating is not None:
                mask &= self._movies['avgRating'] >= min_rating
            
            filtered = self._movies[mask].head(limit)
            return [self._row_to_dto(row) for _, row in filtered.iterrows()]
    
    def get_all_genres(self) -> List[str]:
        with self._lock:
            all_genres = set()
            for genres in self._movies['genres'].dropna():
                all_genres.update(genres.split('|'))
            return sorted(list(all_genres - {'(no genres listed)', ''}))
    
    def get_popular(self, limit: int = 10) -> List[MovieDTO]:
        with self._lock:
            popular = self._movies.nlargest(limit, 'ratingCount')
            return [self._row_to_dto(row) for _, row in popular.iterrows()]
    
    def get_count(self) -> int:
        with self._lock:
            return len(self._movies)
    
    def get_by_genre(self, genre: str, limit: int = 10) -> List[MovieDTO]:
        with self._lock:
            filtered = self._movies[
                self._movies['genres'].str.contains(genre, case=False, na=False)
            ].head(limit)
            return [self._row_to_dto(row) for _, row in filtered.iterrows()]


class PandasRatingRepository(IRatingRepository):
    """Pandas DataFrame-based rating repository"""
    
    def __init__(self, ratings_df: pd.DataFrame):
        self._ratings = ratings_df.copy()
        self._lock = threading.RLock()
    
    def _row_to_dto(self, row: pd.Series) -> RatingDTO:
        """Convert DataFrame row to DTO"""
        timestamp = row.get('timestamp')
        if pd.notna(timestamp):
            if isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp)
        else:
            timestamp = None
        
        return RatingDTO(
            user_id=int(row.get('userId', 0)),
            movie_id=int(row.get('movieId', 0)),
            rating=float(row.get('rating', 0)),
            timestamp=timestamp
        )
    
    def get_by_user(self, user_id: int) -> List[RatingDTO]:
        with self._lock:
            user_ratings = self._ratings[self._ratings['userId'] == user_id]
            return [self._row_to_dto(row) for _, row in user_ratings.iterrows()]
    
    def get_by_movie(self, movie_id: int) -> List[RatingDTO]:
        with self._lock:
            movie_ratings = self._ratings[self._ratings['movieId'] == movie_id]
            return [self._row_to_dto(row) for _, row in movie_ratings.iterrows()]
    
    def get_rating(self, user_id: int, movie_id: int) -> Optional[RatingDTO]:
        with self._lock:
            rating = self._ratings[
                (self._ratings['userId'] == user_id) & 
                (self._ratings['movieId'] == movie_id)
            ]
            if rating.empty:
                return None
            return self._row_to_dto(rating.iloc[0])
    
    def add_rating(self, rating: RatingDTO) -> bool:
        with self._lock:
            # Check if already exists
            existing = self.get_rating(rating.user_id, rating.movie_id)
            if existing:
                return self.update_rating(rating)
            
            new_row = {
                'userId': rating.user_id,
                'movieId': rating.movie_id,
                'rating': rating.rating,
                'timestamp': rating.timestamp.timestamp() if rating.timestamp else None
            }
            self._ratings = pd.concat([
                self._ratings, 
                pd.DataFrame([new_row])
            ], ignore_index=True)
            return True
    
    def update_rating(self, rating: RatingDTO) -> bool:
        with self._lock:
            mask = (
                (self._ratings['userId'] == rating.user_id) & 
                (self._ratings['movieId'] == rating.movie_id)
            )
            if not mask.any():
                return False
            
            self._ratings.loc[mask, 'rating'] = rating.rating
            if rating.timestamp:
                self._ratings.loc[mask, 'timestamp'] = rating.timestamp.timestamp()
            return True
    
    def delete_rating(self, user_id: int, movie_id: int) -> bool:
        with self._lock:
            mask = (
                (self._ratings['userId'] == user_id) & 
                (self._ratings['movieId'] == movie_id)
            )
            if not mask.any():
                return False
            
            self._ratings = self._ratings[~mask]
            return True
    
    def get_count(self) -> int:
        with self._lock:
            return len(self._ratings)
    
    def get_average_by_movie(self, movie_id: int) -> Optional[float]:
        with self._lock:
            movie_ratings = self._ratings[self._ratings['movieId'] == movie_id]
            if movie_ratings.empty:
                return None
            return movie_ratings['rating'].mean()
    
    def get_all_dataframe(self) -> pd.DataFrame:
        with self._lock:
            return self._ratings.copy()


class PandasUserRepository(IUserRepository):
    """Pandas DataFrame-based user repository"""
    
    def __init__(self, ratings_df: pd.DataFrame, movies_df: Optional[pd.DataFrame] = None):
        self._ratings = ratings_df
        self._movies = movies_df
        self._user_cache: Dict[int, UserDTO] = {}
        self._lock = threading.RLock()
    
    def _compute_user_stats(self, user_id: int) -> Optional[UserDTO]:
        """Compute user statistics from ratings"""
        user_ratings = self._ratings[self._ratings['userId'] == user_id]
        
        if user_ratings.empty:
            return None
        
        # Basic stats
        rating_count = len(user_ratings)
        avg_rating = user_ratings['rating'].mean()
        
        # Timestamps
        first_rating = None
        last_rating = None
        if 'timestamp' in user_ratings.columns:
            timestamps = user_ratings['timestamp'].dropna()
            if not timestamps.empty:
                first_rating = datetime.fromtimestamp(timestamps.min())
                last_rating = datetime.fromtimestamp(timestamps.max())
        
        # Favorite genres
        favorite_genres = []
        if self._movies is not None:
            merged = user_ratings.merge(self._movies[['movieId', 'genres']], on='movieId')
            genre_counts: Dict[str, int] = {}
            for genres in merged['genres'].dropna():
                for genre in genres.split('|'):
                    if genre and genre != '(no genres listed)':
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
            favorite_genres = sorted(genre_counts, key=genre_counts.get, reverse=True)[:5]
        
        return UserDTO(
            user_id=user_id,
            rating_count=rating_count,
            avg_rating=avg_rating,
            favorite_genres=favorite_genres,
            first_rating_date=first_rating,
            last_rating_date=last_rating
        )
    
    def get_by_id(self, user_id: int) -> Optional[UserDTO]:
        with self._lock:
            if user_id in self._user_cache:
                return self._user_cache[user_id]
            
            user = self._compute_user_stats(user_id)
            if user:
                self._user_cache[user_id] = user
            return user
    
    def exists(self, user_id: int) -> bool:
        with self._lock:
            return user_id in self._ratings['userId'].values
    
    def get_all_ids(self) -> List[int]:
        with self._lock:
            return self._ratings['userId'].unique().tolist()
    
    def get_count(self) -> int:
        with self._lock:
            return self._ratings['userId'].nunique()
    
    def get_active_users(self, min_ratings: int = 10) -> List[int]:
        with self._lock:
            user_counts = self._ratings.groupby('userId').size()
            return user_counts[user_counts >= min_ratings].index.tolist()


# =============================================================================
# Unit of Work Pattern
# =============================================================================

class IUnitOfWork(ABC):
    """Abstract unit of work interface"""
    
    movies: IMovieRepository
    ratings: IRatingRepository
    users: IUserRepository
    
    @abstractmethod
    def begin(self) -> None:
        """Begin transaction"""
        pass
    
    @abstractmethod
    def commit(self) -> None:
        """Commit transaction"""
        pass
    
    @abstractmethod
    def rollback(self) -> None:
        """Rollback transaction"""
        pass
    
    @abstractmethod
    @contextmanager
    def transaction(self):
        """Context manager for transactions"""
        pass


class PandasUnitOfWork(IUnitOfWork):
    """Pandas-based unit of work implementation"""
    
    def __init__(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame
    ):
        self._movies_df = movies_df
        self._ratings_df = ratings_df
        self._snapshot_movies: Optional[pd.DataFrame] = None
        self._snapshot_ratings: Optional[pd.DataFrame] = None
        
        # Initialize repositories
        self.movies = PandasMovieRepository(movies_df, ratings_df)
        self.ratings = PandasRatingRepository(ratings_df)
        self.users = PandasUserRepository(ratings_df, movies_df)
    
    def begin(self) -> None:
        """Create snapshot for potential rollback"""
        self._snapshot_movies = self._movies_df.copy()
        self._snapshot_ratings = self._ratings_df.copy()
        logger.debug("Transaction started")
    
    def commit(self) -> None:
        """Clear snapshot (commit changes)"""
        self._snapshot_movies = None
        self._snapshot_ratings = None
        logger.debug("Transaction committed")
    
    def rollback(self) -> None:
        """Restore from snapshot"""
        if self._snapshot_movies is not None:
            self._movies_df = self._snapshot_movies
            self.movies = PandasMovieRepository(self._movies_df, self._ratings_df)
        if self._snapshot_ratings is not None:
            self._ratings_df = self._snapshot_ratings
            self.ratings = PandasRatingRepository(self._ratings_df)
            self.users = PandasUserRepository(self._ratings_df, self._movies_df)
        logger.debug("Transaction rolled back")
    
    @contextmanager
    def transaction(self):
        """Context manager for atomic operations"""
        self.begin()
        try:
            yield self
            self.commit()
        except Exception as e:
            self.rollback()
            logger.error(f"Transaction failed: {e}")
            raise


# =============================================================================
# Repository Factory
# =============================================================================

class RepositoryFactory:
    """Factory for creating repository instances"""
    
    @staticmethod
    def create_from_csv(
        movies_path: Path,
        ratings_path: Path
    ) -> PandasUnitOfWork:
        """Create repositories from CSV files"""
        movies_df = pd.read_csv(movies_path)
        ratings_df = pd.read_csv(ratings_path)
        return PandasUnitOfWork(movies_df, ratings_df)
    
    @staticmethod
    def create_from_dataframes(
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame
    ) -> PandasUnitOfWork:
        """Create repositories from DataFrames"""
        return PandasUnitOfWork(movies_df, ratings_df)
