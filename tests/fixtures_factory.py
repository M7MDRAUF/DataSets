"""
CineMatch V2.1.6 - Test Fixtures Factory

Reusable test fixtures and data generators for comprehensive testing.
Task 3.5: Create test fixtures factory.

Author: CineMatch Development Team
Date: December 5, 2025
"""

import pytest
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MovieData:
    """Container for movie data"""
    movie_id: int
    title: str
    genres: str
    year: Optional[int] = None


@dataclass
class RatingData:
    """Container for rating data"""
    user_id: int
    movie_id: int
    rating: float
    timestamp: Optional[int] = None


@dataclass
class UserProfile:
    """Container for user profile data"""
    user_id: int
    preferred_genres: List[str]
    avg_rating: float
    num_ratings: int


@dataclass
class TestDataset:
    """Container for complete test dataset"""
    movies: pd.DataFrame
    ratings: pd.DataFrame
    n_users: int
    n_movies: int
    n_ratings: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# FIXTURE FACTORY CLASS
# =============================================================================

class FixtureFactory:
    """Factory for creating test fixtures and data"""
    
    GENRES = [
        'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
        'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
        'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
        'Thriller', 'War', 'Western'
    ]
    
    def __init__(self, seed: int = 42):
        """Initialize factory with random seed"""
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def reset_seed(self, seed: Optional[int] = None):
        """Reset random seed"""
        if seed is not None:
            self.seed = seed
        self.rng = np.random.RandomState(self.seed)
    
    # -------------------------------------------------------------------------
    # MOVIE GENERATORS
    # -------------------------------------------------------------------------
    
    def create_movie(
        self,
        movie_id: int,
        title: Optional[str] = None,
        genres: Optional[str] = None,
        year: Optional[int] = None
    ) -> MovieData:
        """Create a single movie"""
        if title is None:
            if year is None:
                year = self.rng.randint(1950, 2025)
            title = f"Test Movie {movie_id} ({year})"
        
        if genres is None:
            n_genres = self.rng.randint(1, 4)
            genres = '|'.join(self.rng.choice(self.GENRES, n_genres, replace=False))
        
        return MovieData(
            movie_id=movie_id,
            title=title,
            genres=genres,
            year=year
        )
    
    def create_movies_df(
        self,
        n_movies: int,
        genre_distribution: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """Create movies DataFrame"""
        movies = []
        
        for i in range(1, n_movies + 1):
            if genre_distribution:
                # Weighted genre selection
                genres_list = list(genre_distribution.keys())
                weights = list(genre_distribution.values())
                weights = np.array(weights) / sum(weights)
                
                n_genres = self.rng.randint(1, 4)
                selected = self.rng.choice(
                    genres_list, n_genres, replace=False, p=weights[:len(genres_list)]
                )
                genres = '|'.join(selected)
            else:
                n_genres = self.rng.randint(1, 4)
                genres = '|'.join(self.rng.choice(self.GENRES, n_genres, replace=False))
            
            year = self.rng.randint(1950, 2025)
            movies.append({
                'movieId': i,
                'title': f'Test Movie {i} ({year})',
                'genres': genres
            })
        
        return pd.DataFrame(movies)
    
    # -------------------------------------------------------------------------
    # RATING GENERATORS
    # -------------------------------------------------------------------------
    
    def create_rating(
        self,
        user_id: int,
        movie_id: int,
        rating: Optional[float] = None,
        timestamp: Optional[int] = None
    ) -> RatingData:
        """Create a single rating"""
        if rating is None:
            rating = round(self.rng.uniform(0.5, 5.0) * 2) / 2  # 0.5 increments
        
        if timestamp is None:
            # Random timestamp in past 5 years
            base_ts = int(datetime.now().timestamp())
            timestamp = base_ts - self.rng.randint(0, 5 * 365 * 24 * 3600)
        
        return RatingData(
            user_id=user_id,
            movie_id=movie_id,
            rating=rating,
            timestamp=timestamp
        )
    
    def create_ratings_df(
        self,
        n_users: int,
        n_movies: int,
        n_ratings: int,
        rating_distribution: str = 'normal'
    ) -> pd.DataFrame:
        """Create ratings DataFrame"""
        ratings = []
        seen_pairs = set()
        
        while len(ratings) < n_ratings:
            user_id = self.rng.randint(1, n_users + 1)
            movie_id = self.rng.randint(1, n_movies + 1)
            
            pair = (user_id, movie_id)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            
            # Generate rating based on distribution
            if rating_distribution == 'normal':
                rating = self.rng.normal(3.5, 1.0)
            elif rating_distribution == 'uniform':
                rating = self.rng.uniform(0.5, 5.0)
            elif rating_distribution == 'biased_high':
                rating = self.rng.beta(5, 2) * 4.5 + 0.5
            elif rating_distribution == 'biased_low':
                rating = self.rng.beta(2, 5) * 4.5 + 0.5
            else:
                rating = self.rng.uniform(0.5, 5.0)
            
            # Clamp and round to nearest 0.5
            rating = max(0.5, min(5.0, rating))
            rating = round(rating * 2) / 2
            
            base_ts = int(datetime.now().timestamp())
            timestamp = base_ts - self.rng.randint(0, 5 * 365 * 24 * 3600)
            
            ratings.append({
                'userId': user_id,
                'movieId': movie_id,
                'rating': rating,
                'timestamp': timestamp
            })
        
        return pd.DataFrame(ratings)
    
    # -------------------------------------------------------------------------
    # DATASET GENERATORS
    # -------------------------------------------------------------------------
    
    def create_minimal_dataset(self) -> TestDataset:
        """Create minimal dataset for fast tests"""
        movies = self.create_movies_df(10)
        ratings = self.create_ratings_df(5, 10, 30)
        
        return TestDataset(
            movies=movies,
            ratings=ratings,
            n_users=5,
            n_movies=10,
            n_ratings=len(ratings),
            metadata={'type': 'minimal'}
        )
    
    def create_small_dataset(self) -> TestDataset:
        """Create small dataset for unit tests"""
        movies = self.create_movies_df(100)
        ratings = self.create_ratings_df(50, 100, 1000)
        
        return TestDataset(
            movies=movies,
            ratings=ratings,
            n_users=50,
            n_movies=100,
            n_ratings=len(ratings),
            metadata={'type': 'small'}
        )
    
    def create_medium_dataset(self) -> TestDataset:
        """Create medium dataset for integration tests"""
        movies = self.create_movies_df(500)
        ratings = self.create_ratings_df(500, 500, 25000)
        
        return TestDataset(
            movies=movies,
            ratings=ratings,
            n_users=500,
            n_movies=500,
            n_ratings=len(ratings),
            metadata={'type': 'medium'}
        )
    
    def create_large_dataset(self) -> TestDataset:
        """Create large dataset for load tests"""
        movies = self.create_movies_df(2000)
        ratings = self.create_ratings_df(2000, 2000, 100000)
        
        return TestDataset(
            movies=movies,
            ratings=ratings,
            n_users=2000,
            n_movies=2000,
            n_ratings=len(ratings),
            metadata={'type': 'large'}
        )
    
    # -------------------------------------------------------------------------
    # SPECIALIZED DATASETS
    # -------------------------------------------------------------------------
    
    def create_sparse_dataset(self, sparsity: float = 0.99) -> TestDataset:
        """Create sparse dataset with low density"""
        n_users = 100
        n_movies = 200
        max_ratings = n_users * n_movies
        n_ratings = int(max_ratings * (1 - sparsity))
        n_ratings = max(50, n_ratings)  # Minimum 50 ratings
        
        movies = self.create_movies_df(n_movies)
        ratings = self.create_ratings_df(n_users, n_movies, n_ratings)
        
        return TestDataset(
            movies=movies,
            ratings=ratings,
            n_users=n_users,
            n_movies=n_movies,
            n_ratings=len(ratings),
            metadata={'type': 'sparse', 'sparsity': sparsity}
        )
    
    def create_dense_dataset(self, density: float = 0.5) -> TestDataset:
        """Create dense dataset with high density"""
        n_users = 20
        n_movies = 30
        max_ratings = n_users * n_movies
        n_ratings = int(max_ratings * density)
        
        movies = self.create_movies_df(n_movies)
        ratings = self.create_ratings_df(n_users, n_movies, n_ratings)
        
        return TestDataset(
            movies=movies,
            ratings=ratings,
            n_users=n_users,
            n_movies=n_movies,
            n_ratings=len(ratings),
            metadata={'type': 'dense', 'density': density}
        )
    
    def create_cold_start_dataset(self) -> TestDataset:
        """Create dataset with cold start users and items"""
        movies = self.create_movies_df(100)
        
        # Create ratings but leave some users/movies with no ratings
        ratings_data = []
        seen_pairs = set()
        
        # Only use first 80 users and first 80 movies
        for _ in range(800):
            user_id = self.rng.randint(1, 81)  # Users 1-80 have ratings
            movie_id = self.rng.randint(1, 81)  # Movies 1-80 have ratings
            
            pair = (user_id, movie_id)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            
            ratings_data.append({
                'userId': user_id,
                'movieId': movie_id,
                'rating': round(self.rng.uniform(0.5, 5.0) * 2) / 2,
                'timestamp': int(datetime.now().timestamp())
            })
        
        ratings = pd.DataFrame(ratings_data)
        
        # Add a few cold start users (81-100) with very few ratings
        for user_id in range(81, 101):
            movie_id = self.rng.randint(1, 81)
            ratings = pd.concat([ratings, pd.DataFrame([{
                'userId': user_id,
                'movieId': movie_id,
                'rating': 4.0,
                'timestamp': int(datetime.now().timestamp())
            }])], ignore_index=True)
        
        return TestDataset(
            movies=movies,
            ratings=ratings,
            n_users=100,
            n_movies=100,
            n_ratings=len(ratings),
            metadata={
                'type': 'cold_start',
                'warm_users': list(range(1, 81)),
                'cold_users': list(range(81, 101)),
                'warm_movies': list(range(1, 81)),
                'cold_movies': list(range(81, 101))
            }
        )
    
    def create_genre_focused_dataset(self, primary_genre: str) -> TestDataset:
        """Create dataset focused on a specific genre"""
        # 70% of movies have the primary genre
        genre_dist = {primary_genre: 0.7}
        for g in self.GENRES:
            if g != primary_genre:
                genre_dist[g] = 0.3 / (len(self.GENRES) - 1)
        
        movies = self.create_movies_df(100, genre_distribution=genre_dist)
        ratings = self.create_ratings_df(50, 100, 1500)
        
        return TestDataset(
            movies=movies,
            ratings=ratings,
            n_users=50,
            n_movies=100,
            n_ratings=len(ratings),
            metadata={'type': 'genre_focused', 'primary_genre': primary_genre}
        )
    
    def create_biased_rating_dataset(self, bias: str = 'high') -> TestDataset:
        """Create dataset with biased rating distribution"""
        movies = self.create_movies_df(100)
        
        dist = 'biased_high' if bias == 'high' else 'biased_low'
        ratings = self.create_ratings_df(50, 100, 1000, rating_distribution=dist)
        
        return TestDataset(
            movies=movies,
            ratings=ratings,
            n_users=50,
            n_movies=100,
            n_ratings=len(ratings),
            metadata={'type': 'biased', 'bias': bias}
        )
    
    # -------------------------------------------------------------------------
    # USER PROFILE GENERATORS
    # -------------------------------------------------------------------------
    
    def create_user_profile(
        self,
        user_id: int,
        preferred_genres: Optional[List[str]] = None,
        avg_rating: Optional[float] = None,
        num_ratings: Optional[int] = None
    ) -> UserProfile:
        """Create a user profile"""
        if preferred_genres is None:
            n_genres = self.rng.randint(1, 4)
            preferred_genres = list(self.rng.choice(self.GENRES, n_genres, replace=False))
        
        if avg_rating is None:
            avg_rating = round(self.rng.uniform(2.5, 4.5), 2)
        
        if num_ratings is None:
            num_ratings = self.rng.randint(5, 100)
        
        return UserProfile(
            user_id=user_id,
            preferred_genres=preferred_genres,
            avg_rating=avg_rating,
            num_ratings=num_ratings
        )
    
    def create_user_profiles(self, n_users: int) -> List[UserProfile]:
        """Create multiple user profiles"""
        return [self.create_user_profile(i) for i in range(1, n_users + 1)]


# =============================================================================
# PYTEST FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def fixture_factory():
    """Provide fixture factory instance"""
    return FixtureFactory(seed=42)


@pytest.fixture(scope="module")
def minimal_test_data(fixture_factory):
    """Provide minimal test dataset"""
    return fixture_factory.create_minimal_dataset()


@pytest.fixture(scope="module")
def small_test_data(fixture_factory):
    """Provide small test dataset"""
    return fixture_factory.create_small_dataset()


@pytest.fixture(scope="module")
def medium_test_data(fixture_factory):
    """Provide medium test dataset"""
    return fixture_factory.create_medium_dataset()


@pytest.fixture(scope="function")
def fresh_minimal_data():
    """Provide fresh minimal dataset per test"""
    factory = FixtureFactory(seed=42)
    return factory.create_minimal_dataset()


@pytest.fixture(scope="function")
def sparse_data():
    """Provide sparse dataset"""
    factory = FixtureFactory(seed=42)
    return factory.create_sparse_dataset(sparsity=0.995)


@pytest.fixture(scope="function")
def dense_data():
    """Provide dense dataset"""
    factory = FixtureFactory(seed=42)
    return factory.create_dense_dataset(density=0.5)


@pytest.fixture(scope="function")
def cold_start_data():
    """Provide cold start dataset"""
    factory = FixtureFactory(seed=42)
    return factory.create_cold_start_dataset()


# =============================================================================
# MOCK OBJECTS
# =============================================================================

class MockRecommender:
    """Mock recommender for testing"""
    
    def __init__(self, recommendations: Optional[pd.DataFrame] = None):
        self.recommendations = recommendations
        self.fit_called = False
        self.recommend_called = False
    
    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        self.fit_called = True
        self.movies = movies
        self.ratings = ratings
    
    def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:
        self.recommend_called = True
        
        if self.recommendations is not None:
            return self.recommendations.head(n)
        
        # Return dummy recommendations
        return pd.DataFrame({
            'movieId': range(1, n + 1),
            'title': [f'Movie {i}' for i in range(1, n + 1)],
            'predicted_rating': [4.5 - i * 0.1 for i in range(n)]
        })
    
    def predict(self, user_id: int, movie_id: int) -> float:
        return 3.5


class MockDatabase:
    """Mock database for testing"""
    
    def __init__(self):
        self.data = {}
    
    def store(self, key: str, value: Any):
        self.data[key] = value
    
    def retrieve(self, key: str) -> Any:
        return self.data.get(key)
    
    def delete(self, key: str):
        if key in self.data:
            del self.data[key]
    
    def clear(self):
        self.data.clear()


@pytest.fixture
def mock_recommender():
    """Provide mock recommender"""
    return MockRecommender()


@pytest.fixture
def mock_database():
    """Provide mock database"""
    return MockDatabase()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def assert_valid_recommendations(recs: pd.DataFrame, n: int = 10):
    """Assert recommendations DataFrame is valid"""
    assert isinstance(recs, pd.DataFrame)
    assert len(recs) <= n
    assert 'movieId' in recs.columns or 'movie_id' in recs.columns
    
    if 'predicted_rating' in recs.columns:
        assert recs['predicted_rating'].between(0, 5).all()


def assert_valid_dataset(dataset: TestDataset):
    """Assert test dataset is valid"""
    assert isinstance(dataset.movies, pd.DataFrame)
    assert isinstance(dataset.ratings, pd.DataFrame)
    assert 'movieId' in dataset.movies.columns
    assert 'title' in dataset.movies.columns
    assert 'userId' in dataset.ratings.columns
    assert 'movieId' in dataset.ratings.columns
    assert 'rating' in dataset.ratings.columns


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    # Quick self-test
    factory = FixtureFactory(seed=42)
    
    print("Testing FixtureFactory...")
    
    # Test movie creation
    movie = factory.create_movie(1)
    print(f"✓ Created movie: {movie}")
    
    # Test rating creation
    rating = factory.create_rating(1, 1)
    print(f"✓ Created rating: {rating}")
    
    # Test dataset creation
    for name, method in [
        ('minimal', factory.create_minimal_dataset),
        ('small', factory.create_small_dataset),
        ('sparse', factory.create_sparse_dataset),
        ('dense', factory.create_dense_dataset),
        ('cold_start', factory.create_cold_start_dataset),
    ]:
        dataset = method()
        assert_valid_dataset(dataset)
        print(f"✓ Created {name} dataset: {dataset.n_ratings} ratings")
    
    print("\n✓ All tests passed!")
