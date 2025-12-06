"""
CineMatch V2.1.6 - Test Configuration

Pytest configuration and shared fixtures for the CineMatch test suite.

This module provides:
    - Path configuration for test imports
    - Shared fixtures for sample data (ratings, movies)
    - Mock factories for algorithm testing
    - Test utilities for common assertions

Usage:
    Fixtures defined here are automatically available to all test files.
    
    Example test using fixtures:
        def test_recommendations(sample_ratings_df, sample_movies_df):
            # Fixtures are injected automatically
            assert len(sample_ratings_df) > 0

Fixture Categories:
    - Data Fixtures: sample_ratings_df, sample_movies_df, sample_user_ids
    - Algorithm Fixtures: mock_algorithm_manager, trained_svd
    - Utility Fixtures: temp_model_dir, clean_cache

Author: CineMatch Development Team
Date: December 2025
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Generator, List, Dict, Any
from unittest.mock import Mock, MagicMock

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# DATA FIXTURES
# =============================================================================

@pytest.fixture(scope="function")
def sample_movies_df() -> pd.DataFrame:
    """
    Create a sample movies DataFrame for testing.
    
    Note: Uses function scope to ensure test isolation and prevent
    state leakage between tests that may modify the DataFrame.
    
    Returns:
        pd.DataFrame: Sample movie data with movieId, title, and genres columns.
        Contains 10 diverse movies covering various genres for comprehensive testing.
    
    Example:
        def test_movie_filtering(sample_movies_df):
            action_movies = sample_movies_df[
                sample_movies_df['genres'].str.contains('Action')
            ]
            assert len(action_movies) > 0
    """
    return pd.DataFrame({
        'movieId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'title': [
            'Toy Story (1995)',
            'Jumanji (1995)',
            'Grumpier Old Men (1995)',
            'Waiting to Exhale (1995)',
            'Father of the Bride Part II (1995)',
            'Heat (1995)',
            'Sabrina (1995)',
            'Tom and Huck (1995)',
            'Sudden Death (1995)',
            'GoldenEye (1995)'
        ],
        'genres': [
            'Adventure|Animation|Children|Comedy|Fantasy',
            'Adventure|Children|Fantasy',
            'Comedy|Romance',
            'Comedy|Drama|Romance',
            'Comedy',
            'Action|Crime|Thriller',
            'Comedy|Romance',
            'Adventure|Children',
            'Action',
            'Action|Adventure|Thriller'
        ]
    })


@pytest.fixture(scope="function")
def sample_ratings_df() -> pd.DataFrame:
    """
    Create a sample ratings DataFrame for testing.
    
    Note: Uses function scope to ensure test isolation and prevent
    state leakage between tests that may modify the DataFrame.
    
    Returns:
        pd.DataFrame: Sample rating data with userId, movieId, and rating columns.
        Contains ratings from 6 users across 10 movies, repeated to ensure
        sufficient data for algorithm training.
    
    Note:
        The data is repeated 10x to provide enough samples for KNN algorithms
        which require minimum neighbor counts.
    
    Example:
        def test_user_ratings(sample_ratings_df):
            user_1_ratings = sample_ratings_df[sample_ratings_df['userId'] == 1]
            assert user_1_ratings['rating'].mean() > 0
    """
    base_data = {
        'userId': [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6],
        'movieId': [1, 2, 3, 6, 1, 2, 4, 2, 3, 4, 5, 1, 3, 6, 2, 4, 5, 7, 1, 3],
        'rating': [5.0, 4.0, 3.5, 4.5, 4.0, 5.0, 3.0, 4.5, 5.0, 3.5, 4.0, 3.5, 4.0, 5.0, 4.5, 4.0, 4.5, 3.5, 5.0, 4.0]
    }
    # Repeat data 10x for sufficient training samples
    return pd.DataFrame({
        k: v * 10 for k, v in base_data.items()
    })


@pytest.fixture(scope="function")
def sample_user_ids() -> List[int]:
    """
    Provide a list of valid test user IDs.
    
    Note: Uses function scope for consistency with other data fixtures.
    
    Returns:
        List[int]: User IDs that exist in sample_ratings_df for testing.
    
    Example:
        def test_recommendations_for_users(sample_user_ids, recommender):
            for user_id in sample_user_ids:
                recs = recommender.recommend(user_id, n=5)
                assert len(recs) > 0
    """
    return [1, 2, 3, 4, 5, 6]


@pytest.fixture(scope="function")
def sample_movie_ids() -> List[int]:
    """
    Provide a list of valid test movie IDs.
    
    Note: Uses function scope for consistency with other data fixtures.
    
    Returns:
        List[int]: Movie IDs that exist in sample_movies_df for testing.
    """
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# =============================================================================
# ALGORITHM FIXTURES
# =============================================================================

@pytest.fixture
def mock_algorithm_manager() -> MagicMock:
    """
    Create a mock AlgorithmManager for isolated testing.
    
    Returns:
        MagicMock: A mock manager that simulates algorithm operations
        without loading actual models.
    
    Example:
        def test_algorithm_switching(mock_algorithm_manager):
            mock_algorithm_manager.get_algorithm.return_value = Mock()
            algo = mock_algorithm_manager.get_algorithm('SVD')
            assert algo is not None
    """
    manager = MagicMock()
    manager.is_initialized = True
    manager.available_algorithms = ['SVD', 'USER_KNN', 'ITEM_KNN', 'CONTENT_BASED', 'HYBRID']
    manager.current_algorithm = 'SVD'
    return manager


@pytest.fixture
def mock_recommender() -> Mock:
    """
    Create a mock recommender for testing recommendation workflows.
    
    Returns:
        Mock: A mock recommender with pre-configured return values.
    """
    recommender = Mock()
    recommender.name = "MockRecommender"
    recommender.is_trained = True
    recommender.recommend.return_value = pd.DataFrame({
        'movieId': [1, 2, 3],
        'title': ['Movie 1', 'Movie 2', 'Movie 3'],
        'predicted_rating': [4.5, 4.2, 4.0],
        'genres': ['Action', 'Comedy', 'Drama']
    })
    return recommender


# =============================================================================
# UTILITY FIXTURES
# =============================================================================

@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """
    Create a temporary directory for model files.
    
    Args:
        tmp_path: pytest built-in fixture for temporary paths.
    
    Returns:
        Path: Path to temporary models directory.
    
    Example:
        def test_model_save(temp_model_dir, trained_model):
            model_path = temp_model_dir / "test_model.pkl"
            trained_model.save(model_path)
            assert model_path.exists()
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


@pytest.fixture
def clean_cache() -> Generator[None, None, None]:
    """
    Fixture that clears Streamlit cache before and after tests.
    
    Yields:
        None: Provides clean cache state for test execution.
    
    Note:
        Only effective when running in Streamlit context.
    """
    try:
        import streamlit as st
        st.cache_data.clear()
        st.cache_resource.clear()
    except (ImportError, RuntimeError):
        pass  # Not in Streamlit context
    
    yield
    
    try:
        import streamlit as st
        st.cache_data.clear()
        st.cache_resource.clear()
    except (ImportError, RuntimeError):
        pass


# =============================================================================
# PYTEST HOOKS
# =============================================================================

def pytest_configure(config):
    """
    Configure custom pytest markers.
    
    Markers:
        - slow: Tests that take > 5 seconds
        - integration: Tests requiring external resources
        - unit: Fast isolated unit tests
        - e2e: End-to-end tests
    """
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")

