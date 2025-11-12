"""
CineMatch V2.1.0 - Feature Engineering Module

This module provides reusable feature engineering components for
content-based filtering and other recommendation algorithms.

Modules:
- movie_features: Extract and combine movie features (genres, tags, titles)
- similarity_matrix: Compute similarity matrices
- user_profile: Build user profiles from rating history

Author: CineMatch Development Team
Date: November 12, 2025
Version: 2.1.0
"""

from .movie_features import MovieFeatureExtractor
from .similarity_matrix import SimilarityMatrixBuilder  
from .user_profile import UserProfileBuilder

__all__ = [
    'MovieFeatureExtractor',
    'SimilarityMatrixBuilder',
    'UserProfileBuilder'
]

__version__ = '2.1.0'
