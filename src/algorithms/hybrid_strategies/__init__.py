"""
CineMatch V2.0.0 - Hybrid Recommendation Strategies

Strategy Pattern implementation for user profile-based recommendation strategies.

Phase 2 Refactoring: Extracted from hybrid_recommender.py to enable:
- Clear separation of concerns
- Easy addition of new strategies
- Independent testing of each strategy
- Maintainable strategy selection logic

Author: CineMatch Development Team
Date: November 12, 2025
"""

from .base_strategy import BaseRecommendationStrategy
from .new_user_strategy import NewUserStrategy
from .sparse_user_strategy import SparseUserStrategy
from .dense_user_strategy import DenseUserStrategy

__all__ = [
    'BaseRecommendationStrategy',
    'NewUserStrategy',
    'SparseUserStrategy',
    'DenseUserStrategy'
]
