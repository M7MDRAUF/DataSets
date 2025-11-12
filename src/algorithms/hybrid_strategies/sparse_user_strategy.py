"""
Sparse User Strategy

Recommendation strategy for users with sparse rating history (< threshold ratings).
Emphasizes User KNN to leverage similar users' preferences.

Author: CineMatch Development Team
Date: November 12, 2025
"""

from typing import Dict, Any
import pandas as pd
from .base_strategy import BaseRecommendationStrategy


class SparseUserStrategy(BaseRecommendationStrategy):
    """
    Strategy for users with sparse rating history.
    
    Approach:
    - Emphasizes User KNN (finds similar users with similar sparse patterns)
    - Supplements with SVD (global patterns) and Content-Based (content similarity)
    - De-emphasizes Item KNN (requires more data to be effective)
    
    Weight Distribution:
    - User KNN: 50% (leverages similar users effectively)
    - SVD: 30% (fills gaps with global patterns)
    - Content-Based: 20% (content-based fallback)
    
    Typical Use: Users with 1-50 ratings (below sparse_user_threshold)
    """
    
    def __init__(self, threshold: int = 50):
        """
        Initialize sparse user strategy.
        
        Args:
            threshold: Number of ratings below which a user is considered sparse
        """
        super().__init__(name=f"Sparse User (< {threshold} ratings)")
        self.threshold = threshold
    
    def get_recommendations(
        self,
        user_id: int,
        n: int,
        exclude_rated: bool,
        algorithms: Dict[str, Any],
        weights: Dict[str, float],
        aggregate_fn: callable
    ) -> pd.DataFrame:
        """
        Generate recommendations for a sparse user.
        
        Uses User KNN + SVD + Content-Based with weights optimized for
        sparse rating profiles.
        """
        print(f"  • Sparse user profile - using {self.name} strategy")
        
        # Define algorithm configuration for sparse users
        algorithm_configs = [
            {
                'model': algorithms['user_knn'],
                'weight': 0.5,
                'name': 'UserKNN'
            },
            {
                'model': algorithms['svd'],
                'weight': 0.3,
                'name': 'SVD'
            },
            {
                'model': algorithms['content_based'],
                'weight': 0.2,
                'name': 'CBF'
            }
        ]
        
        # Collect recommendations from all algorithms
        all_recs_list = self._collect_recommendations(
            user_id, n, exclude_rated, algorithm_configs
        )
        
        if not all_recs_list:
            # Fallback to User KNN only if collection fails
            print("    ⚠️  Falling back to User KNN only")
            return algorithms['user_knn'].get_recommendations(user_id, n, exclude_rated)
        
        # Aggregate recommendations
        print(f"    ✓ Collected {sum(len(r) for r in all_recs_list)} total recommendations")
        all_recs = pd.concat(all_recs_list)
        return aggregate_fn(all_recs, n)
    
    def get_description(self) -> str:
        """Describe when this strategy is used."""
        return (
            f"Sparse User Strategy: For users with < {self.threshold} ratings. "
            "Emphasizes User KNN (50%) to find similar users with sparse patterns, "
            "supplemented by SVD (30%) and Content-Based (20%) for robustness."
        )
