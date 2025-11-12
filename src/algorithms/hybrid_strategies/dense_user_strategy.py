"""
Dense User Strategy

Recommendation strategy for users with dense rating history (≥ threshold ratings).
Uses full hybrid approach with all four algorithms and calculated weights.

Author: CineMatch Development Team
Date: November 12, 2025
"""

from typing import Dict, Any
import pandas as pd
from .base_strategy import BaseRecommendationStrategy


class DenseUserStrategy(BaseRecommendationStrategy):
    """
    Strategy for users with dense rating history.
    
    Approach:
    - Uses all four algorithms: SVD, User KNN, Item KNN, and Content-Based
    - Applies dynamically calculated weights based on algorithm performance
    - Provides most accurate recommendations due to rich user profile
    
    Weight Distribution:
    - Uses weights calculated by HybridRecommender based on:
      * Algorithm RMSE scores
      * Data availability
      * Historical performance
    
    Typical Use: Users with ≥50 ratings (above sparse_user_threshold)
    """
    
    def __init__(self, threshold: int = 50):
        """
        Initialize dense user strategy.
        
        Args:
            threshold: Number of ratings at/above which a user is considered dense
        """
        super().__init__(name=f"Dense User (≥ {threshold} ratings)")
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
        Generate recommendations for a dense user.
        
        Uses all four algorithms with dynamically calculated weights
        based on their performance metrics.
        """
        print(f"  • Dense user profile - using {self.name} strategy")
        print(f"  • Using weights: SVD={weights['svd']:.2f}, UserKNN={weights['user_knn']:.2f}, "
              f"ItemKNN={weights['item_knn']:.2f}, CBF={weights['content_based']:.2f}")
        
        # Define algorithm configuration using calculated weights
        algorithm_configs = [
            {
                'model': algorithms['svd'],
                'weight': weights['svd'],
                'name': 'SVD'
            },
            {
                'model': algorithms['user_knn'],
                'weight': weights['user_knn'],
                'name': 'UserKNN'
            },
            {
                'model': algorithms['item_knn'],
                'weight': weights['item_knn'],
                'name': 'ItemKNN'
            },
            {
                'model': algorithms['content_based'],
                'weight': weights['content_based'],
                'name': 'CBF'
            }
        ]
        
        # Collect recommendations from all algorithms
        all_recs_list = self._collect_recommendations(
            user_id, n, exclude_rated, algorithm_configs
        )
        
        if not all_recs_list:
            # Fallback to SVD only if collection fails
            print("    ⚠️  Falling back to SVD only")
            return algorithms['svd'].get_recommendations(user_id, n, exclude_rated)
        
        # Aggregate recommendations
        total_recs = sum(len(r) for r in all_recs_list)
        print(f"    ✓ Collected {total_recs} total recommendations from {len(all_recs_list)} algorithms")
        print(f"  • Aggregating recommendations...")
        
        all_recs = pd.concat(all_recs_list)
        return aggregate_fn(all_recs, n)
    
    def get_description(self) -> str:
        """Describe when this strategy is used."""
        return (
            f"Dense User Strategy: For users with ≥ {self.threshold} ratings. "
            "Uses full hybrid approach with all four algorithms (SVD, User KNN, "
            "Item KNN, Content-Based) weighted by their performance metrics. "
            "Provides most accurate recommendations due to rich user profile."
        )
