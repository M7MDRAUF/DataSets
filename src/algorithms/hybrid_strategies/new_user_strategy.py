"""
New User Strategy

Recommendation strategy for users with no rating history (cold-start problem).
Uses algorithms that don't require user history: SVD, Item KNN, and Content-Based.

Author: CineMatch Development Team
Date: November 12, 2025
"""

from typing import Dict, Any
import pandas as pd
from .base_strategy import BaseRecommendationStrategy


class NewUserStrategy(BaseRecommendationStrategy):
    """
    Strategy for new users with zero ratings.
    
    Approach:
    - Emphasizes algorithms that work well for cold-start scenarios
    - Uses SVD (global patterns), Item KNN (popular items), and Content-Based (genre/metadata)
    - Avoids User KNN (requires user similarity which doesn't exist for new users)
    
    Weight Distribution:
    - SVD: 40% (captures overall rating patterns)
    - Item KNN: 30% (popular items among all users)
    - Content-Based: 30% (helps with cold start via content similarity)
    """
    
    def __init__(self):
        super().__init__(name="New User (Cold-Start)")
    
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
        Generate recommendations for a new user.
        
        Uses SVD + Item KNN + Content-Based with predefined weights optimized
        for cold-start scenarios.
        """
        print(f"  • New user detected - using {self.name} strategy")
        
        # Define algorithm configuration for new users
        algorithm_configs = [
            {
                'model': algorithms['svd'],
                'weight': 0.4,
                'name': 'SVD'
            },
            {
                'model': algorithms['item_knn'],
                'weight': 0.3,
                'name': 'ItemKNN'
            },
            {
                'model': algorithms['content_based'],
                'weight': 0.3,
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
        print(f"    ✓ Collected {sum(len(r) for r in all_recs_list)} total recommendations")
        all_recs = pd.concat(all_recs_list)
        return aggregate_fn(all_recs, n)
    
    def get_description(self) -> str:
        """Describe when this strategy is used."""
        return (
            "New User Strategy: For users with 0 ratings (cold-start). "
            "Uses SVD (40%), Item KNN (30%), and Content-Based (30%) to "
            "provide recommendations based on global patterns and content similarity "
            "rather than user history."
        )
