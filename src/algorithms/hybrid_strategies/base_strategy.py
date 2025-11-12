"""
Base Strategy for Hybrid Recommendations

Abstract base class defining the interface for all recommendation strategies.

Author: CineMatch Development Team
Date: November 12, 2025
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import pandas as pd


class BaseRecommendationStrategy(ABC):
    """
    Abstract base class for recommendation strategies.
    
    Each strategy implements a different approach to generating recommendations
    based on user profile characteristics (new, sparse, or dense ratings).
    """
    
    def __init__(self, name: str):
        """
        Initialize the base strategy.
        
        Args:
            name: Human-readable name for the strategy
        """
        self.name = name
    
    @abstractmethod
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
        Generate recommendations using this strategy.
        
        Args:
            user_id: ID of the user to generate recommendations for
            n: Number of recommendations to return
            exclude_rated: Whether to exclude already-rated items
            algorithms: Dictionary of available algorithm instances
                       (e.g., {'svd': svd_model, 'user_knn': user_knn_model, ...})
            weights: Dictionary of algorithm weights
                     (e.g., {'svd': 0.34, 'user_knn': 0.23, ...})
            aggregate_fn: Function to aggregate recommendations from multiple algorithms
        
        Returns:
            DataFrame with top n recommendations
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """
        Get a description of this strategy.
        
        Returns:
            String describing when and why this strategy is used
        """
        pass
    
    def _collect_recommendations(
        self,
        user_id: int,
        n: int,
        exclude_rated: bool,
        algorithm_configs: List[Dict[str, Any]]
    ) -> List[pd.DataFrame]:
        """
        Helper method to collect recommendations from multiple algorithms.
        
        Args:
            user_id: User ID
            n: Number of recommendations per algorithm
            exclude_rated: Whether to exclude rated items
            algorithm_configs: List of dicts with 'model', 'weight', and 'name' keys
        
        Returns:
            List of DataFrames with recommendations (weight column added)
        """
        recommendations = []
        import time as perf_time
        
        for i, config in enumerate(algorithm_configs):
            try:
                t0 = perf_time.time()
                model = config['model']
                recs = model.get_recommendations(user_id, n, exclude_rated)
                t1 = perf_time.time()
                
                # Add weight column
                recs = recs.copy()
                recs['weight'] = config['weight']
                recommendations.append(recs)
                
                print(f"    ✓ {config['name']}: {len(recs)} recommendations in {t1-t0:.2f}s")
                
            except Exception as e:
                print(f"    ⚠️  {config['name']} failed: {e}")
                continue
        
        return recommendations
