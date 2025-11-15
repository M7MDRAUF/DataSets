"""
CineMatch V1.0.0 - Base Recommendation Algorithm Interface

Abstract base class defining the standard API for all recommendation algorithms.
Ensures consistency and interchangeability between SVD, KNN, and hybrid approaches.

Author: CineMatch Development Team
Date: November 7, 2025
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
import time


class AlgorithmMetrics:
    """Container for algorithm performance metrics"""
    
    def __init__(self):
        self.name: str = ""
        self.rmse: float = 0.0
        self.training_time: float = 0.0
        self.prediction_time: float = 0.0
        self.memory_usage_mb: float = 0.0
        self.coverage: float = 0.0  # % of items that can be recommended
        self.diversity: float = 0.0  # Average diversity of recommendations
        self.novelty: float = 0.0   # Average novelty (popularity-based)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for display"""
        return {
            'Algorithm': self.name,
            'RMSE': f"{self.rmse:.4f}",
            'Training Time': f"{self.training_time:.1f}s",
            'Prediction Time': f"{self.prediction_time:.3f}s",
            'Memory Usage': f"{self.memory_usage_mb:.1f} MB",
            'Coverage': f"{self.coverage:.1f}%",
            'Diversity': f"{self.diversity:.3f}",
            'Novelty': f"{self.novelty:.3f}"
        }


class BaseRecommender(ABC):
    """
    Abstract base class for all recommendation algorithms.
    
    Defines the standard interface that all algorithms must implement,
    ensuring consistency and interchangeability.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize the recommender.
        
        Args:
            name: Human-readable algorithm name
            **kwargs: Algorithm-specific parameters
        """
        self.name = name
        self.is_trained = False
        self.metrics = AlgorithmMetrics()
        self.metrics.name = name
        
        # Data containers
        self.ratings_df: Optional[pd.DataFrame] = None
        self.movies_df: Optional[pd.DataFrame] = None
        self.user_ids: Optional[np.ndarray] = None
        self.movie_ids: Optional[np.ndarray] = None
        
        # Algorithm-specific parameters
        self.params = kwargs
        
        # Performance tracking
        self._last_prediction_time = 0.0
    
    @abstractmethod
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        """
        Train the recommendation algorithm on the provided data.
        
        Args:
            ratings_df: DataFrame with columns ['userId', 'movieId', 'rating']
            movies_df: DataFrame with columns ['movieId', 'title', 'genres']
        """
        pass
    
    @abstractmethod
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for a specific user-movie pair.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating (float)
        """
        pass
    
    @abstractmethod
    def get_recommendations(
        self,
        user_id: int,
        n: int = 10,
        exclude_rated: bool = True
    ) -> pd.DataFrame:
        """
        Generate top-N recommendations for a user.
        
        Args:
            user_id: User ID to generate recommendations for
            n: Number of recommendations to return
            exclude_rated: Whether to exclude already-rated movies
            
        Returns:
            DataFrame with columns ['movieId', 'predicted_rating', 'title', 'genres']
        """
        pass
    
    @abstractmethod
    def get_similar_items(
        self,
        item_id: int,
        n: int = 10
    ) -> pd.DataFrame:
        """
        Find items similar to the given item.
        
        Args:
            item_id: Movie ID to find similar movies for
            n: Number of similar items to return
            
        Returns:
            DataFrame with columns ['movieId', 'similarity', 'title', 'genres']
        """
        pass
    
    def get_user_history(self, user_id: int) -> pd.DataFrame:
        """
        Get user's rating history.
        
        Args:
            user_id: User ID
            
        Returns:
            DataFrame with user's rated movies
        """
        if self.ratings_df is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id].copy()
        
        # Merge with movie info
        if self.movies_df is not None:
            user_ratings = user_ratings.merge(
                self.movies_df[['movieId', 'title', 'genres']],
                on='movieId',
                how='left'
            )
        
        # Sort by rating (descending)
        user_ratings = user_ratings.sort_values('rating', ascending=False)
        
        return user_ratings
    
    def validate_user_exists(self, user_id: int) -> bool:
        """Check if user exists in the dataset"""
        if self.ratings_df is None:
            return False
        return user_id in self.ratings_df['userId'].values
    
    def validate_movie_exists(self, movie_id: int) -> bool:
        """Check if movie exists in the dataset"""
        if self.movies_df is None:
            return False
        return movie_id in self.movies_df['movieId'].values
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        Get algorithm information and capabilities.
        
        Returns:
            Dictionary with algorithm metadata
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'is_trained': self.is_trained,
            'parameters': self.params,
            'capabilities': self._get_capabilities(),
            'description': self._get_description(),
            'strengths': self._get_strengths(),
            'ideal_for': self._get_ideal_use_cases()
        }
    
    @abstractmethod
    def _get_capabilities(self) -> List[str]:
        """Return list of algorithm capabilities"""
        pass
    
    @abstractmethod
    def _get_description(self) -> str:
        """Return human-readable algorithm description"""
        pass
    
    @abstractmethod
    def _get_strengths(self) -> List[str]:
        """Return list of algorithm strengths"""
        pass
    
    @abstractmethod
    def _get_ideal_use_cases(self) -> List[str]:
        """Return list of ideal use cases"""
        pass
    
    def save_model(self, path: Path) -> None:
        """
        Save trained model to disk.
        
        Args:
            path: Path to save the model
        """
        import joblib
        
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Save model data
        model_data = {
            'name': self.name,
            'params': self.params,
            'metrics': self.metrics,
            'is_trained': self.is_trained,
            'model_state': self._get_model_state()
        }
        
        joblib.dump(model_data, path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path: Path) -> None:
        """
        Load trained model from disk.
        
        Args:
            path: Path to load the model from
        """
        import joblib
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model_data = joblib.load(path)
        
        self.name = model_data['name']
        self.params = model_data['params']
        self.metrics = model_data.get('metrics', AlgorithmMetrics())
        self.is_trained = model_data['is_trained']
        
        # If metrics are empty/default (from old model files), mark for recalculation
        if self.metrics.rmse == 0.0 and self.metrics.training_time == 0.0:
            # Reset metrics to trigger recalculation when needed
            self.metrics = AlgorithmMetrics()
            self.metrics.name = self.name
        
        self._set_model_state(model_data['model_state'])
        print(f"✓ Model loaded from {path}")
    
    @abstractmethod
    def _get_model_state(self) -> Dict[str, Any]:
        """Get algorithm-specific state for saving"""
        pass
    
    @abstractmethod
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Set algorithm-specific state from loading"""
        pass
    
    def _start_prediction_timer(self) -> None:
        """Start timing for prediction performance"""
        self._prediction_start_time = time.time()
    
    def _end_prediction_timer(self) -> None:
        """End timing for prediction performance"""
        self._last_prediction_time = time.time() - self._prediction_start_time
        self.metrics.prediction_time = self._last_prediction_time
    
    def get_performance_summary(self) -> str:
        """Get formatted performance summary"""
        return f"""
{self.name} Performance Summary:
─────────────────────────────
• RMSE: {self.metrics.rmse:.4f}
• Training Time: {self.metrics.training_time:.1f}s  
• Last Prediction: {self.metrics.prediction_time:.3f}s
• Memory Usage: {self.metrics.memory_usage_mb:.1f} MB
• Coverage: {self.metrics.coverage:.1f}%
• Status: {'✓ Trained' if self.is_trained else '✗ Not Trained'}
        """.strip()
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.name} ({'Trained' if self.is_trained else 'Not Trained'})"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return f"{self.__class__.__name__}(name='{self.name}', trained={self.is_trained})"