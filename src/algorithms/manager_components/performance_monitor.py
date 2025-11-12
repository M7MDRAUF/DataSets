"""
Performance Monitor

Responsible for tracking and comparing algorithm performance metrics.
Handles the "how well" aspect of algorithm management.

Author: CineMatch Development Team
Date: November 12, 2025
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.algorithms.base_recommender import BaseRecommender
from .algorithm_factory import AlgorithmType, AlgorithmFactory


class PerformanceMonitor:
    """
    Monitor and compare algorithm performance.
    
    Responsibilities:
    - Tracking algorithm performance metrics
    - Comparing algorithms side-by-side
    - Caching metrics for efficiency
    - Generating performance reports
    """
    
    def __init__(self, factory: AlgorithmFactory):
        """
        Initialize performance monitor.
        
        Args:
            factory: AlgorithmFactory instance for algorithm information
        """
        self.factory = factory
        self._metrics_cache: Dict[AlgorithmType, Dict[str, Any]] = {}
    
    def get_performance_comparison(self, 
                                  algorithms: Dict[AlgorithmType, BaseRecommender]) -> pd.DataFrame:
        """
        Get performance comparison of all trained algorithms.
        
        Args:
            algorithms: Dictionary of algorithm_type -> algorithm_instance
            
        Returns:
            DataFrame with performance metrics for comparison
        """
        performance_data = []
        
        for algorithm_type, algorithm in algorithms.items():
            if algorithm.is_trained:
                info = self.factory.get_algorithm_info(algorithm_type)
                performance_data.append({
                    'Algorithm': algorithm.name,
                    'RMSE': f"{algorithm.metrics.rmse:.4f}",
                    'Training Time': f"{algorithm.metrics.training_time:.1f}s",
                    'Coverage': f"{algorithm.metrics.coverage:.1f}%",
                    'Memory Usage': f"{algorithm.metrics.memory_usage_mb:.1f} MB",
                    'Prediction Speed': f"{algorithm.metrics.prediction_time:.4f}s",
                    'Icon': info.get('icon', '🎯'),
                    'Interpretability': info.get('interpretability', 'Medium')
                })
        
        return pd.DataFrame(performance_data)
    
    def get_algorithm_metrics(self,
                            algorithm: BaseRecommender,
                            algorithm_type: AlgorithmType,
                            training_data: Optional[tuple] = None,
                            use_cache: bool = True) -> Dict[str, Any]:
        """
        Get detailed performance metrics for a specific algorithm.
        
        Args:
            algorithm: The algorithm instance
            algorithm_type: Type of the algorithm
            training_data: Optional tuple of (ratings_df, movies_df) for evaluation
            use_cache: Whether to use cached metrics if available
            
        Returns:
            Dictionary with detailed metrics
        """
        # Check cache first
        if use_cache and algorithm_type in self._metrics_cache:
            print(f"✓ Using cached metrics for {algorithm_type.value}")
            return self._metrics_cache[algorithm_type]
        
        if not algorithm.is_trained:
            return {
                "algorithm": algorithm_type.value,
                "status": "Not trained",
                "metrics": {}
            }
        
        # Get basic algorithm info
        info = self.factory.get_algorithm_info(algorithm_type)
        
        # Basic metrics from algorithm
        metrics = {
            "algorithm": algorithm_type.value,
            "name": algorithm.name,
            "status": "Trained",
            "rmse": algorithm.metrics.rmse,
            "training_time": algorithm.metrics.training_time,
            "coverage": algorithm.metrics.coverage,
            "memory_mb": algorithm.metrics.memory_usage_mb,
            "prediction_time": algorithm.metrics.prediction_time,
            "interpretability": info.get('interpretability', 'Medium'),
            "complexity": info.get('complexity', 'Medium'),
            "speed": info.get('speed', 'Medium')
        }
        
        # Calculate additional metrics if training data is available
        if training_data is not None:
            additional_metrics = self._calculate_evaluation_metrics(
                algorithm, training_data[0], sample_size=1000
            )
            metrics.update(additional_metrics)
        
        # Cache the metrics
        self._metrics_cache[algorithm_type] = metrics
        return metrics
    
    def _calculate_evaluation_metrics(self,
                                     algorithm: BaseRecommender,
                                     ratings_df: pd.DataFrame,
                                     sample_size: int = 1000) -> Dict[str, Any]:
        """
        Calculate evaluation metrics on a sample of data.
        
        Args:
            algorithm: The algorithm to evaluate
            ratings_df: Ratings DataFrame
            sample_size: Number of samples to use for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        if len(ratings_df) <= sample_size:
            test_sample = ratings_df
        else:
            test_sample = ratings_df.sample(n=sample_size, random_state=42)
        
        predictions = []
        actuals = []
        
        print(f"🔍 Evaluating on {len(test_sample)} samples...")
        
        for i, (_, row) in enumerate(test_sample.iterrows()):
            try:
                pred = algorithm.predict(row['userId'], row['movieId'])
                if pred is not None and pred > 0:
                    predictions.append(pred)
                    actuals.append(row['rating'])
            except Exception as e:
                # Skip failed predictions
                if i < 5:  # Only print first 5 errors
                    print(f"    ⚠️  Prediction failed for user {row['userId']}, movie {row['movieId']}: {e}")
                continue
        
        print(f"    ✓ Got {len(predictions)} valid predictions")
        
        # Calculate metrics if we have enough predictions
        metrics = {}
        if len(predictions) > 10:
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            
            # Calculate prediction accuracy (% within 0.5 stars)
            errors = np.abs(np.array(predictions) - np.array(actuals))
            accuracy_05 = np.mean(errors <= 0.5) * 100
            accuracy_10 = np.mean(errors <= 1.0) * 100
            
            metrics.update({
                'eval_rmse': rmse,
                'eval_mae': mae,
                'accuracy_within_0.5': accuracy_05,
                'accuracy_within_1.0': accuracy_10,
                'sample_size': len(predictions)
            })
        else:
            metrics.update({
                'eval_rmse': None,
                'eval_mae': None,
                'sample_size': len(predictions),
                'note': 'Insufficient predictions for metrics'
            })
        
        return metrics
    
    def clear_cache(self, algorithm_type: Optional[AlgorithmType] = None) -> None:
        """
        Clear metrics cache.
        
        Args:
            algorithm_type: If provided, clear only this algorithm. Otherwise clear all.
        """
        if algorithm_type:
            if algorithm_type in self._metrics_cache:
                del self._metrics_cache[algorithm_type]
                print(f"✓ Cleared metrics cache for {algorithm_type.value}")
        else:
            self._metrics_cache.clear()
            print("✓ Cleared all metrics cache")
    
    def get_cached_metrics_count(self) -> int:
        """Get number of algorithms with cached metrics."""
        return len(self._metrics_cache)
    
    def generate_performance_report(self, 
                                   algorithms: Dict[AlgorithmType, BaseRecommender]) -> str:
        """
        Generate a human-readable performance report.
        
        Args:
            algorithms: Dictionary of algorithm_type -> algorithm_instance
            
        Returns:
            Formatted string report
        """
        report_lines = ["=" * 80]
        report_lines.append("ALGORITHM PERFORMANCE REPORT")
        report_lines.append("=" * 80)
        
        for algorithm_type, algorithm in algorithms.items():
            if not algorithm.is_trained:
                continue
            
            info = self.factory.get_algorithm_info(algorithm_type)
            report_lines.append(f"\n{info.get('icon', '🎯')} {algorithm.name}")
            report_lines.append("-" * 80)
            report_lines.append(f"RMSE: {algorithm.metrics.rmse:.4f}")
            report_lines.append(f"Training Time: {algorithm.metrics.training_time:.2f}s")
            report_lines.append(f"Coverage: {algorithm.metrics.coverage:.1f}%")
            report_lines.append(f"Memory Usage: {algorithm.metrics.memory_usage_mb:.1f} MB")
            report_lines.append(f"Prediction Speed: {algorithm.metrics.prediction_time:.4f}s")
            report_lines.append(f"Complexity: {info.get('complexity', 'N/A')}")
            report_lines.append(f"Interpretability: {info.get('interpretability', 'N/A')}")
        
        report_lines.append("\n" + "=" * 80)
        return "\n".join(report_lines)
