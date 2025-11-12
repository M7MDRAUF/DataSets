"""
CineMatch V2.0.0 - Algorithm Manager

Central management system for all recommendation algorithms.
Handles instantiation, switching, lifecycle management, and intelligent caching.

Phase 3 Refactoring: Decomposed into specialized components
- AlgorithmFactory: Creates and manages algorithm instances
- LifecycleManager: Handles loading, caching, switching
- PerformanceMonitor: Tracks and compares metrics

Author: CineMatch Development Team
Date: November 12, 2025
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Any, List
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.algorithms.base_recommender import BaseRecommender

# Phase 3: Import decomposed components
from src.algorithms.manager_components import (
    AlgorithmFactory,
    LifecycleManager,
    PerformanceMonitor
)
from src.algorithms.manager_components.algorithm_factory import AlgorithmType


class AlgorithmManager:
    """
    Central manager for all recommendation algorithms.
    
    Features:
    - Lazy loading of algorithms (only load when requested)
    - Intelligent caching with Streamlit session state
    - Thread-safe algorithm switching
    - Performance monitoring and comparison
    - Graceful error handling and fallbacks
    """
    
    def __init__(self):
        """
        Initialize the Algorithm Manager.
        
        Phase 3: Delegated to specialized components for better separation of concerns.
        """
        # Initialize the three specialized components
        self.factory = AlgorithmFactory()
        self.lifecycle = LifecycleManager(self.factory)
        self.performance = PerformanceMonitor(self.factory)
    
    @staticmethod
    def get_instance() -> 'AlgorithmManager':
        """Get singleton instance of AlgorithmManager (Streamlit-compatible)"""
        if 'algorithm_manager' not in st.session_state:
            st.session_state.algorithm_manager = AlgorithmManager()
        return st.session_state.algorithm_manager
    
    def initialize_data(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        """
        Initialize with training data.
        
        Phase 3: Delegated to LifecycleManager.
        """
        self.lifecycle.initialize_data(ratings_df, movies_df)
        print("🎯 Algorithm Manager initialized with data")
    
    def get_algorithm(self, algorithm_type: AlgorithmType, 
                     custom_params: Optional[Dict[str, Any]] = None,
                     suppress_ui: bool = False) -> BaseRecommender:
        """
        Get a trained algorithm instance with lazy loading.
        
        Phase 3A: Fully delegated to LifecycleManager.
        
        Args:
            algorithm_type: Type of algorithm to get
            custom_params: Optional custom parameters (overrides defaults)
            suppress_ui: If True, suppress st.spinner and st.success UI elements
            
        Returns:
            Trained BaseRecommender instance
        """
        return self.lifecycle.get_algorithm(algorithm_type, custom_params, suppress_ui)
    
    def get_current_algorithm(self) -> Optional[BaseRecommender]:
        """
        Get the currently active algorithm.
        
        Phase 3C: Delegated to LifecycleManager.
        """
        return self.lifecycle.get_current_algorithm()
    
    def switch_algorithm(self, algorithm_type: AlgorithmType, 
                        custom_params: Optional[Dict[str, Any]] = None,
                        suppress_ui: bool = False) -> BaseRecommender:
        """
        Switch to a different algorithm with smooth transition.
        
        Phase 3C: Delegated to LifecycleManager.
        
        Args:
            algorithm_type: Algorithm to switch to
            custom_params: Optional custom parameters
            suppress_ui: If True, suppress UI updates (for nested contexts)
            
        Returns:
            The new algorithm instance
        """
        return self.lifecycle.switch_algorithm(algorithm_type, custom_params, suppress_ui)
    
    def get_available_algorithms(self) -> List[AlgorithmType]:
        """
        Get list of all available algorithm types.
        
        Phase 3: Delegated to AlgorithmFactory.
        """
        return self.factory.get_available_algorithms()
    
    def get_algorithm_info(self, algorithm_type: AlgorithmType) -> Dict[str, Any]:
        """
        Get detailed information about an algorithm without loading it.
        
        Phase 3D: Delegated to AlgorithmFactory.
        
        Returns:
            Dictionary with algorithm description, capabilities, etc.
        """
        return self.factory.get_algorithm_info(algorithm_type)
    
    def get_performance_comparison(self) -> pd.DataFrame:
        """
        Get performance comparison of all trained algorithms.
        
        Phase 3E: Delegated to PerformanceMonitor.
        
        Returns:
            DataFrame with performance metrics
        """
        return self.performance.get_performance_comparison(self.lifecycle._algorithms)
    
    def get_cached_algorithms(self) -> List[AlgorithmType]:
        """
        Get list of currently cached (trained) algorithms.
        
        Phase 3: Delegated to LifecycleManager.
        """
        return self.lifecycle.get_cached_algorithms()
    
    def clear_cache(self, algorithm_type: Optional[AlgorithmType] = None) -> None:
        """
        Clear algorithm cache (useful for memory management).
        
        Phase 3C: Delegated to LifecycleManager.
        
        Args:
            algorithm_type: Specific algorithm to clear, or None to clear all
        """
        self.lifecycle.clear_cache(algorithm_type)
    
    def preload_algorithm(self, algorithm_type: AlgorithmType, 
                         custom_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Preload an algorithm in the background for faster switching.
        
        Phase 3C: Delegated to LifecycleManager.
        
        Args:
            algorithm_type: Algorithm to preload
            custom_params: Optional custom parameters
        """
        self.lifecycle.preload_algorithm(algorithm_type, custom_params)
    
    def get_recommendation_explanation(self, algorithm_type: AlgorithmType,
                                    user_id: int, movie_id: int) -> str:
        """
        Get human-readable explanation for why a movie was recommended.
        
        Phase 3D: Delegated to AlgorithmFactory (with algorithm context lookup).
        
        Args:
            algorithm_type: Algorithm that made the recommendation
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Human-readable explanation string
        """
        # Get the algorithm and its context
        algorithm = self.lifecycle.get_current_algorithm()
        if not algorithm:
            return "Algorithm not loaded."
        
        context = algorithm.get_explanation_context(user_id, movie_id)
        
        # Delegate explanation generation to factory
        return self.factory.get_recommendation_explanation(algorithm_type, context)

    def get_algorithm_metrics(self, algorithm_type: AlgorithmType) -> Dict[str, Any]:
        """
        Get performance metrics for a specific algorithm with caching.
        
        Phase 3E: Delegated to PerformanceMonitor.
        
        Args:
            algorithm_type: Type of algorithm to get metrics for
            
        Returns:
            Dictionary with performance metrics
        """
        # Get or train the algorithm
        algorithm = self.get_algorithm(algorithm_type)
        
        # Delegate to PerformanceMonitor
        return self.performance.get_algorithm_metrics(
            algorithm,
            algorithm_type,
            training_data=self.lifecycle._training_data,
            use_cache=True
        )

    def get_all_algorithm_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics for all available algorithms.
        
        Phase 3E: Simplified iteration over available algorithms.
        
        Returns:
            Dictionary mapping algorithm names to their metrics
        """
        all_metrics = {}
        
        for algorithm_type in self.get_available_algorithms():
            try:
                metrics = self.get_algorithm_metrics(algorithm_type)
                all_metrics[algorithm_type.value] = metrics
            except Exception as e:
                all_metrics[algorithm_type.value] = {
                    "algorithm": algorithm_type.value,
                    "status": "Error",
                    "error": str(e),
                    "metrics": {}
                }
        
        return all_metrics


# Global singleton instance
algorithm_manager = None

def get_algorithm_manager() -> AlgorithmManager:
    """Get the global algorithm manager instance"""
    global algorithm_manager
    if algorithm_manager is None:
        algorithm_manager = AlgorithmManager.get_instance()
    return algorithm_manager