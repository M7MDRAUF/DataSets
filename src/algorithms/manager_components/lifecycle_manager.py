"""
Lifecycle Manager

Responsible for algorithm lifecycle management: loading, caching, switching, and cleanup.
Handles the "when" and "where" of algorithm state management.

Author: CineMatch Development Team
Date: November 12, 2025
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Any, List
import threading
import time
import pandas as pd
import streamlit as st

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.algorithms.base_recommender import BaseRecommender
from .algorithm_factory import AlgorithmType, AlgorithmFactory


class LifecycleManager:
    """
    Manager for algorithm lifecycle operations.
    
    Responsibilities:
    - Loading pre-trained models from disk
    - Training algorithms when needed
    - Caching trained algorithms
    - Switching between algorithms
    - Cache management and cleanup
    """
    
    def __init__(self, factory: AlgorithmFactory):
        """
        Initialize lifecycle manager.
        
        Args:
            factory: AlgorithmFactory instance for creating algorithms
        """
        self.factory = factory
        self._algorithms: Dict[AlgorithmType, BaseRecommender] = {}
        self._current_algorithm: Optional[AlgorithmType] = None
        self._lock = threading.Lock()
        self._training_data: Optional[tuple] = None
        self._is_initialized = False
    
    def initialize_data(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        """
        Initialize with training data.
        
        Args:
            ratings_df: User ratings DataFrame
            movies_df: Movies metadata DataFrame
        """
        self._training_data = (ratings_df, movies_df)
        self._is_initialized = True
    
    def get_algorithm(self,
                     algorithm_type: AlgorithmType,
                     custom_params: Optional[Dict[str, Any]] = None,
                     suppress_ui: bool = False) -> BaseRecommender:
        """
        Get a trained algorithm instance with lazy loading and caching.
        
        Args:
            algorithm_type: Type of algorithm to get
            custom_params: Optional custom parameters (overrides defaults)
            suppress_ui: If True, suppress st.spinner and st.success UI elements
            
        Returns:
            Trained BaseRecommender instance
        """
        if not self._is_initialized:
            raise ValueError("LifecycleManager not initialized. Call initialize_data() first.")
        
        with self._lock:
            # Check if algorithm is already cached and trained
            if algorithm_type in self._algorithms:
                algorithm = self._algorithms[algorithm_type]
                if algorithm.is_trained:
                    print(f"✓ Using cached {algorithm.name}")
                    return algorithm
            
            # Need to create and train algorithm
            print(f"🔄 Loading {algorithm_type.value}...")
            
            # Create algorithm instance using factory
            algorithm = self.factory.create_algorithm(algorithm_type, custom_params)
            
            # Try to load pre-trained model first
            if self._try_load_pretrained_model(algorithm, algorithm_type):
                # Pre-trained model loaded successfully
                self._algorithms[algorithm_type] = algorithm
                return algorithm
            
            # Train algorithm if no pre-trained model available
            ratings_df, movies_df = self._training_data
            
            # Show progress in Streamlit (only if not suppressed)
            if suppress_ui:
                # Train without UI updates (used when called from Hybrid algorithm)
                start_time = time.time()
                algorithm.fit(ratings_df, movies_df)
                training_time = time.time() - start_time
                self._algorithms[algorithm_type] = algorithm
                print(f"✓ {algorithm.name} trained in {training_time:.1f}s")
            else:
                # Normal training with UI feedback
                with st.spinner(f'Training {algorithm.name}... This may take a moment.'):
                    start_time = time.time()
                    algorithm.fit(ratings_df, movies_df)
                    training_time = time.time() - start_time
                    
                    # Cache the trained algorithm
                    self._algorithms[algorithm_type] = algorithm
                    
                    print(f"✓ {algorithm.name} trained in {training_time:.1f}s")
                    
                    # Show success message
                    st.success(f"✅ {algorithm.name} ready! (Trained in {training_time:.1f}s)")
                
            return algorithm
    
    def _try_load_pretrained_model(self, algorithm: BaseRecommender, algorithm_type: AlgorithmType) -> bool:
        """
        Try to load a pre-trained model for KNN, Content-Based, and Hybrid algorithms.
        
        Args:
            algorithm: The algorithm instance to load into
            algorithm_type: The type of algorithm
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        # Only try to load for KNN, Content-Based, and Hybrid algorithms
        if algorithm_type not in [AlgorithmType.USER_KNN, AlgorithmType.ITEM_KNN, 
                                   AlgorithmType.CONTENT_BASED, AlgorithmType.HYBRID]:
            return False
        
        # Define model paths
        model_paths = {
            AlgorithmType.USER_KNN: Path("models/user_knn_model.pkl"),
            AlgorithmType.ITEM_KNN: Path("models/item_knn_model.pkl"),
            AlgorithmType.CONTENT_BASED: Path("models/content_based_model.pkl"),
            AlgorithmType.HYBRID: Path("models/hybrid_model.pkl")
        }
        
        model_path = model_paths.get(algorithm_type)
        if not model_path:
            return False
        
        if not model_path.exists():
            print(f"   • No pre-trained model found at {model_path}")
            return False
        
        try:
            print(f"   • Loading pre-trained model from {model_path}")
            start_time = time.time()
            algorithm.load_model(model_path)
            load_time = time.time() - start_time
            
            # Provide data context to the loaded model
            ratings_df, movies_df = self._training_data
            algorithm.ratings_df = ratings_df.copy()
            algorithm.movies_df = movies_df.copy()
            # Add genres_list if not present
            if 'genres_list' not in algorithm.movies_df.columns:
                algorithm.movies_df['genres_list'] = algorithm.movies_df['genres'].str.split('|')
            print(f"   • Data context provided to loaded model")
            
            # For Hybrid algorithm, also provide data context to sub-algorithms
            if algorithm_type == AlgorithmType.HYBRID:
                print(f"   • Providing data context to Hybrid sub-algorithms...")
                algorithm.svd_model.ratings_df = ratings_df.copy()
                algorithm.svd_model.movies_df = movies_df.copy()
                algorithm.user_knn_model.ratings_df = ratings_df.copy()
                algorithm.user_knn_model.movies_df = movies_df.copy()
                algorithm.item_knn_model.ratings_df = ratings_df.copy()
                algorithm.item_knn_model.movies_df = movies_df.copy()
                algorithm.content_based_model.ratings_df = ratings_df.copy()
                algorithm.content_based_model.movies_df = movies_df.copy()
                # Add genres_list to sub-algorithms
                for sub_model in [algorithm.svd_model, algorithm.user_knn_model, 
                                 algorithm.item_knn_model, algorithm.content_based_model]:
                    if 'genres_list' not in sub_model.movies_df.columns:
                        sub_model.movies_df['genres_list'] = sub_model.movies_df['genres'].str.split('|')
                print(f"   ✓ Data context provided to all 4 sub-algorithms")
            
            # Verify the model is properly loaded and trained
            if algorithm.is_trained:
                print(f"   ✓ Pre-trained {algorithm.name} loaded in {load_time:.2f}s")
                st.success(f"🚀 {algorithm.name} loaded from pre-trained model! ({load_time:.2f}s)")
                return True
            else:
                print(f"   ⚠ Pre-trained model loaded but not marked as trained")
                return False
                
        except Exception as e:
            print(f"   ❌ Failed to load pre-trained model: {e}")
            print(f"   → Will train from scratch instead")
            return False
    
    def get_current_algorithm(self) -> Optional[BaseRecommender]:
        """Get the currently active algorithm, if any."""
        if self._current_algorithm and self._current_algorithm in self._algorithms:
            return self._algorithms[self._current_algorithm]
        return None
    
    def switch_algorithm(self,
                        algorithm_type: AlgorithmType,
                        custom_params: Optional[Dict[str, Any]] = None,
                        suppress_ui: bool = False) -> BaseRecommender:
        """
        Switch to a different algorithm.
        
        Args:
            algorithm_type: Type of algorithm to switch to
            custom_params: Optional custom parameters
            suppress_ui: If True, suppress UI updates
            
        Returns:
            The newly activated algorithm instance
        """
        print(f"🔄 Switching to {algorithm_type.value}")
        algorithm = self.get_algorithm(algorithm_type, custom_params, suppress_ui)
        self._current_algorithm = algorithm_type
        return algorithm
    
    def get_cached_algorithms(self) -> List[AlgorithmType]:
        """Get list of algorithms currently in cache."""
        return [algo_type for algo_type, algo in self._algorithms.items() if algo.is_trained]
    
    def clear_cache(self, algorithm_type: Optional[AlgorithmType] = None) -> None:
        """
        Clear algorithm cache.
        
        Args:
            algorithm_type: If provided, clear only this algorithm. Otherwise clear all.
        """
        with self._lock:
            if algorithm_type:
                if algorithm_type in self._algorithms:
                    del self._algorithms[algorithm_type]
                    print(f"✓ Cleared {algorithm_type.value} from cache")
            else:
                self._algorithms.clear()
                print("✓ Cleared all algorithms from cache")
    
    def preload_algorithm(self,
                         algorithm_type: AlgorithmType,
                         custom_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Preload an algorithm in the background without switching to it.
        
        Args:
            algorithm_type: Type of algorithm to preload
            custom_params: Optional custom parameters
        """
        print(f"⏳ Preloading {algorithm_type.value} in background...")
        self.get_algorithm(algorithm_type, custom_params, suppress_ui=True)
