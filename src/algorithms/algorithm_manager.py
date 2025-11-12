"""
CineMatch V1.0.0 - Algorithm Manager

Central management system for all recommendation algorithms.
Handles instantiation, switching, lifecycle management, and intelligent caching.

Author: CineMatch Development Team
Date: November 7, 2025
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Any, Type, List
import streamlit as st
import pandas as pd
import time
import threading
from enum import Enum

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.algorithms.base_recommender import BaseRecommender
from src.algorithms.svd_recommender import SVDRecommender
from src.algorithms.user_knn_recommender import UserKNNRecommender
from src.algorithms.item_knn_recommender import ItemKNNRecommender
from src.algorithms.content_based_recommender import ContentBasedRecommender
from src.algorithms.hybrid_recommender import HybridRecommender


class AlgorithmType(Enum):
    """Enumeration of available recommendation algorithms"""
    SVD = "SVD Matrix Factorization"
    USER_KNN = "KNN User-Based"
    ITEM_KNN = "KNN Item-Based"
    CONTENT_BASED = "Content-Based Filtering"
    HYBRID = "Hybrid (Best of All)"


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
        """Initialize the Algorithm Manager"""
        self._algorithms: Dict[AlgorithmType, BaseRecommender] = {}
        self._algorithm_classes: Dict[AlgorithmType, Type[BaseRecommender]] = {
            AlgorithmType.SVD: SVDRecommender,
            AlgorithmType.USER_KNN: UserKNNRecommender,
            AlgorithmType.ITEM_KNN: ItemKNNRecommender,
            AlgorithmType.CONTENT_BASED: ContentBasedRecommender,
            AlgorithmType.HYBRID: HybridRecommender
        }
        self._default_params: Dict[AlgorithmType, Dict[str, Any]] = {
            AlgorithmType.SVD: {'n_components': 100},
            AlgorithmType.USER_KNN: {'n_neighbors': 50, 'similarity_metric': 'cosine'},
            AlgorithmType.ITEM_KNN: {'n_neighbors': 30, 'similarity_metric': 'cosine', 'min_ratings': 5},
            AlgorithmType.CONTENT_BASED: {
                'genre_weight': 0.5, 
                'tag_weight': 0.3, 
                'title_weight': 0.2,
                'min_similarity': 0.01
            },
            AlgorithmType.HYBRID: {
                'svd_params': {'n_components': 100},
                'user_knn_params': {'n_neighbors': 50, 'similarity_metric': 'cosine'},
                'item_knn_params': {'n_neighbors': 30, 'similarity_metric': 'cosine', 'min_ratings': 5},
                'content_based_params': {
                    'genre_weight': 0.5, 
                    'tag_weight': 0.3, 
                    'title_weight': 0.2,
                    'min_similarity': 0.01
                },
                'weighting_strategy': 'adaptive'
            }
        }
        self._lock = threading.Lock()
        self._training_data: Optional[tuple] = None
        self._is_initialized = False
    
    @staticmethod
    def get_instance() -> 'AlgorithmManager':
        """Get singleton instance of AlgorithmManager (Streamlit-compatible)"""
        if 'algorithm_manager' not in st.session_state:
            st.session_state.algorithm_manager = AlgorithmManager()
        return st.session_state.algorithm_manager
    
    def initialize_data(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        """Initialize with training data (call once when app starts)"""
        self._training_data = (ratings_df.copy(), movies_df.copy())
        self._is_initialized = True
        print("🎯 Algorithm Manager initialized with data")
    
    def get_algorithm(self, algorithm_type: AlgorithmType, 
                     custom_params: Optional[Dict[str, Any]] = None) -> BaseRecommender:
        """
        Get a trained algorithm instance with lazy loading.
        
        Args:
            algorithm_type: Type of algorithm to get
            custom_params: Optional custom parameters (overrides defaults)
            
        Returns:
            Trained BaseRecommender instance
        """
        if not self._is_initialized:
            raise ValueError("AlgorithmManager not initialized. Call initialize_data() first.")
        
        with self._lock:
            # Check if algorithm is already cached and trained
            if algorithm_type in self._algorithms:
                algorithm = self._algorithms[algorithm_type]
                if algorithm.is_trained:
                    print(f"✓ Using cached {algorithm.name}")
                    return algorithm
            
            # Need to create and train algorithm
            print(f"🔄 Loading {algorithm_type.value}...")
            
            # Merge default and custom parameters
            params = self._default_params[algorithm_type].copy()
            if custom_params:
                params.update(custom_params)
            
            # Instantiate algorithm
            algorithm_class = self._algorithm_classes[algorithm_type]
            algorithm = algorithm_class(**params)
            
            # Try to load pre-trained model first (for KNN models)
            if self._try_load_pretrained_model(algorithm, algorithm_type):
                # Pre-trained model loaded successfully
                self._algorithms[algorithm_type] = algorithm
                return algorithm
            
            # Train algorithm if no pre-trained model available
            ratings_df, movies_df = self._training_data
            
            # Show progress in Streamlit
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
        Try to load a pre-trained model for KNN algorithms.
        
        Args:
            algorithm: The algorithm instance to load into
            algorithm_type: The type of algorithm
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        # Only try to load for KNN and Content-Based algorithms
        if algorithm_type not in [AlgorithmType.USER_KNN, AlgorithmType.ITEM_KNN, AlgorithmType.CONTENT_BASED]:
            return False
        
        # Define model paths
        model_paths = {
            AlgorithmType.USER_KNN: Path("models/user_knn_model.pkl"),
            AlgorithmType.ITEM_KNN: Path("models/item_knn_model.pkl"),
            AlgorithmType.CONTENT_BASED: Path("models/content_based_model.pkl")
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
            
            # IMPORTANT: Provide data context to the loaded model
            # Pre-trained models need access to current data for some operations
            ratings_df, movies_df = self._training_data
            algorithm.ratings_df = ratings_df.copy()
            algorithm.movies_df = movies_df.copy()
            # Add genres_list if not present
            if 'genres_list' not in algorithm.movies_df.columns:
                algorithm.movies_df['genres_list'] = algorithm.movies_df['genres'].str.split('|')
            print(f"   • Data context provided to loaded model")
            
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
        """Get the currently selected algorithm from Streamlit session state"""
        if 'selected_algorithm' not in st.session_state:
            return None
        
        algorithm_type = st.session_state.selected_algorithm
        if algorithm_type in self._algorithms:
            return self._algorithms[algorithm_type]
        
        return None
    
    def switch_algorithm(self, algorithm_type: AlgorithmType, 
                        custom_params: Optional[Dict[str, Any]] = None) -> BaseRecommender:
        """
        Switch to a different algorithm with smooth transition.
        
        Args:
            algorithm_type: Algorithm to switch to
            custom_params: Optional custom parameters
            
        Returns:
            The new algorithm instance
        """
        print(f"🔄 Switching to {algorithm_type.value}")
        
        # Store selection in session state
        st.session_state.selected_algorithm = algorithm_type
        
        # Get the algorithm (will load if not cached)
        algorithm = self.get_algorithm(algorithm_type, custom_params)
        
        # Update UI state
        if 'algorithm_switched' not in st.session_state:
            st.session_state.algorithm_switched = True
        
        return algorithm
    
    def get_available_algorithms(self) -> List[AlgorithmType]:
        """Get list of all available algorithm types"""
        return list(AlgorithmType)
    
    def get_algorithm_info(self, algorithm_type: AlgorithmType) -> Dict[str, Any]:
        """
        Get detailed information about an algorithm without loading it.
        
        Returns:
            Dictionary with algorithm description, capabilities, etc.
        """
        info_map = {
            AlgorithmType.SVD: {
                'name': 'SVD Matrix Factorization',
                'description': 'Uses Singular Value Decomposition to discover hidden patterns in user ratings. Excellent for finding complex relationships between users and movies.',
                'strengths': ['High accuracy', 'Handles sparse data well', 'Discovers latent factors', 'Good for diverse recommendations'],
                'ideal_for': ['Users with varied taste', 'Discovering hidden gems', 'Academic research', 'High-accuracy needs'],
                'complexity': 'High',
                'speed': 'Medium',
                'interpretability': 'Medium',
                'icon': '🔮'
            },
            AlgorithmType.USER_KNN: {
                'name': 'KNN User-Based',
                'description': 'Finds users with similar taste and recommends movies those users loved. Simple and intuitive approach.',
                'strengths': ['Highly interpretable', 'Good for sparse users', 'Handles new items well', 'Community-based recommendations'],
                'ideal_for': ['New users', 'Sparse rating profiles', 'Social recommendations', 'Explainable results'],
                'complexity': 'Low',
                'speed': 'Fast',
                'interpretability': 'Very High',
                'icon': '👥'
            },
            AlgorithmType.ITEM_KNN: {
                'name': 'KNN Item-Based',
                'description': 'Analyzes movies with similar rating patterns and recommends items similar to what you enjoyed.',
                'strengths': ['Stable recommendations', 'Good for frequent users', 'Pre-computed similarities', 'Genre-aware'],
                'ideal_for': ['Users with many ratings', 'Discovering similar movies', 'Stable preferences', 'Genre exploration'],
                'complexity': 'Medium',
                'speed': 'Fast',
                'interpretability': 'High',
                'icon': '🎬'
            },
            AlgorithmType.CONTENT_BASED: {
                'name': 'Content-Based Filtering',
                'description': 'Analyzes movie features (genres, tags, titles) and recommends movies similar to what you enjoyed. Perfect for cold-start scenarios.',
                'strengths': ['No cold-start problem', 'Feature-based recommendations', 'Highly interpretable', 'Tag and genre aware', 'Works for new users'],
                'ideal_for': ['New users', 'Genre-specific discovery', 'Feature-based exploration', 'Cold-start scenarios', 'Explainable recommendations'],
                'complexity': 'Medium',
                'speed': 'Fast',
                'interpretability': 'Very High',
                'icon': '🔍'
            },
            AlgorithmType.HYBRID: {
                'name': 'Hybrid (Best of All)',
                'description': 'Intelligently combines all algorithms with dynamic weighting based on your profile and context.',
                'strengths': ['Best overall accuracy', 'Adapts to user type', 'Robust performance', 'Combines multiple paradigms'],
                'ideal_for': ['Production systems', 'Best accuracy', 'All user types', 'Research comparison'],
                'complexity': 'Very High',
                'speed': 'Medium',
                'interpretability': 'Medium',
                'icon': '🚀'
            }
        }
        
        return info_map.get(algorithm_type, {})
    
    def get_performance_comparison(self) -> pd.DataFrame:
        """
        Get performance comparison of all trained algorithms.
        
        Returns:
            DataFrame with performance metrics
        """
        performance_data = []
        
        for algorithm_type, algorithm in self._algorithms.items():
            if algorithm.is_trained:
                info = self.get_algorithm_info(algorithm_type)
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
    
    def get_cached_algorithms(self) -> List[AlgorithmType]:
        """Get list of currently cached (trained) algorithms"""
        return [algo_type for algo_type, algo in self._algorithms.items() if algo.is_trained]
    
    def clear_cache(self, algorithm_type: Optional[AlgorithmType] = None) -> None:
        """
        Clear algorithm cache (useful for memory management).
        
        Args:
            algorithm_type: Specific algorithm to clear, or None to clear all
        """
        with self._lock:
            if algorithm_type is None:
                # Clear all algorithms
                self._algorithms.clear()
                print("🗑️ Cleared all algorithm cache")
            elif algorithm_type in self._algorithms:
                # Clear specific algorithm
                del self._algorithms[algorithm_type]
                print(f"🗑️ Cleared {algorithm_type.value} from cache")
    
    def preload_algorithm(self, algorithm_type: AlgorithmType, 
                         custom_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Preload an algorithm in the background for faster switching.
        
        Args:
            algorithm_type: Algorithm to preload
            custom_params: Optional custom parameters
        """
        if algorithm_type not in self._algorithms:
            print(f"🔄 Preloading {algorithm_type.value} in background...")
            # This will cache the algorithm for future use
            self.get_algorithm(algorithm_type, custom_params)
    
    def get_recommendation_explanation(self, algorithm_type: AlgorithmType,
                                    user_id: int, movie_id: int) -> str:
        """
        Get human-readable explanation for why a movie was recommended.
        
        Args:
            algorithm_type: Algorithm that made the recommendation
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Human-readable explanation string
        """
        if algorithm_type not in self._algorithms:
            return "Algorithm not loaded."
        
        algorithm = self._algorithms[algorithm_type]
        context = algorithm.get_explanation_context(user_id, movie_id)
        
        if not context:
            return "Unable to generate explanation."
        
        # Generate explanation based on algorithm type and context
        if algorithm_type == AlgorithmType.SVD:
            return self._explain_svd(context)
        elif algorithm_type == AlgorithmType.USER_KNN:
            return self._explain_user_knn(context)
        elif algorithm_type == AlgorithmType.ITEM_KNN:
            return self._explain_item_knn(context)
        elif algorithm_type == AlgorithmType.HYBRID:
            return self._explain_hybrid(context)
        
        return "Explanation not available."
    
    def _explain_svd(self, context: Dict[str, Any]) -> str:
        """Generate SVD-specific explanation"""
        pred = context.get('prediction', 0)
        return (f"SVD predicts you'll rate this movie **{pred:.1f}/5.0** based on "
               f"latent patterns discovered in your rating history and similar users' preferences.")
    
    def _explain_user_knn(self, context: Dict[str, Any]) -> str:
        """Generate User KNN-specific explanation"""
        similar_users = context.get('similar_users_count', 0)
        pred = context.get('prediction', 0)
        
        if similar_users > 0:
            return (f"**{similar_users} users** with similar taste loved this movie! "
                   f"Predicted rating: **{pred:.1f}/5.0**")
        else:
            return f"Based on users with similar preferences. Predicted rating: **{pred:.1f}/5.0**"
    
    def _explain_item_knn(self, context: Dict[str, Any]) -> str:
        """Generate Item KNN-specific explanation"""
        similar_movies = context.get('similar_movies_count', 0)
        pred = context.get('prediction', 0)
        
        if similar_movies > 0:
            return (f"Because you enjoyed **{similar_movies} similar movies**. "
                   f"Predicted rating: **{pred:.1f}/5.0**")
        else:
            return f"Based on movies with similar rating patterns. Predicted rating: **{pred:.1f}/5.0**"
    
    def _explain_hybrid(self, context: Dict[str, Any]) -> str:
        """Generate Hybrid-specific explanation"""
        primary = context.get('primary_algorithm', 'multiple algorithms')
        pred = context.get('prediction', 0)
        weights = context.get('algorithm_weights', {})
        
        return (f"Hybrid prediction (**{pred:.1f}/5.0**) combining multiple algorithms. "
               f"Primary method: **{primary}**. Algorithm weights: "
               f"SVD {weights.get('svd', 0):.2f}, User KNN {weights.get('user_knn', 0):.2f}, "
               f"Item KNN {weights.get('item_knn', 0):.2f}")

    def get_algorithm_metrics(self, algorithm_type: AlgorithmType) -> Dict[str, Any]:
        """Get performance metrics for a specific algorithm"""
        # Get or train the algorithm
        algorithm = self.get_algorithm(algorithm_type)
        
        if not algorithm.is_trained:
            return {
                "algorithm": algorithm_type.value,
                "status": "Not trained",
                "metrics": {}
            }
        
        # Get basic algorithm info
        info = self.get_algorithm_info(algorithm_type)
        
        # Create a sample of test data for metrics calculation
        if self._training_data is not None and len(self._training_data[0]) > 1000:
            ratings_df = self._training_data[0]
            # Use a small sample for quick metrics calculation
            sample_size = min(1000, len(ratings_df) // 10)
            test_sample = ratings_df.sample(n=sample_size, random_state=42)
            
            predictions = []
            actuals = []
            
            print(f"🔍 Debug: Testing {len(test_sample)} samples for {algorithm_type.value}")
            
            for i, (_, row) in enumerate(test_sample.iterrows()):
                try:
                    pred = algorithm.predict(row['userId'], row['movieId'])
                    if pred is not None and pred > 0:
                        predictions.append(pred)
                        actuals.append(row['rating'])
                except Exception as e:
                    # Skip failed predictions
                    if i < 5:  # Only print first 5 errors to avoid spam
                        print(f"    ❌ Prediction failed for user {row['userId']}, movie {row['movieId']}: {e}")
                    continue
            
            print(f"    ✓ Got {len(predictions)} valid predictions out of {len(test_sample)} samples")
            
            # Calculate metrics if we have predictions
            metrics = {}
            if predictions and len(predictions) > 1:  # Reduced threshold from 10 to 1
                import numpy as np
                from sklearn.metrics import mean_squared_error, mean_absolute_error
                
                rmse = np.sqrt(mean_squared_error(actuals, predictions))
                mae = mean_absolute_error(actuals, predictions)
                
                metrics = {
                    "rmse": round(rmse, 3),
                    "mae": round(mae, 3),
                    "sample_size": len(predictions),
                    "coverage": round(len(predictions) / len(test_sample) * 100, 1)
                }
            else:
                print(f"    ❌ Insufficient predictions: got {len(predictions)}, need at least 2")
                metrics = {
                    "error": f"Insufficient valid predictions for metrics calculation (got {len(predictions)})",
                    "predictions_count": len(predictions),
                    "sample_size": len(test_sample)
                }
        else:
            metrics = {
                "error": "Insufficient data for metrics calculation"
            }
        
        return {
            "algorithm": algorithm_type.value,
            "status": "Trained",
            "training_time": info.get("training_time", "Unknown"),
            "model_size": info.get("model_size", "Unknown"),
            "metrics": metrics,
            "last_updated": info.get("last_updated", "Unknown")
        }

    def get_all_algorithm_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all available algorithms"""
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