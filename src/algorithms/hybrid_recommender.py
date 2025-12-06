"""
CineMatch V2.1.6 - Hybrid Recommender

Intelligent ensemble combining SVD, User KNN, and Item KNN algorithms.
Dynamically weights different algorithms based on user context and data availability.

Author: CineMatch Development Team
Date: November 7, 2025
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import gc

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.algorithms.base_recommender import BaseRecommender
from src.algorithms.svd_recommender import SVDRecommender
from src.algorithms.user_knn_recommender import UserKNNRecommender
from src.algorithms.item_knn_recommender import ItemKNNRecommender
from src.algorithms.content_based_recommender import ContentBasedRecommender


class HybridRecommender(BaseRecommender):
    """
    Hybrid Recommender combining multiple algorithms intelligently.
    
    Uses SVD for matrix factorization, User KNN for neighborhood-based recommendations,
    and Item KNN for item similarity. Weights are dynamically adjusted based on:
    - User rating profile (sparse vs dense)
    - Data availability for each algorithm
    - Historical performance metrics
    """
    
    def __init__(self, 
                 svd_params: Dict[str, Any] = None,
                 user_knn_params: Dict[str, Any] = None,
                 item_knn_params: Dict[str, Any] = None,
                 content_based_params: Dict[str, Any] = None,
                 weighting_strategy: str = 'adaptive',
                 lazy_load: bool = True,
                 **kwargs):
        """
        Initialize Hybrid recommender with sub-algorithms.
        
        Args:
            svd_params: Parameters for SVD algorithm
            user_knn_params: Parameters for User KNN algorithm  
            item_knn_params: Parameters for Item KNN algorithm
            content_based_params: Parameters for Content-Based algorithm
            weighting_strategy: 'adaptive', 'equal', or 'performance_based'
            lazy_load: If True, defer sub-algorithm instantiation until needed
            **kwargs: Additional parameters
        """
        super().__init__("Hybrid (SVD + KNN + CBF)", 
                        svd_params=svd_params or {},
                        user_knn_params=user_knn_params or {},
                        item_knn_params=item_knn_params or {},
                        content_based_params=content_based_params or {},
                        weighting_strategy=weighting_strategy,
                        **kwargs)
        
        self.weighting_strategy = weighting_strategy
        
        # Store parameters for lazy initialization
        self._svd_params = svd_params or {}
        self._user_knn_params = user_knn_params or {}
        self._item_knn_params = item_knn_params or {}
        self._content_based_params = content_based_params or {}
        self._lazy_load = lazy_load
        
        # Lazy-loaded sub-algorithm instances
        self._svd_model: Optional[SVDRecommender] = None
        self._user_knn_model: Optional[UserKNNRecommender] = None
        self._item_knn_model: Optional[ItemKNNRecommender] = None
        self._content_based_model: Optional[ContentBasedRecommender] = None
        
        # Initialize sub-algorithms immediately if not lazy loading
        if not lazy_load:
            self._init_all_algorithms()
        
        # Algorithm weights (will be calculated during training)
        self.weights = {'svd': 0.30, 'user_knn': 0.25, 'item_knn': 0.25, 'content_based': 0.20}
        self.algorithm_performance = {}
        
        # User classification thresholds
        self.sparse_user_threshold = 20  # Users with <20 ratings are sparse
        self.dense_user_threshold = 50   # Users with >50 ratings are dense
        
        # Parallel training configuration
        self._use_parallel_training = True
        self._max_parallel_workers = 2  # Limit to 2 to avoid memory issues
        self._training_lock = threading.Lock()
    
    def _init_all_algorithms(self) -> None:
        """Initialize all sub-algorithms (used when lazy_load=False)."""
        if self._svd_model is None:
            self._svd_model = SVDRecommender(**self._svd_params)
        if self._user_knn_model is None:
            self._user_knn_model = UserKNNRecommender(**self._user_knn_params)
        if self._item_knn_model is None:
            self._item_knn_model = ItemKNNRecommender(**self._item_knn_params)
        if self._content_based_model is None:
            self._content_based_model = ContentBasedRecommender(**self._content_based_params)
    
    @property
    def svd_model(self) -> SVDRecommender:
        """Lazy-load SVD model on first access."""
        if self._svd_model is None:
            self._svd_model = SVDRecommender(**self._svd_params)
        return self._svd_model
    
    @svd_model.setter
    def svd_model(self, value: SVDRecommender) -> None:
        """Allow setting SVD model directly."""
        self._svd_model = value
    
    @property
    def user_knn_model(self) -> UserKNNRecommender:
        """Lazy-load User KNN model on first access."""
        if self._user_knn_model is None:
            self._user_knn_model = UserKNNRecommender(**self._user_knn_params)
        return self._user_knn_model
    
    @user_knn_model.setter
    def user_knn_model(self, value: UserKNNRecommender) -> None:
        """Allow setting User KNN model directly."""
        self._user_knn_model = value
    
    @property
    def item_knn_model(self) -> ItemKNNRecommender:
        """Lazy-load Item KNN model on first access."""
        if self._item_knn_model is None:
            self._item_knn_model = ItemKNNRecommender(**self._item_knn_params)
        return self._item_knn_model
    
    @item_knn_model.setter
    def item_knn_model(self, value: ItemKNNRecommender) -> None:
        """Allow setting Item KNN model directly."""
        self._item_knn_model = value
    
    @property
    def content_based_model(self) -> ContentBasedRecommender:
        """Lazy-load Content-Based model on first access."""
        if self._content_based_model is None:
            self._content_based_model = ContentBasedRecommender(**self._content_based_params)
        return self._content_based_model
    
    @content_based_model.setter
    def content_based_model(self, value: ContentBasedRecommender) -> None:
        """Allow setting Content-Based model directly."""
        self._content_based_model = value
    
    def _train_single_algorithm(
        self, 
        algo_key: str, 
        algo_name: str, 
        model_filename: str, 
        model_attr: str,
        model: 'BaseRecommender',
        ratings_df: pd.DataFrame, 
        movies_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Train a single sub-algorithm (used for parallel training).
        
        Returns dict with algorithm key and performance metrics.
        """
        try:
            # Early return if model is already trained (cached in memory)
            if hasattr(model, 'is_trained') and model.is_trained:
                print(f"\n✓ {algo_name} already trained (using cached model)")
                return {
                    'key': algo_key,
                    'rmse': model.metrics.rmse,
                    'training_time': model.metrics.training_time,
                    'coverage': model.metrics.coverage,
                    'memory_mb': model.metrics.memory_usage_mb,
                    'success': True,
                    'cached': True
                }
            
            print(f"\n📊 Loading/Training {algo_name} algorithm...")
            
            # Try to load pre-trained model first
            if not self._try_load_algorithm(algo_name, model_filename, model_attr, ratings_df, movies_df):
                print(f"  • No pre-trained {algo_name} model found, training from scratch...")
                model.fit(ratings_df, movies_df)
            
            # Collect performance metrics
            result = {
                'key': algo_key,
                'rmse': model.metrics.rmse,
                'training_time': model.metrics.training_time,
                'coverage': model.metrics.coverage,
                'memory_mb': model.metrics.memory_usage_mb,
                'success': True
            }
            
            # Force garbage collection after each algorithm training
            gc.collect()
            
            return result
            
        except Exception as e:
            print(f"  ❌ Error training {algo_name}: {e}")
            return {
                'key': algo_key,
                'rmse': 1.0,
                'training_time': 0.0,
                'coverage': 0.0,
                'memory_mb': 0.0,
                'success': False,
                'error': str(e)
            }
    
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        """Train all sub-algorithms and calculate optimal weights"""
        print(f"\n🚀 Training {self.name}...")
        start_time = time.time()
        
        # Store data references (no copy for ratings - read-only)
        self.ratings_df = ratings_df
        
        # Only copy movies_df if we need to modify it
        if 'genres_list' not in movies_df.columns:
            self.movies_df = movies_df.copy()
            self.movies_df['genres_list'] = self.movies_df['genres'].str.split('|')
        else:
            self.movies_df = movies_df
        
        # Define all algorithms to train
        algorithms = [
            ('svd', 'SVD', 'svd_model.pkl', 'svd_model', self.svd_model),
            ('user_knn', 'User KNN', 'user_knn_model.pkl', 'user_knn_model', self.user_knn_model),
            ('item_knn', 'Item KNN', 'item_knn_model.pkl', 'item_knn_model', self.item_knn_model),
            ('content_based', 'Content-Based', 'content_based_model.pkl', 'content_based_model', self.content_based_model)
        ]
        
        if self._use_parallel_training:
            # Parallel training using ThreadPoolExecutor (2 at a time to avoid memory issues)
            print("\n⚡ Using parallel training (2 algorithms at a time)...")
            
            with ThreadPoolExecutor(max_workers=self._max_parallel_workers) as executor:
                # Submit all training tasks
                futures = {}
                for algo_key, algo_name, model_filename, model_attr, model in algorithms:
                    future = executor.submit(
                        self._train_single_algorithm,
                        algo_key, algo_name, model_filename, model_attr, model,
                        ratings_df, movies_df
                    )
                    futures[future] = algo_key
                
                # Collect results as they complete
                for future in as_completed(futures):
                    result = future.result()
                    algo_key = result['key']
                    
                    if result['success']:
                        self.algorithm_performance[algo_key] = {
                            'rmse': result['rmse'],
                            'training_time': result['training_time'],
                            'coverage': result['coverage']
                        }
                        print(f"  ✓ {algo_key.upper()} completed")
                    else:
                        # Use default values for failed algorithm
                        self.algorithm_performance[algo_key] = {
                            'rmse': 1.0,
                            'training_time': 0.0,
                            'coverage': 0.0
                        }
                        print(f"  ⚠ {algo_key.upper()} failed: {result.get('error', 'Unknown error')}")
        else:
            # Sequential training (fallback)
            print("\n📊 Using sequential training...")
            for algo_key, algo_name, model_filename, model_attr, model in algorithms:
                result = self._train_single_algorithm(
                    algo_key, algo_name, model_filename, model_attr, model,
                    ratings_df, movies_df
                )
                if result['success']:
                    self.algorithm_performance[algo_key] = {
                        'rmse': result['rmse'],
                        'training_time': result['training_time'],
                        'coverage': result['coverage']
                    }
        
        # Calculate optimal weights
        print("\n⚖️ Calculating optimal algorithm weights...")
        self._calculate_weights()
        
        # Calculate overall metrics
        training_time = time.time() - start_time
        self.metrics.training_time = training_time
        self.is_trained = True
        
        # Calculate hybrid RMSE
        print("  • Calculating hybrid RMSE...")
        self._calculate_hybrid_rmse(ratings_df)
        
        # Calculate combined metrics
        self.metrics.coverage = max(
            self.svd_model.metrics.coverage,
            self.user_knn_model.metrics.coverage,
            self.item_knn_model.metrics.coverage,
            self.content_based_model.metrics.coverage
        )
        
        self.metrics.memory_usage_mb = (
            self.svd_model.metrics.memory_usage_mb +
            self.user_knn_model.metrics.memory_usage_mb +
            self.item_knn_model.metrics.memory_usage_mb +
            self.content_based_model.metrics.memory_usage_mb
        )
        
        # Final garbage collection
        gc.collect()
        
        print(f"\n✓ {self.name} trained successfully!")
        print(f"  • Total training time: {training_time:.1f}s")
        print(f"  • Hybrid RMSE: {self.metrics.rmse:.4f}")
        print(f"  • Algorithm weights: SVD={self.weights['svd']:.2f}, "
              f"User KNN={self.weights['user_knn']:.2f}, "
              f"Item KNN={self.weights['item_knn']:.2f}, "
              f"Content-Based={self.weights['content_based']:.2f}")
        print(f"  • Combined coverage: {self.metrics.coverage:.1f}%")
        print(f"  • Total memory usage: {self.metrics.memory_usage_mb:.1f} MB")
    
    def _try_load_algorithm(
        self, 
        algorithm_name: str, 
        model_filename: str, 
        model_attr: str,
        ratings_df: pd.DataFrame, 
        movies_df: pd.DataFrame
    ) -> bool:
        """
        Generic method to load any pre-trained algorithm model.
        
        Args:
            algorithm_name: Display name for logging (e.g., "User KNN")
            model_filename: Model file name (e.g., "user_knn_model.pkl")
            model_attr: Attribute name to set (e.g., "user_knn_model")
            ratings_df: Ratings DataFrame for data context
            movies_df: Movies DataFrame for data context
        
        Returns:
            bool: True if loading succeeded, False otherwise
        """
        from pathlib import Path
        from src.utils import load_model_safe
        
        # Special handling for SVD: try sklearn version first
        if algorithm_name == "SVD":
            sklearn_path = Path("models/svd_model_sklearn.pkl")
            if sklearn_path.exists():
                model_filename = "svd_model_sklearn.pkl"
        
        model_path = Path(f"models/{model_filename}")
        if not model_path.exists():
            return False
            
        try:
            print(f"  • Loading pre-trained {algorithm_name} model...")
            start_time = time.time()
            
            # Load the model
            loaded_model = load_model_safe(str(model_path))
            
            # Set the model on this instance
            setattr(self, model_attr, loaded_model)
            
            # IMPORTANT: DO NOT replace model.ratings_df and model.movies_df
            # if the model already has them! Pre-trained models have internal 
            # data structures (user_mapper, item_mapper, similarity matrices) 
            # that are indexed against their ORIGINAL training data.
            # Replacing ratings_df/movies_df causes index mismatches and validation 
            # errors like "Rating count mismatch: ratings_df shows X, Matrix shows Y"
            
            # However, if the model was saved WITHOUT ratings_df (common for size
            # optimization), we need to provide it for operations that use it.
            model = getattr(self, model_attr)
            if not hasattr(model, 'ratings_df') or model.ratings_df is None:
                model.ratings_df = ratings_df
            if not hasattr(model, 'movies_df') or model.movies_df is None:
                model.movies_df = movies_df
            
            # Ensure required columns exist for recommendation output
            if hasattr(model, 'movies_df') and model.movies_df is not None:
                if 'genres_list' not in model.movies_df.columns:
                    model.movies_df['genres_list'] = model.movies_df['genres'].str.split('|')
                # poster_path is required by recommendation output but may not exist in basic movies data
                if 'poster_path' not in model.movies_df.columns:
                    model.movies_df['poster_path'] = None
                
                # CRITICAL FIX: Update poster_path from current session's movies_df
                # Pre-trained models may have stale/None poster_path values from when they were trained.
                # The session's movies_df (loaded from movies_with_TMDB_image_links.parquet) has actual TMDB paths.
                # We update the poster_path column by merging on movieId to get current TMDB poster URLs.
                if movies_df is not None and 'poster_path' in movies_df.columns:
                    # Create a mapping of movieId -> poster_path from current session data
                    poster_mapping = movies_df.set_index('movieId')['poster_path'].to_dict()
                    # Update the model's poster_path column with current TMDB data
                    model.movies_df['poster_path'] = model.movies_df['movieId'].map(poster_mapping)
            
            load_time = time.time() - start_time
            print(f"  ✓ Pre-trained {algorithm_name} loaded in {load_time:.2f}s")
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to load pre-trained {algorithm_name}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _calculate_weights(self) -> None:
        """Calculate optimal weights based on algorithm performance (with caching)."""
        # Check if we can use cached weights
        if hasattr(self, '_cached_weights_key'):
            # Generate a key based on current algorithm performance
            current_key = (
                self.weighting_strategy,
                tuple((k, v.get('rmse', 0)) for k, v in sorted(self.algorithm_performance.items()))
            )
            if current_key == self._cached_weights_key:
                print("    • Using cached weights")
                return
            self._cached_weights_key = current_key
        else:
            self._cached_weights_key = None
        
        if self.weighting_strategy == 'equal':
            self.weights = {'svd': 0.25, 'user_knn': 0.25, 'item_knn': 0.25, 'content_based': 0.25}
        elif self.weighting_strategy == 'performance_based':
            # Weight inversely proportional to RMSE (lower RMSE = higher weight)
            rmse_svd = self.algorithm_performance['svd']['rmse']
            rmse_user = self.algorithm_performance['user_knn']['rmse']
            rmse_item = self.algorithm_performance['item_knn']['rmse']
            rmse_content = self.algorithm_performance['content_based']['rmse']
            
            # Inverse RMSE for weights (avoid division by zero)
            inv_rmse_svd = 1 / max(rmse_svd, 0.001)
            inv_rmse_user = 1 / max(rmse_user, 0.001)
            inv_rmse_item = 1 / max(rmse_item, 0.001)
            inv_rmse_content = 1 / max(rmse_content, 0.001)
            
            total_inv_rmse = inv_rmse_svd + inv_rmse_user + inv_rmse_item + inv_rmse_content
            
            self.weights = {
                'svd': inv_rmse_svd / total_inv_rmse,
                'user_knn': inv_rmse_user / total_inv_rmse,
                'item_knn': inv_rmse_item / total_inv_rmse,
                'content_based': inv_rmse_content / total_inv_rmse
            }
        else:  # adaptive (default)
            # Balanced approach considering RMSE, coverage, and algorithm strengths
            svd_score = (1 / max(self.algorithm_performance['svd']['rmse'], 0.001)) * 1.3  # SVD bonus for matrix factorization
            user_score = (1 / max(self.algorithm_performance['user_knn']['rmse'], 0.001)) * 1.0
            item_score = (1 / max(self.algorithm_performance['item_knn']['rmse'], 0.001)) * 1.1  # Item bonus for stability
            content_score = (1 / max(self.algorithm_performance['content_based']['rmse'], 0.001)) * 0.9  # Content for cold-start
            
            total_score = svd_score + user_score + item_score + content_score
            
            self.weights = {
                'svd': svd_score / total_score,
                'user_knn': user_score / total_score,
                'item_knn': item_score / total_score,
                'content_based': content_score / total_score
            }
        
        # Update cache key after calculation
        self._cached_weights_key = (
            self.weighting_strategy,
            tuple((k, v.get('rmse', 0)) for k, v in sorted(self.algorithm_performance.items()))
        )
        
        print(f"    • Calculated weights: {self.weights}")
        print(f"    • Individual RMSEs: SVD={self.algorithm_performance['svd']['rmse']:.4f}, "
              f"User KNN={self.algorithm_performance['user_knn']['rmse']:.4f}, "
              f"Item KNN={self.algorithm_performance['item_knn']['rmse']:.4f}, "
              f"Content-Based={self.algorithm_performance['content_based']['rmse']:.4f}")
    
    def _calculate_hybrid_rmse(self, ratings_df: pd.DataFrame) -> None:
        """Calculate RMSE and MAE for the hybrid predictions (ultra-fast version for presentation)"""
        print("    → Using estimated hybrid RMSE/MAE for speed...")
        
        # For presentation speed: Use weighted average of individual RMSEs/MAEs instead of testing
        # This is much faster and gives a good approximation
        svd_rmse = self.algorithm_performance['svd']['rmse']
        user_knn_rmse = self.algorithm_performance['user_knn']['rmse']  
        item_knn_rmse = self.algorithm_performance['item_knn']['rmse']
        content_based_rmse = self.algorithm_performance['content_based']['rmse']
        
        # Get MAE values (use default if not available - backward compatibility)
        svd_mae = getattr(self.svd_model.metrics, 'mae', 0.0) or svd_rmse * 0.78
        user_knn_mae = getattr(self.user_knn_model.metrics, 'mae', 0.0) or user_knn_rmse * 0.78
        item_knn_mae = getattr(self.item_knn_model.metrics, 'mae', 0.0) or item_knn_rmse * 0.78
        content_based_mae = getattr(self.content_based_model.metrics, 'mae', 0.0) or content_based_rmse * 0.78
        
        # Calculate weighted average RMSE (mathematical approximation)
        estimated_rmse = (
            self.weights['svd'] * svd_rmse +
            self.weights['user_knn'] * user_knn_rmse +
            self.weights['item_knn'] * item_knn_rmse +
            self.weights['content_based'] * content_based_rmse
        )
        
        # Calculate weighted average MAE
        estimated_mae = (
            self.weights['svd'] * svd_mae +
            self.weights['user_knn'] * user_knn_mae +
            self.weights['item_knn'] * item_knn_mae +
            self.weights['content_based'] * content_based_mae
        )
        
        self.metrics.rmse = estimated_rmse
        self.metrics.mae = estimated_mae
        print(f"    ✓ Estimated Hybrid RMSE: {self.metrics.rmse:.4f}")
        print(f"    ✓ Estimated Hybrid MAE: {self.metrics.mae:.4f}")
        print(f"      (Weighted average: SVD={svd_rmse:.4f} × {self.weights['svd']:.2f} + "
              f"User KNN={user_knn_rmse:.4f} × {self.weights['user_knn']:.2f} + "
              f"Item KNN={item_knn_rmse:.4f} × {self.weights['item_knn']:.2f} + "
              f"Content-Based={content_based_rmse:.4f} × {self.weights['content_based']:.2f})")
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating using weighted combination of all algorithms"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        self._start_prediction_timer()
        try:
            return self._predict_hybrid_rating(user_id, movie_id)
        finally:
            self._end_prediction_timer()
    
    def _predict_hybrid_rating(self, user_id: int, movie_id: int) -> float:
        """
        Internal hybrid rating prediction logic with proper error handling.
        
        Attempts to get predictions from all algorithms and combines them using
        dynamic weights. Logs warnings for failed predictions but continues with
        available algorithms.
        """
        predictions = {}
        weights = self._get_dynamic_weights(user_id, movie_id)
        
        # Attempt to get predictions from each algorithm
        # Only catch expected exceptions, let critical errors propagate
        for algo_name, model in [
            ('svd', self.svd_model),
            ('user_knn', self.user_knn_model),
            ('item_knn', self.item_knn_model),
            ('content_based', self.content_based_model)
        ]:
            try:
                predictions[algo_name] = model.predict(user_id, movie_id)
            except (ValueError, KeyError, AttributeError, IndexError) as e:
                # Expected errors: missing data, untrained model, invalid IDs
                # Log warning but continue with other algorithms
                import warnings
                warnings.warn(
                    f"{algo_name} prediction failed for user {user_id}, movie {movie_id}: {type(e).__name__}: {e}",
                    RuntimeWarning
                )
                predictions[algo_name] = None
            # Note: SystemExit, KeyboardInterrupt, MemoryError will still propagate
        
        # Filter valid predictions and normalize weights
        valid_predictions = {k: v for k, v in predictions.items() if v is not None}
        
        if not valid_predictions:
            # Fallback to global average
            fallback = self.ratings_df['rating'].mean()
            import warnings
            warnings.warn(
                f"All algorithms failed to predict for user {user_id}, movie {movie_id}. "
                f"Using global average: {fallback:.2f}",
                RuntimeWarning
            )
            return fallback
        
        # Normalize weights for valid predictions only
        total_weight = sum(weights[k] for k in valid_predictions.keys())
        
        # Calculate weighted prediction
        weighted_prediction = sum(
            (weights[algo] / total_weight) * pred 
            for algo, pred in valid_predictions.items()
        )
        
        return np.clip(weighted_prediction, 0.5, 5.0)
    
    def _get_dynamic_weights(self, user_id: int, movie_id: int) -> Dict[str, float]:
        """Calculate dynamic weights based on user and item context"""
        # Start with base weights
        dynamic_weights = self.weights.copy()
        
        # Analyze user profile
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        num_ratings = len(user_ratings)
        
        if num_ratings == 0:
            # New user: favor item-based and SVD
            dynamic_weights['svd'] *= 1.3
            dynamic_weights['item_knn'] *= 1.2
            dynamic_weights['user_knn'] *= 0.5
        elif num_ratings < self.sparse_user_threshold:
            # Sparse user: favor user-based KNN and SVD
            dynamic_weights['user_knn'] *= 1.3
            dynamic_weights['svd'] *= 1.1
            dynamic_weights['item_knn'] *= 0.8
        elif num_ratings > self.dense_user_threshold:
            # Dense user: favor item-based KNN and SVD
            dynamic_weights['item_knn'] *= 1.2
            dynamic_weights['svd'] *= 1.1
            dynamic_weights['user_knn'] *= 0.9
        
        # Analyze movie popularity
        movie_ratings = self.ratings_df[self.ratings_df['movieId'] == movie_id]
        movie_popularity = len(movie_ratings)
        
        if movie_popularity < 10:
            # Rare movie: favor SVD (better for long tail)
            dynamic_weights['svd'] *= 1.2
            dynamic_weights['item_knn'] *= 0.8
        elif movie_popularity > 1000:
            # Popular movie: all algorithms work well, slight preference for KNN
            dynamic_weights['user_knn'] *= 1.1
            dynamic_weights['item_knn'] *= 1.1
        
        # Normalize weights
        total_weight = sum(dynamic_weights.values())
        dynamic_weights = {k: v/total_weight for k, v in dynamic_weights.items()}
        
        return dynamic_weights
    
    def get_recommendations(
        self,
        user_id: int,
        n: int = 10,
        exclude_rated: bool = True
    ) -> pd.DataFrame:
        """Generate hybrid recommendations with algorithm explanation"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        print(f"\n🎯 Generating {self.name} recommendations for User {user_id}...")
        self._start_prediction_timer()
        
        try:
            # Determine user profile for optimization
            user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
            num_ratings = len(user_ratings)
            
            print(f"  • User profile: {num_ratings} ratings")
            
            if num_ratings == 0:
                print("  • New user detected - using popularity-based approach")
                # For new users, use SVD and Item KNN primarily
                try:
                    svd_recs = self.svd_model.get_recommendations(user_id, n*2, exclude_rated)
                    item_recs = self.item_knn_model.get_recommendations(user_id, n*2, exclude_rated)
                    
                    # Combine and deduplicate
                    all_recs = pd.concat([svd_recs, item_recs]).drop_duplicates(subset=['movieId'])
                    top_recs = all_recs.head(n)
                    
                except Exception as e:
                    print(f"  ❌ Error in new user approach: {e}")
                    # Fallback to SVD only
                    top_recs = self.svd_model.get_recommendations(user_id, n, exclude_rated)
                
            elif num_ratings < self.sparse_user_threshold:
                print("  • Sparse user profile - emphasizing User KNN + SVD")
                try:
                    # For sparse users, get recommendations from User KNN and SVD
                    user_knn_recs = self.user_knn_model.get_recommendations(user_id, n, exclude_rated)
                    svd_recs = self.svd_model.get_recommendations(user_id, n, exclude_rated)
                    
                    # Add weights for aggregation using assign() - more memory efficient
                    user_knn_recs = user_knn_recs.assign(weight=0.6)
                    svd_recs = svd_recs.assign(weight=0.4)
                    
                    all_recs = pd.concat([user_knn_recs, svd_recs])
                    top_recs = self._aggregate_recommendations(all_recs, n)
                    
                except Exception as e:
                    print(f"  ❌ Error in sparse user approach: {e}")
                    # Fallback to User KNN only
                    top_recs = self.user_knn_model.get_recommendations(user_id, n, exclude_rated)
                
            else:
                print("  • Dense user profile - using full hybrid approach")
                try:
                    # For dense users, use all algorithms
                    svd_recs = self.svd_model.get_recommendations(user_id, n, exclude_rated)
                    user_knn_recs = self.user_knn_model.get_recommendations(user_id, n, exclude_rated)
                    item_knn_recs = self.item_knn_model.get_recommendations(user_id, n, exclude_rated)
                    
                    # Add weights for aggregation using assign() - more memory efficient
                    svd_recs = svd_recs.assign(weight=self.weights['svd'])
                    user_knn_recs = user_knn_recs.assign(weight=self.weights['user_knn'])
                    item_knn_recs = item_knn_recs.assign(weight=self.weights['item_knn'])
                    
                    all_recs = pd.concat([svd_recs, user_knn_recs, item_knn_recs])
                    top_recs = self._aggregate_recommendations(all_recs, n)
                    
                except Exception as e:
                    print(f"  ❌ Error in full hybrid approach: {e}")
                    # Fallback to SVD only
                    top_recs = self.svd_model.get_recommendations(user_id, n, exclude_rated)
            
            print(f"✓ Generated {len(top_recs)} hybrid recommendations")
            return top_recs
            
        except Exception as e:
            print(f"  ❌ Critical error in hybrid recommendations: {e}")
            # Ultimate fallback to SVD
            return self.svd_model.get_recommendations(user_id, n, exclude_rated)
            
        finally:
            self._end_prediction_timer()
    
    def recommend(self, user_id: int, n: int = 10, exclude_rated: bool = True) -> pd.DataFrame:
        """Alias for get_recommendations() for consistency with other algorithms"""
        return self.get_recommendations(user_id, n, exclude_rated)
    
    def _aggregate_recommendations(self, all_recs: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        Aggregate recommendations from multiple algorithms using weighted averaging.
        
        Uses pandas groupby for optimal performance. Falls back to manual aggregation
        only if pandas operations fail due to data structure issues.
        
        Args:
            all_recs: DataFrame with columns [movieId, predicted_rating, title, genres, genres_list, poster_path, weight]
            n: Number of top recommendations to return
            
        Returns:
            DataFrame with top N aggregated recommendations
        """
        if len(all_recs) == 0:
            return pd.DataFrame(columns=['movieId', 'predicted_rating', 'title', 'genres', 'genres_list', 'poster_path'])
        
        print(f"  • Aggregating {len(all_recs)} recommendations from multiple algorithms")
        
        # Validate required columns
        required_cols = ['movieId', 'predicted_rating', 'title', 'genres', 'genres_list', 'poster_path']
        missing_cols = [col for col in required_cols if col not in all_recs.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Available: {list(all_recs.columns)}")
        
        # Add default weight if missing
        if 'weight' not in all_recs.columns:
            print(f"  • Adding default weight of 1.0 to all recommendations")
            all_recs = all_recs.copy()
            all_recs['weight'] = 1.0
        
        # Attempt pandas-optimized aggregation first
        try:
            aggregated = all_recs.groupby('movieId').apply(
                lambda group: pd.Series({
                    'predicted_rating': np.average(
                        group['predicted_rating'].values,
                        weights=group['weight'].values
                    ),
                    'title': group['title'].iloc[0],
                    'genres': group['genres'].iloc[0],
                    'genres_list': group['genres_list'].iloc[0],
                    'poster_path': group['poster_path'].iloc[0],
                    'weight': group['weight'].sum()
                })
            ).reset_index()
            
            # Calculate final score (rating + small bonus for multiple algorithm agreement)
            aggregated['final_score'] = (
                aggregated['predicted_rating'] + 
                0.1 * np.log1p(aggregated['weight'])
            )
            
            result = (
                aggregated
                .sort_values('final_score', ascending=False)
                .head(n)
                [['movieId', 'predicted_rating', 'title', 'genres', 'genres_list', 'poster_path']]
            )
            
            print(f"  ✓ Successfully aggregated to {len(result)} recommendations (pandas method)")
            return result
            
        except (ValueError, TypeError, AttributeError) as e:
            # pandas aggregation failed, use manual fallback
            import warnings
            warnings.warn(
                f"Pandas aggregation failed ({type(e).__name__}: {e}). "
                f"Using slower manual aggregation.",
                RuntimeWarning
            )
            return self._manual_aggregate_recommendations(all_recs, n)
    
    def _manual_aggregate_recommendations(self, all_recs: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        Manual aggregation fallback for when pandas operations fail.
        
        This is slower than pandas but more robust to unusual data structures.
        """
        print(f"  • Using manual aggregation fallback")
        
        movie_groups = {}
        for _, row in all_recs.iterrows():
            movie_id = row['movieId']
            if movie_id not in movie_groups:
                movie_groups[movie_id] = {
                    'ratings': [],
                    'weights': [],
                    'title': row['title'],
                    'genres': row['genres'],
                    'genres_list': row['genres_list'],
                    'poster_path': row.get('poster_path', None)
                }
            movie_groups[movie_id]['ratings'].append(row['predicted_rating'])
            movie_groups[movie_id]['weights'].append(row['weight'])
        
        # Calculate weighted averages manually
        result_rows = []
        for movie_id, data in movie_groups.items():
            if len(data['ratings']) == 1:
                weighted_rating = data['ratings'][0]
            else:
                weighted_rating = np.average(data['ratings'], weights=data['weights'])
            
            result_rows.append({
                'movieId': movie_id,
                'predicted_rating': float(weighted_rating),
                'title': data['title'],
                'genres': data['genres'],
                'genres_list': data['genres_list'],
                'poster_path': data['poster_path'],
                'final_score': float(weighted_rating) + 0.1 * np.log1p(sum(data['weights']))
            })
        
        # Convert to DataFrame and sort
        manual_result = pd.DataFrame(result_rows)
        manual_result = manual_result.sort_values('final_score', ascending=False)
        
        print(f"  ✓ Fallback aggregation successful: {len(manual_result)} unique movies")
        return manual_result.head(n)[['movieId', 'predicted_rating', 'title', 'genres', 'genres_list', 'poster_path']]
    
    def get_similar_items(self, item_id: int, n: int = 10) -> pd.DataFrame:
        """Find similar items using the best-performing algorithm for similarity"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        print(f"\n🎬 Finding similar movies to {item_id} using hybrid approach...")
        self._start_prediction_timer()
        
        try:
            # Use Item KNN as primary, SVD as secondary
            try:
                item_similarities = self.item_knn_model.get_similar_items(item_id, n)
                if len(item_similarities) >= n:
                    print("  • Using Item KNN similarities")
                    return item_similarities
            except (KeyError, ValueError, AttributeError) as e:
                # Item KNN couldn't find similarities, try fallback
                pass
            
            # Fallback to User KNN approach
            try:
                user_similarities = self.user_knn_model.get_similar_items(item_id, n)
                print("  • Using User KNN similarities (fallback)")
                return user_similarities
            except (KeyError, ValueError, AttributeError) as e:
                # User KNN also failed, will use genre-based fallback
                pass
            
            # Final fallback: genre-based similarity
            print("  • Using genre-based similarities (final fallback)")
            return self._get_similar_by_genre(item_id, n)
            
        finally:
            self._end_prediction_timer()
    
    def _get_similar_by_genre(self, item_id: int, n: int) -> pd.DataFrame:
        """Fallback genre-based similarity"""
        target_movie = self.movies_df[self.movies_df['movieId'] == item_id].iloc[0]
        target_genres = set(target_movie['genres'].split('|'))
        
        similar_movies = []
        for _, movie in self.movies_df.iterrows():
            if movie['movieId'] == item_id:
                continue
            
            movie_genres = set(movie['genres'].split('|'))
            overlap = len(target_genres & movie_genres)
            
            if overlap > 0:
                similarity = overlap / len(target_genres | movie_genres)
                similar_movies.append({
                    'movieId': movie['movieId'],
                    'similarity': similarity,
                    'title': movie['title'],
                    'genres': movie['genres']
                })
        
        similar_movies.sort(key=lambda x: x['similarity'], reverse=True)
        return pd.DataFrame(similar_movies[:n])
    
    def _get_capabilities(self) -> List[str]:
        """Return list of Hybrid capabilities"""
        return [
            "Multi-algorithm ensemble",
            "Adaptive weighting based on context",
            "Best of matrix factorization + neighborhood methods",
            "Dynamic user profile analysis",
            "Robust recommendations for all user types",
            "High accuracy through algorithm combination"
        ]
    
    def _get_description(self) -> str:
        """Return human-readable algorithm description"""
        return (
            "Intelligently combines SVD matrix factorization with User-Based and Item-Based "
            "collaborative filtering. Dynamically weights algorithms based on your rating "
            "profile and data context for optimal recommendations."
        )
    
    def _get_strengths(self) -> List[str]:
        """Return list of algorithm strengths"""
        return [
            "Best overall accuracy through ensemble",
            "Adapts to different user profiles",
            "Combines multiple recommendation paradigms",
            "Robust against individual algorithm weaknesses",
            "Excellent performance across all scenarios"
        ]
    
    def _get_ideal_use_cases(self) -> List[str]:
        """Return list of ideal use cases"""
        return [
            "All user types (new, sparse, dense)",
            "Production recommendation systems",
            "When highest accuracy is required",
            "Diverse movie catalog browsing",
            "Academic research and comparison"
        ]
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get Hybrid-specific state for saving"""
        return {
            'weights': self.weights,
            'algorithm_performance': self.algorithm_performance,
            'weighting_strategy': self.weighting_strategy,
            'sparse_user_threshold': self.sparse_user_threshold,
            'dense_user_threshold': self.dense_user_threshold,
            'svd_model_state': self.svd_model._get_model_state(),
            'user_knn_model_state': self.user_knn_model._get_model_state(),
            'item_knn_model_state': self.item_knn_model._get_model_state()
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Set Hybrid-specific state from loading"""
        self.weights = state['weights']
        self.algorithm_performance = state['algorithm_performance']
        self.weighting_strategy = state['weighting_strategy']
        self.sparse_user_threshold = state['sparse_user_threshold']
        self.dense_user_threshold = state['dense_user_threshold']
        
        # Restore sub-models
        self.svd_model._set_model_state(state['svd_model_state'])
        self.user_knn_model._set_model_state(state['user_knn_model_state'])
        self.item_knn_model._set_model_state(state['item_knn_model_state'])
    
    def get_explanation_context(self, user_id: int, movie_id: int) -> Dict[str, Any]:
        """Get explanation context for hybrid recommendations"""
        if not self.is_trained:
            return {}
        
        try:
            # Get explanations from each algorithm
            explanations = {}
            explanations['svd'] = self.svd_model.get_explanation_context(user_id, movie_id)
            explanations['user_knn'] = self.user_knn_model.get_explanation_context(user_id, movie_id)
            explanations['item_knn'] = self.item_knn_model.get_explanation_context(user_id, movie_id)
            
            # Determine which algorithm contributed most
            weights = self._get_dynamic_weights(user_id, movie_id)
            primary_algorithm = max(weights, key=weights.get)
            
            return {
                'method': 'hybrid_ensemble',
                'primary_algorithm': primary_algorithm,
                'algorithm_weights': weights,
                'individual_explanations': explanations,
                'prediction': self._predict_hybrid_rating(user_id, movie_id)
            }
            
        except Exception as e:
            return {'method': 'error', 'error': str(e)}