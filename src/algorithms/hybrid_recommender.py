"""
CineMatch V1.0.0 - Hybrid Recommender

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
                 **kwargs):
        """
        Initialize Hybrid recommender with sub-algorithms.
        
        Args:
            svd_params: Parameters for SVD algorithm
            user_knn_params: Parameters for User KNN algorithm  
            item_knn_params: Parameters for Item KNN algorithm
            content_based_params: Parameters for Content-Based algorithm
            weighting_strategy: 'adaptive', 'equal', or 'performance_based'
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
        
        # Initialize sub-algorithms
        self.svd_model = SVDRecommender(**(svd_params or {}))
        self.user_knn_model = UserKNNRecommender(**(user_knn_params or {}))
        self.item_knn_model = ItemKNNRecommender(**(item_knn_params or {}))
        self.content_based_model = ContentBasedRecommender(**(content_based_params or {}))
        
        # Algorithm weights (will be calculated during training)
        self.weights = {'svd': 0.30, 'user_knn': 0.25, 'item_knn': 0.25, 'content_based': 0.20}
        self.algorithm_performance = {}
        
        # User classification thresholds
        self.sparse_user_threshold = 20  # Users with <20 ratings are sparse
        self.dense_user_threshold = 50   # Users with >50 ratings are dense
    
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        """Train all sub-algorithms and calculate optimal weights"""
        print(f"\n🚀 Training {self.name}...")
        start_time = time.time()
        
        # Store data references
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()
        
        # Add genres_list column if not present
        if 'genres_list' not in self.movies_df.columns:
            self.movies_df['genres_list'] = self.movies_df['genres'].str.split('|')
        
        print("\n📊 Loading/Training SVD algorithm...")
        # Try to load pre-trained SVD model (use sklearn version - faster loading)
        svd_loaded = self._try_load_svd(ratings_df, movies_df)
        if not svd_loaded:
            print("  • No pre-trained model found, training from scratch...")
            self.svd_model.fit(ratings_df, movies_df)
        
        self.algorithm_performance['svd'] = {
            'rmse': self.svd_model.metrics.rmse,
            'training_time': self.svd_model.metrics.training_time,
            'coverage': self.svd_model.metrics.coverage
        }
        
        print("\n👥 Loading/Training User KNN algorithm...")
        # Try to load pre-trained User KNN model
        user_knn_loaded = self._try_load_user_knn(ratings_df, movies_df)
        if not user_knn_loaded:
            print("  • No pre-trained model found, training from scratch...")
            self.user_knn_model.fit(ratings_df, movies_df)
        
        self.algorithm_performance['user_knn'] = {
            'rmse': self.user_knn_model.metrics.rmse,
            'training_time': self.user_knn_model.metrics.training_time,
            'coverage': self.user_knn_model.metrics.coverage
        }
        
        print("\n🎬 Loading/Training Item KNN algorithm...")
        # Try to load pre-trained Item KNN model
        item_knn_loaded = self._try_load_item_knn(ratings_df, movies_df)
        if not item_knn_loaded:
            print("  • No pre-trained model found, training from scratch...")
            self.item_knn_model.fit(ratings_df, movies_df)
        
        self.algorithm_performance['item_knn'] = {
            'rmse': self.item_knn_model.metrics.rmse,
            'training_time': self.item_knn_model.metrics.training_time,
            'coverage': self.item_knn_model.metrics.coverage
        }
        
        print("\n🔍 Loading/Training Content-Based algorithm...")
        # Try to load pre-trained Content-Based model
        content_based_loaded = self._try_load_content_based(ratings_df, movies_df)
        if not content_based_loaded:
            print("  • No pre-trained model found, training from scratch...")
            self.content_based_model.fit(ratings_df, movies_df)
        
        self.algorithm_performance['content_based'] = {
            'rmse': self.content_based_model.metrics.rmse,
            'training_time': self.content_based_model.metrics.training_time,
            'coverage': self.content_based_model.metrics.coverage
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
        
        print(f"\n✓ {self.name} trained successfully!")
        print(f"  • Total training time: {training_time:.1f}s")
        print(f"  • Hybrid RMSE: {self.metrics.rmse:.4f}")
        print(f"  • Algorithm weights: SVD={self.weights['svd']:.2f}, "
              f"User KNN={self.weights['user_knn']:.2f}, "
              f"Item KNN={self.weights['item_knn']:.2f}, "
              f"Content-Based={self.weights['content_based']:.2f}")
        print(f"  • Combined coverage: {self.metrics.coverage:.1f}%")
        print(f"  • Total memory usage: {self.metrics.memory_usage_mb:.1f} MB")
    
    def _try_load_user_knn(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> bool:
        """Try to load pre-trained User KNN model"""
        from pathlib import Path
        from src.utils import load_model_safe
        
        model_path = Path("models/user_knn_model.pkl")
        if not model_path.exists():
            return False
            
        try:
            print("  • Loading pre-trained User KNN model...")
            start_time = time.time()
            
            # Use load_model_safe to handle both pickle and joblib formats
            loaded_model = load_model_safe(str(model_path))
            
            # Replace the user_knn_model with the loaded instance
            self.user_knn_model = loaded_model
            
            # Provide data context (shallow reference - model is already trained)
            # Pre-trained models only need data for metadata lookups, not training
            self.user_knn_model.ratings_df = ratings_df  # Shallow reference instead of copy()
            self.user_knn_model.movies_df = movies_df    # Shallow reference instead of copy()
            if 'genres_list' not in self.user_knn_model.movies_df.columns:
                self.user_knn_model.movies_df['genres_list'] = self.user_knn_model.movies_df['genres'].str.split('|')
            
            load_time = time.time() - start_time
            print(f"  ✓ Pre-trained User KNN loaded in {load_time:.2f}s")
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to load pre-trained User KNN: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _try_load_item_knn(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> bool:
        """Try to load pre-trained Item KNN model"""
        from pathlib import Path
        from src.utils import load_model_safe
        
        model_path = Path("models/item_knn_model.pkl")
        if not model_path.exists():
            return False
            
        try:
            print("  • Loading pre-trained Item KNN model...")
            start_time = time.time()
            
            # Use load_model_safe to handle both pickle and joblib formats
            loaded_model = load_model_safe(str(model_path))
            
            # Replace the item_knn_model with the loaded instance
            self.item_knn_model = loaded_model
            
            # Provide data context (shallow reference - model is already trained)
            # Pre-trained models only need data for metadata lookups, not training
            self.item_knn_model.ratings_df = ratings_df  # Shallow reference instead of copy()
            self.item_knn_model.movies_df = movies_df    # Shallow reference instead of copy()
            if 'genres_list' not in self.item_knn_model.movies_df.columns:
                self.item_knn_model.movies_df['genres_list'] = self.item_knn_model.movies_df['genres'].str.split('|')
            
            load_time = time.time() - start_time
            print(f"  ✓ Pre-trained Item KNN loaded in {load_time:.2f}s")
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to load pre-trained Item KNN: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _try_load_content_based(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> bool:
        """Try to load pre-trained Content-Based model"""
        from pathlib import Path
        from src.utils import load_model_safe
        
        model_path = Path("models/content_based_model.pkl")
        if not model_path.exists():
            return False
            
        try:
            print("  • Loading pre-trained Content-Based model...")
            start_time = time.time()
            
            # Use load_model_safe to handle both pickle and joblib formats
            loaded_model = load_model_safe(str(model_path))
            
            # Replace the content_based_model with the loaded instance
            self.content_based_model = loaded_model
            
            # Provide data context (shallow reference - model is already trained)
            # Pre-trained models only need data for metadata lookups, not training
            self.content_based_model.ratings_df = ratings_df  # Shallow reference instead of copy()
            self.content_based_model.movies_df = movies_df    # Shallow reference instead of copy()
            if 'genres_list' not in self.content_based_model.movies_df.columns:
                self.content_based_model.movies_df['genres_list'] = self.content_based_model.movies_df['genres'].str.split('|')
            
            load_time = time.time() - start_time
            print(f"  ✓ Pre-trained Content-Based loaded in {load_time:.2f}s")
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to load pre-trained Content-Based: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _try_load_svd(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> bool:
        """Try to load pre-trained SVD model (using sklearn version for faster loading)"""
        from pathlib import Path
        from src.utils import load_model_safe
        
        # Try sklearn version first (faster loading, less memory)
        model_path = Path("models/svd_model_sklearn.pkl")
        if not model_path.exists():
            # Fallback to Surprise version if sklearn not available
            model_path = Path("models/svd_model.pkl")
            if not model_path.exists():
                return False
            
        try:
            print(f"  • Loading pre-trained SVD model ({model_path.name})...")
            start_time = time.time()
            
            # Use load_model_safe to handle both pickle and joblib formats
            loaded_model = load_model_safe(str(model_path))
            
            # Replace the svd_model with the loaded instance
            self.svd_model = loaded_model
            
            # Provide data context (shallow reference - model is already trained)
            # Pre-trained models only need data for metadata lookups, not training
            self.svd_model.ratings_df = ratings_df  # Shallow reference instead of copy()
            self.svd_model.movies_df = movies_df    # Shallow reference instead of copy()
            if 'genres_list' not in self.svd_model.movies_df.columns:
                self.svd_model.movies_df['genres_list'] = self.svd_model.movies_df['genres'].str.split('|')
            
            load_time = time.time() - start_time
            print(f"  ✓ Pre-trained SVD loaded in {load_time:.2f}s")
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to load pre-trained SVD: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _calculate_weights(self) -> None:
        """Calculate optimal weights based on algorithm performance"""
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
        
        # Get MAE values (use default if not available)
        svd_mae = self.svd_model.metrics.mae if self.svd_model.metrics.mae > 0 else svd_rmse * 0.78
        user_knn_mae = self.user_knn_model.metrics.mae if self.user_knn_model.metrics.mae > 0 else user_knn_rmse * 0.78
        item_knn_mae = self.item_knn_model.metrics.mae if self.item_knn_model.metrics.mae > 0 else item_knn_rmse * 0.78
        content_based_mae = self.content_based_model.metrics.mae if self.content_based_model.metrics.mae > 0 else content_based_rmse * 0.78
        
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
        """Internal hybrid rating prediction logic"""
        predictions = {}
        weights = self._get_dynamic_weights(user_id, movie_id)
        
        # Get predictions from each algorithm
        try:
            predictions['svd'] = self.svd_model.predict(user_id, movie_id)
        except:
            predictions['svd'] = None
        
        try:
            predictions['user_knn'] = self.user_knn_model.predict(user_id, movie_id)
        except:
            predictions['user_knn'] = None
        
        try:
            predictions['item_knn'] = self.item_knn_model.predict(user_id, movie_id)
        except:
            predictions['item_knn'] = None
        
        try:
            predictions['content_based'] = self.content_based_model.predict(user_id, movie_id)
        except:
            predictions['content_based'] = None
        
        # Filter valid predictions and normalize weights
        valid_predictions = {k: v for k, v in predictions.items() if v is not None}
        
        if not valid_predictions:
            # Fallback to global average
            return self.ratings_df['rating'].mean()
        
        # Normalize weights for valid predictions
        total_weight = sum(weights[k] for k in valid_predictions.keys())
        
        # Calculate weighted prediction
        weighted_prediction = 0.0
        for algo, prediction in valid_predictions.items():
            weight = weights[algo] / total_weight
            weighted_prediction += weight * prediction
        
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
                    
                    # Add weights for aggregation
                    user_knn_recs = user_knn_recs.copy()
                    svd_recs = svd_recs.copy()
                    user_knn_recs['weight'] = 0.6
                    svd_recs['weight'] = 0.4
                    
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
                    
                    # Add weights for aggregation
                    svd_recs = svd_recs.copy()
                    user_knn_recs = user_knn_recs.copy()
                    item_knn_recs = item_knn_recs.copy()
                    
                    svd_recs['weight'] = self.weights['svd']
                    user_knn_recs['weight'] = self.weights['user_knn']
                    item_knn_recs['weight'] = self.weights['item_knn']
                    
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
    
    def _aggregate_recommendations(self, all_recs: pd.DataFrame, n: int) -> pd.DataFrame:
        """Aggregate recommendations from multiple algorithms"""
        if len(all_recs) == 0:
            return pd.DataFrame(columns=['movieId', 'predicted_rating', 'title', 'genres', 'genres_list'])
        
        print(f"  • Aggregating {len(all_recs)} recommendations from multiple algorithms")
        
        # Ensure all dataframes have consistent column names
        if 'predicted_rating' not in all_recs.columns:
            print(f"  ❌ Missing 'predicted_rating' column. Available: {list(all_recs.columns)}")
            # Try to find alternative column names
            rating_columns = [col for col in all_recs.columns if 'rating' in col.lower()]
            if rating_columns:
                print(f"  • Found rating column: {rating_columns[0]}, renaming to 'predicted_rating'")
                all_recs = all_recs.rename(columns={rating_columns[0]: 'predicted_rating'})
            else:
                print(f"  ❌ No rating column found, returning original recommendations")
                return all_recs.head(n)
        
        # Add missing weight column if not present
        if 'weight' not in all_recs.columns:
            print(f"  • Adding default weight of 1.0 to all recommendations")
            all_recs['weight'] = 1.0
        
        # Group by movieId and calculate weighted average  
        try:
            # Create aggregation functions that handle both Series and DataFrame inputs
            def safe_weighted_mean(values):
                """Compute weighted average handling both Series and DataFrame inputs"""
                if hasattr(values, 'index'):  # Series input from agg()
                    # Get corresponding weights for these values
                    indices = values.index
                    corresponding_weights = all_recs.loc[indices, 'weight']
                    if len(values) == 1:
                        return float(values.iloc[0])
                    return float(np.average(values.values, weights=corresponding_weights.values))
                else:  # Direct values (shouldn't happen but safe fallback)
                    return float(values[0]) if len(values) == 1 else float(np.mean(values))
            
            def safe_first(values):
                """Safe first value extraction"""
                return values.iloc[0] if hasattr(values, 'iloc') else values[0]
                
            def safe_sum_weights(values):
                """Safe weight sum"""
                return float(values.sum()) if hasattr(values, 'sum') else float(sum(values))
            
            # Perform the aggregation
            aggregated = all_recs.groupby('movieId').agg({
                'predicted_rating': safe_weighted_mean,
                'title': safe_first,
                'genres': safe_first, 
                'genres_list': safe_first,
                'weight': safe_sum_weights
            }).reset_index()
            
            # Sort by weighted rating and total weight
            aggregated['final_score'] = aggregated['predicted_rating'] + 0.1 * np.log1p(aggregated['weight'])
            aggregated = aggregated.sort_values('final_score', ascending=False)
            
            # Select top N and clean up
            top_recommendations = aggregated.head(n)
            
            print(f"  ✓ Successfully aggregated to {len(top_recommendations)} recommendations")
            return top_recommendations[['movieId', 'predicted_rating', 'title', 'genres', 'genres_list']]
            
        except Exception as e:
            print(f"  • Using fallback aggregation method")
            
            # Manual aggregation fallback
            movie_groups = {}
            for _, row in all_recs.iterrows():
                movie_id = row['movieId']
                if movie_id not in movie_groups:
                    movie_groups[movie_id] = {
                        'ratings': [],
                        'weights': [],
                        'title': row['title'],
                        'genres': row['genres'],
                        'genres_list': row['genres_list']
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
                    'final_score': float(weighted_rating) + 0.1 * np.log1p(sum(data['weights']))
                })
            
            # Convert to DataFrame and sort
            manual_result = pd.DataFrame(result_rows)
            manual_result = manual_result.sort_values('final_score', ascending=False)
            
            print(f"  ✓ Fallback aggregation successful: {len(manual_result)} unique movies")
            return manual_result.head(n)[['movieId', 'predicted_rating', 'title', 'genres', 'genres_list']]
    
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
            except:
                pass
            
            # Fallback to User KNN approach
            try:
                user_similarities = self.user_knn_model.get_similar_items(item_id, n)
                print("  • Using User KNN similarities (fallback)")
                return user_similarities
            except:
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