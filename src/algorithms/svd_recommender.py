"""
CineMatch V1.0.0 - SVD Recommender Wrapper

Wraps the existing SimpleSVDRecommender to conform to BaseRecommender interface.
Maintains backwards compatibility while enabling multi-algorithm support.

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
from src.svd_model_sklearn import SimpleSVDRecommender


class SVDRecommender(BaseRecommender):
    """
    SVD (Singular Value Decomposition) Recommender.
    
    Wrapper around SimpleSVDRecommender to provide BaseRecommender interface.
    Uses matrix factorization to find latent factors in user-movie interactions.
    """
    
    def __init__(self, n_components: int = 100, **kwargs):
        """
        Initialize SVD recommender.
        
        Args:
            n_components: Number of latent factors (default: 100)
            **kwargs: Additional parameters
        """
        super().__init__("SVD Matrix Factorization", n_components=n_components, **kwargs)
        self.n_components = n_components
        self.model = SimpleSVDRecommender(n_components=n_components)
    
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        """Train the SVD model on the provided data"""
        print(f"\n🧠 Training {self.name}...")
        start_time = time.time()
        
        # Store data references
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()
        
        # Add genres_list column if not present
        if 'genres_list' not in self.movies_df.columns:
            self.movies_df['genres_list'] = self.movies_df['genres'].str.split('|')
        
        # Train the underlying model
        self.model.fit(ratings_df)
        
        # Update training metrics
        training_time = time.time() - start_time
        self.metrics.training_time = training_time
        self.is_trained = True
        
        # Calculate RMSE on a test set (sample for speed)
        test_sample = ratings_df.sample(min(10000, len(ratings_df)), random_state=42)
        rmse_sum = 0
        for _, row in test_sample.iterrows():
            try:
                pred = self.model.predict(row['userId'], row['movieId'])
                rmse_sum += (pred - row['rating']) ** 2
            except:
                continue
        
        self.metrics.rmse = np.sqrt(rmse_sum / len(test_sample))
        
        # Calculate coverage (% of movies that can be recommended)
        self.metrics.coverage = (len(self.model.movie_ids) / len(movies_df)) * 100
        
        # Calculate memory usage (estimate based on model components)
        memory_bytes = 0
        if hasattr(self.model, 'user_factors') and self.model.user_factors is not None:
            memory_bytes += self.model.user_factors.nbytes
        if hasattr(self.model, 'movie_factors') and self.model.movie_factors is not None:
            memory_bytes += self.model.movie_factors.nbytes
        self.metrics.memory_usage_mb = memory_bytes / (1024 * 1024)
        
        print(f"✓ {self.name} trained successfully!")
        print(f"  • Training time: {training_time:.1f}s")
        print(f"  • RMSE: {self.metrics.rmse:.4f}")
        print(f"  • Coverage: {self.metrics.coverage:.1f}%")
        print(f"  • Memory: {self.metrics.memory_usage_mb:.1f} MB")
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating for a specific user-movie pair"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        self._start_prediction_timer()
        try:
            prediction = self.model.predict(user_id, movie_id)
            return float(prediction)
        finally:
            self._end_prediction_timer()
    
    def get_recommendations(
        self,
        user_id: int,
        n: int = 10,
        exclude_rated: bool = True
    ) -> pd.DataFrame:
        """Generate top-N recommendations for a user"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Check if user exists, if not use popularity-based fallback
        user_exists = self.validate_user_exists(user_id)
        if not user_exists:
            print(f"\n🎯 User {user_id} not in training data - generating popular recommendations...")
            return self._get_popular_movies(n)
        
        print(f"\n🎯 Generating {self.name} recommendations for User {user_id}...")
        self._start_prediction_timer()
        
        try:
            # Get user's rated movies if excluding them
            rated_movie_ids = set()
            if exclude_rated:
                user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
                rated_movie_ids = set(user_ratings['movieId'].values)
                print(f"  • Excluding {len(rated_movie_ids)} already-rated movies")
            
            # Get all candidate movies
            candidate_movies = [
                mid for mid in self.model.movie_ids 
                if not exclude_rated or mid not in rated_movie_ids
            ]
            
            print(f"  • Evaluating {len(candidate_movies)} candidate movies...")
            
            # Generate predictions
            predictions = []
            for movie_id in candidate_movies:
                try:
                    pred_rating = self.model.predict(user_id, movie_id)
                    predictions.append({
                        'movieId': movie_id,
                        'predicted_rating': float(pred_rating)
                    })
                except:
                    continue  # Skip movies that can't be predicted
            
            # Convert to DataFrame and sort
            predictions_df = pd.DataFrame(predictions)
            predictions_df = predictions_df.sort_values('predicted_rating', ascending=False)
            
            # Get top N
            top_predictions = predictions_df.head(n)
            
            # Merge with movie info
            recommendations = top_predictions.merge(
                self.movies_df[['movieId', 'title', 'genres', 'genres_list']],
                on='movieId',
                how='left'
            )
            
            print(f"✓ Generated {len(recommendations)} recommendations")
            return recommendations
            
        finally:
            self._end_prediction_timer()
    
    def _get_popular_movies(self, n: int) -> pd.DataFrame:
        """Fallback: return most popular movies for new users"""
        movie_ratings = self.ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        movie_ratings.columns = ['movieId', 'count', 'mean_rating']
        
        # Sort by popularity (count * mean_rating)
        movie_ratings['popularity'] = movie_ratings['count'] * movie_ratings['mean_rating']
        popular_movies = movie_ratings.sort_values('popularity', ascending=False).head(n)
        
        # Format as recommendations
        recommendations = popular_movies.merge(
            self.movies_df[['movieId', 'title', 'genres', 'genres_list']],
            on='movieId',
            how='left'
        )
        recommendations['predicted_rating'] = recommendations['mean_rating']
        
        print(f"✓ Generated {len(recommendations)} popular movie recommendations")
        return recommendations[['movieId', 'predicted_rating', 'title', 'genres', 'genres_list']]
    
    def get_similar_items(self, item_id: int, n: int = 10) -> pd.DataFrame:
        """Find movies similar to the given movie using latent factors"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        if not self.validate_movie_exists(item_id):
            raise ValueError(f"Movie ID {item_id} not found in dataset")
        
        print(f"\n🎬 Finding movies similar to Movie {item_id}...")
        self._start_prediction_timer()
        
        try:
            # Get movie factors for the target movie
            if item_id not in self.model.movie_mapper:
                raise ValueError(f"Movie {item_id} not in trained model")
            
            movie_idx = self.model.movie_mapper[item_id]
            movie_factors = self.model.movie_factors[movie_idx]
            
            # Calculate similarities with all other movies
            similarities = []
            for other_idx, other_movie_id in self.model.movie_inv_mapper.items():
                if other_movie_id == item_id:
                    continue  # Skip self
                
                other_factors = self.model.movie_factors[other_idx]
                
                # Cosine similarity
                similarity = np.dot(movie_factors, other_factors) / (
                    np.linalg.norm(movie_factors) * np.linalg.norm(other_factors) + 1e-8
                )
                
                similarities.append({
                    'movieId': other_movie_id,
                    'similarity': float(similarity)
                })
            
            # Convert to DataFrame and sort
            similarities_df = pd.DataFrame(similarities)
            similarities_df = similarities_df.sort_values('similarity', ascending=False)
            
            # Get top N
            top_similar = similarities_df.head(n)
            
            # Merge with movie info
            similar_movies = top_similar.merge(
                self.movies_df[['movieId', 'title', 'genres']],
                on='movieId',
                how='left'
            )
            
            print(f"✓ Found {len(similar_movies)} similar movies")
            return similar_movies
            
        finally:
            self._end_prediction_timer()
    
    def _get_capabilities(self) -> List[str]:
        """Return list of SVD capabilities"""
        return [
            "Matrix factorization",
            "Latent factor modeling", 
            "Handles sparse data efficiently",
            "Captures hidden user preferences",
            "Scalable to large datasets",
            "Global optimization"
        ]
    
    def _get_description(self) -> str:
        """Return human-readable algorithm description"""
        return (
            "Uses Singular Value Decomposition to discover hidden patterns in user ratings. "
            "Learns latent factors that represent user preferences and movie characteristics, "
            "enabling accurate predictions even for users with few ratings."
        )
    
    def _get_strengths(self) -> List[str]:
        """Return list of algorithm strengths"""
        return [
            "Excellent accuracy on dense datasets",
            "Captures complex user-item interactions", 
            "Handles the 'long tail' of unpopular items",
            "Dimensionality reduction reduces noise",
            "Proven performance on MovieLens datasets"
        ]
    
    def _get_ideal_use_cases(self) -> List[str]:
        """Return list of ideal use cases"""
        return [
            "Users with rich rating history (>50 ratings)",
            "Large-scale recommendation systems",
            "When accuracy is the primary goal",
            "Discovering hidden preferences",
            "Cold-start items (new movies with few ratings)"
        ]
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get SVD-specific state for saving"""
        return {
            'n_components': self.n_components,
            'model_state': {
                'user_mapper': self.model.user_mapper,
                'movie_mapper': self.model.movie_mapper,
                'user_inv_mapper': self.model.user_inv_mapper,
                'movie_inv_mapper': self.model.movie_inv_mapper,
                'user_factors': self.model.user_factors,
                'movie_factors': self.model.movie_factors,
                'global_mean': self.model.global_mean,
                'user_bias': self.model.user_bias,
                'movie_bias': self.model.movie_bias,
                'user_ids': self.model.user_ids,
                'movie_ids': self.model.movie_ids
            },
            'metrics': {
                'rmse': self.metrics.rmse,
                'training_time': self.metrics.training_time,
                'coverage': self.metrics.coverage,
                'memory_usage_mb': self.metrics.memory_usage_mb
            }
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Set SVD-specific state from loading"""
        self.n_components = state['n_components']
        model_state = state['model_state']
        
        # Recreate the model
        self.model = SimpleSVDRecommender(n_components=self.n_components)
        
        # Restore all state
        for key, value in model_state.items():
            setattr(self.model, key, value)
        
        # Restore metrics if available
        if 'metrics' in state:
            metrics_data = state['metrics']
            self.metrics.rmse = metrics_data.get('rmse', 0.0)
            self.metrics.training_time = metrics_data.get('training_time', 0.0)
            self.metrics.coverage = metrics_data.get('coverage', 0.0)
            self.metrics.memory_usage_mb = metrics_data.get('memory_usage_mb', 0.0)
        else:
            # Calculate coverage from loaded model if metrics not saved
            if hasattr(self.model, 'movie_ids') and self.movies_df is not None:
                self.metrics.coverage = (len(self.model.movie_ids) / len(self.movies_df)) * 100
        
        # Mark as trained (critical for Hybrid loading)
        self.is_trained = True
    
    def get_explanation_context(self, user_id: int, movie_id: int) -> Dict[str, Any]:
        """
        Get context for explaining why this movie was recommended.
        
        Returns:
            Dictionary with explanation context specific to SVD
        """
        if not self.is_trained:
            return {}
        
        try:
            # Get user and movie factors
            user_idx = self.model.user_mapper.get(user_id)
            movie_idx = self.model.movie_mapper.get(movie_id)
            
            if user_idx is None or movie_idx is None:
                return {'method': 'fallback', 'reason': 'User or movie not in training data'}
            
            user_factors = self.model.user_factors[user_idx]
            movie_factors = self.model.movie_factors[movie_idx]
            
            # Calculate factor contributions
            factor_contributions = user_factors * movie_factors
            top_factors = np.argsort(factor_contributions)[-3:][::-1]
            
            return {
                'method': 'latent_factors',
                'prediction': self.model.predict(user_id, movie_id),
                'global_mean': self.model.global_mean,
                'user_bias': self.model.user_bias.get(user_id, 0),
                'movie_bias': self.model.movie_bias.get(movie_id, 0),
                'top_factors': top_factors.tolist(),
                'factor_strengths': factor_contributions[top_factors].tolist()
            }
            
        except Exception as e:
            return {'method': 'error', 'error': str(e)}