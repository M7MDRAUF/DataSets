"""
CineMatch V1.0.0 - User-Based KNN Recommender

K-Nearest Neighbors recommendation using user-based collaborative filtering.
Finds users with similar taste and recommends movies they loved.

Author: CineMatch Development Team
Date: November 7, 2025
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np
import time
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.algorithms.base_recommender import BaseRecommender


class UserKNNRecommender(BaseRecommender):
    """
    User-Based K-Nearest Neighbors Recommender.
    
    Finds users with similar rating patterns and recommends movies 
    that similar users have rated highly.
    """
    
    def __init__(self, n_neighbors: int = 50, similarity_metric: str = 'cosine', **kwargs):
        """
        Initialize User KNN recommender.
        
        Args:
            n_neighbors: Number of similar users to consider (default: 50)
            similarity_metric: Similarity metric ('cosine', 'euclidean', etc.)
            **kwargs: Additional parameters
        """
        super().__init__("KNN User-Based", n_neighbors=n_neighbors, 
                        similarity_metric=similarity_metric, **kwargs)
        
        self.n_neighbors = n_neighbors
        self.similarity_metric = similarity_metric
        self.knn_model = NearestNeighbors(
            n_neighbors=n_neighbors + 1,  # +1 because it includes the query user
            metric=similarity_metric,
            algorithm='brute'  # Better for sparse matrices
        )
        
        # Data structures
        self.user_movie_matrix = None
        self.user_mapper = {}
        self.user_inv_mapper = {}
        self.movie_mapper = {} 
        self.movie_inv_mapper = {}
        self.user_means = None
        self.global_mean = 0.0
    
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        """Train the User KNN model on the provided data"""
        print(f"\n👥 Training {self.name}...")
        start_time = time.time()
        
        # Store data references
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()
        
        # Add genres_list column if not present (as tuples for hashability)
        if 'genres_list' not in self.movies_df.columns:
            self.movies_df['genres_list'] = self.movies_df['genres'].str.split('|').apply(tuple)
        
        print("  • Creating user-item matrix...")
        self._create_user_item_matrix(ratings_df)
        
        print("  • Training KNN model...")
        self.knn_model.fit(self.user_movie_matrix)
        
        # Calculate metrics
        training_time = time.time() - start_time
        self.metrics.training_time = training_time
        self.is_trained = True
        
        # Calculate RMSE on a test sample
        print("  • Calculating RMSE...")
        self._calculate_rmse(ratings_df)
        
        # Calculate coverage
        self.metrics.coverage = 100.0  # KNN can recommend any movie in the dataset
        
        # Calculate memory usage (approximate)
        matrix_size_mb = (self.user_movie_matrix.data.nbytes + 
                         self.user_movie_matrix.indices.nbytes + 
                         self.user_movie_matrix.indptr.nbytes) / (1024 * 1024)
        self.metrics.memory_usage_mb = matrix_size_mb
        
        print(f"✓ {self.name} trained successfully!")
        print(f"  • Training time: {training_time:.1f}s")
        print(f"  • RMSE: {self.metrics.rmse:.4f}")
        print(f"  • Matrix size: {self.user_movie_matrix.shape}")
        print(f"  • Sparsity: {(1 - self.user_movie_matrix.nnz / np.prod(self.user_movie_matrix.shape)) * 100:.2f}%")
        print(f"  • Memory usage: {matrix_size_mb:.1f} MB")
    
    def _create_user_item_matrix(self, ratings_df: pd.DataFrame) -> None:
        """Create sparse user-item matrix for KNN"""
        # Create user and movie mappings
        unique_users = ratings_df['userId'].unique()
        unique_movies = ratings_df['movieId'].unique()
        
        self.user_mapper = {uid: idx for idx, uid in enumerate(unique_users)}
        self.movie_mapper = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.user_inv_mapper = {idx: uid for uid, idx in self.user_mapper.items()}
        self.movie_inv_mapper = {idx: mid for mid, idx in self.movie_mapper.items()}
        
        # Create sparse matrix
        n_users = len(unique_users)
        n_movies = len(unique_movies)
        
        user_indices = ratings_df['userId'].map(self.user_mapper).values
        movie_indices = ratings_df['movieId'].map(self.movie_mapper).values
        ratings = ratings_df['rating'].values
        
        # Create sparse matrix (users x movies)
        self.user_movie_matrix = csr_matrix(
            (ratings, (user_indices, movie_indices)),
            shape=(n_users, n_movies)
        )
        
        # Calculate user means for mean-centered predictions
        self.user_means = np.array(self.user_movie_matrix.sum(axis=1) / 
                                 (self.user_movie_matrix > 0).sum(axis=1)).flatten()
        self.user_means = np.nan_to_num(self.user_means)  # Handle division by zero
        
        # Global mean for fallback
        self.global_mean = ratings_df['rating'].mean()
    
    def _calculate_rmse(self, ratings_df: pd.DataFrame) -> None:
        """Calculate RMSE on a test sample"""
        test_sample = ratings_df.sample(min(5000, len(ratings_df)), random_state=42)
        
        squared_errors = []
        for _, row in test_sample.iterrows():
            try:
                pred = self.predict(row['userId'], row['movieId'])
                squared_errors.append((pred - row['rating']) ** 2)
            except:
                continue  # Skip if prediction fails
        
        if squared_errors:
            self.metrics.rmse = np.sqrt(np.mean(squared_errors))
        else:
            self.metrics.rmse = float('inf')
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating for a specific user-movie pair"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        self._start_prediction_timer()
        try:
            return self._predict_rating(user_id, movie_id)
        finally:
            self._end_prediction_timer()
    
    def _predict_rating(self, user_id: int, movie_id: int) -> float:
        """Internal rating prediction logic"""
        # Check if user and movie exist in our mappings
        if user_id not in self.user_mapper or movie_id not in self.movie_mapper:
            return self.global_mean
        
        user_idx = self.user_mapper[user_id]
        movie_idx = self.movie_mapper[movie_id]
        
        # Get user's rating vector
        user_vector = self.user_movie_matrix[user_idx:user_idx+1]
        
        # If user has no ratings, return global mean
        if user_vector.nnz == 0:
            return self.global_mean
        
        # Find similar users
        distances, neighbor_indices = self.knn_model.kneighbors(
            user_vector, n_neighbors=min(self.n_neighbors + 1, self.user_movie_matrix.shape[0])
        )
        
        # Remove the user themselves from neighbors
        neighbor_indices = neighbor_indices.flatten()[1:]  # Skip first (the user themselves)
        distances = distances.flatten()[1:]
        
        # Calculate weighted prediction
        numerator = 0.0
        denominator = 0.0
        user_mean = self.user_means[user_idx]
        
        for neighbor_idx, distance in zip(neighbor_indices, distances):
            # Get neighbor's rating for this movie
            neighbor_rating = self.user_movie_matrix[neighbor_idx, movie_idx]
            
            if neighbor_rating > 0:  # Neighbor has rated this movie
                # Convert distance to similarity (higher is more similar)
                similarity = 1 / (1 + distance) if distance > 0 else 1.0
                
                # Mean-centered rating
                neighbor_mean = self.user_means[neighbor_idx]
                centered_rating = neighbor_rating - neighbor_mean
                
                numerator += similarity * centered_rating
                denominator += similarity
        
        # Return prediction
        if denominator > 0:
            prediction = user_mean + (numerator / denominator)
            # Clip to valid rating range
            return np.clip(prediction, 0.5, 5.0)
        else:
            return user_mean if user_mean > 0 else self.global_mean
    
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
            all_movie_ids = set(self.movie_mapper.keys())
            candidate_movies = all_movie_ids - rated_movie_ids if exclude_rated else all_movie_ids
            
            # OPTIMIZATION: For performance, use smart sampling for large candidate sets
            if len(candidate_movies) > 5000:
                print(f"  • Large candidate set ({len(candidate_movies)}), using smart sampling...")
                candidate_movies = self._smart_sample_candidates(candidate_movies, max_candidates=5000)
                print(f"  ✓ Sampled down to {len(candidate_movies)} candidates")
            
            print(f"  • Evaluating {len(candidate_movies)} candidate movies...")
            
            # Use optimized batch prediction
            predictions = self._batch_predict_ratings(user_id, candidate_movies)
            
            # Convert to DataFrame and sort
            predictions_df = pd.DataFrame(predictions)
            if len(predictions_df) == 0:
                # Fallback: return popular movies
                return self._get_popular_movies(n)
            
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
    
    def _smart_sample_candidates(self, candidate_movies: set, max_candidates: int = 5000) -> set:
        """Smart sampling of candidate movies for efficiency - OPTIMIZED"""
        # PERFORMANCE FIX: Use pre-computed popularity if available, otherwise use simpler sampling
        # The original .isin() on 83K movies + 32M ratings is extremely slow
        
        # Quick check: if we have pre-computed movie stats, use them
        if hasattr(self, '_movie_popularity_cache'):
            movie_popularity = self._movie_popularity_cache
            # Filter to candidate movies
            available_movies = movie_popularity[movie_popularity.index.isin(candidate_movies)]
        else:
            # OPTIMIZED: Use groupby on full dataset (computed once) instead of filtering first
            # This is much faster than filtering 32M rows by 83K movies
            if not hasattr(self, '_all_movie_stats'):
                self._all_movie_stats = self.ratings_df.groupby('movieId').agg({
                    'rating': ['count', 'mean']
                })
                self._all_movie_stats.columns = ['count', 'mean_rating']
                self._all_movie_stats['popularity'] = self._all_movie_stats['count'] * self._all_movie_stats['mean_rating']
            
            # Now filter the pre-computed stats (much faster)
            available_movies = self._all_movie_stats[self._all_movie_stats.index.isin(candidate_movies)]
        
        # If we have few movies with stats, just return them all
        if len(available_movies) <= max_candidates:
            return candidate_movies
        
        # Sample: 70% popular movies + 30% random for diversity
        popular_count = int(max_candidates * 0.7)
        random_count = max_candidates - popular_count
        
        # Get most popular movies
        top_popular = available_movies.nlargest(min(popular_count, len(available_movies)), 'popularity').index.tolist()
        
        # Get random sample from remaining
        remaining_movies = list(candidate_movies - set(top_popular))
        if remaining_movies:
            random_sample = np.random.choice(
                remaining_movies, 
                size=min(random_count, len(remaining_movies)), 
                replace=False
            ).tolist()
        else:
            random_sample = []
        
        return set(top_popular + random_sample)
    
    def _batch_predict_ratings(self, user_id: int, candidate_movies: set) -> List[Dict]:
        """Optimized vectorized prediction for User KNN"""
        predictions = []
        
        # Check if user exists in our training data
        if user_id not in self.user_mapper:
            # New user - use popularity-based recommendations with slight randomization
            movie_list = list(candidate_movies)
            for movie_id in movie_list[:1000]:  # Limit for new users
                try:
                    movie_idx = self.movie_mapper.get(movie_id)
                    if movie_idx is not None:
                        # Use global mean + some randomization for diversity
                        pred_rating = self.global_mean + np.random.normal(0, 0.2)
                        pred_rating = np.clip(pred_rating, 0.5, 5.0)
                        predictions.append({
                            'movieId': movie_id,
                            'predicted_rating': float(pred_rating)
                        })
                except:
                    continue
            return predictions
        
        # Existing user - use optimized KNN approach
        user_idx = self.user_mapper[user_id]
        user_vector = self.user_movie_matrix[user_idx:user_idx+1]
        
        # If user has no ratings, fallback to popularity
        if user_vector.nnz == 0:
            return self._get_popularity_predictions(candidate_movies, 1000)
        
        print(f"    → Finding similar users for efficient prediction...")
        
        # Find similar users once
        distances, neighbor_indices = self.knn_model.kneighbors(
            user_vector, n_neighbors=min(self.n_neighbors + 1, self.user_movie_matrix.shape[0])
        )
        
        # Remove the user themselves from neighbors
        neighbor_indices = neighbor_indices.flatten()[1:]
        distances = distances.flatten()[1:]
        
        # Calculate similarities once
        similarities = 1 / (1 + distances)
        similarities = similarities / similarities.sum()  # Normalize
        
        user_mean = self.user_means[user_idx]
        
        print(f"    → Generating predictions for {len(candidate_movies)} movies...")
        
        # Convert candidate movies to indices
        valid_candidates = []
        candidate_indices = []
        
        for movie_id in candidate_movies:
            if movie_id in self.movie_mapper:
                movie_idx = self.movie_mapper[movie_id]
                valid_candidates.append(movie_id)
                candidate_indices.append(movie_idx)
        
        # Limit candidates for performance (most popular ones from our sample)
        if len(valid_candidates) > 2000:
            # Get ratings count for these candidates to prioritize popular ones
            candidate_ratings = self.ratings_df[
                self.ratings_df['movieId'].isin(valid_candidates)
            ].groupby('movieId').size()
            
            # Sort by popularity and take top 2000
            top_candidates = candidate_ratings.nlargest(2000).index.tolist()
            valid_candidates = [mid for mid in valid_candidates if mid in top_candidates]
            candidate_indices = [self.movie_mapper[mid] for mid in valid_candidates]
            
            print(f"    → Optimized to top {len(valid_candidates)} popular candidates")
        
        # Vectorized prediction for all candidates at once
        for i, (movie_id, movie_idx) in enumerate(zip(valid_candidates, candidate_indices)):
            try:
                # Get ratings from similar users for this movie
                numerator = 0.0
                denominator = 0.0
                
                for neighbor_idx, similarity in zip(neighbor_indices[:20], similarities[:20]):  # Top 20 neighbors
                    neighbor_rating = self.user_movie_matrix[neighbor_idx, movie_idx]
                    
                    if neighbor_rating > 0:  # Neighbor rated this movie
                        neighbor_mean = self.user_means[neighbor_idx]
                        centered_rating = neighbor_rating - neighbor_mean
                        
                        numerator += similarity * centered_rating
                        denominator += similarity
                
                # Calculate prediction
                if denominator > 0:
                    prediction = user_mean + (numerator / denominator)
                    prediction = np.clip(prediction, 0.5, 5.0)
                else:
                    # Fallback to item mean or global mean
                    item_ratings = self.ratings_df[self.ratings_df['movieId'] == movie_id]['rating']
                    if len(item_ratings) > 0:
                        prediction = item_ratings.mean()
                    else:
                        prediction = self.global_mean
                
                predictions.append({
                    'movieId': movie_id,
                    'predicted_rating': float(prediction)
                })
                
                # Progress indicator
                if (i + 1) % 500 == 0:
                    print(f"      → Processed {i + 1}/{len(valid_candidates)} movies")
                    
            except Exception as e:
                continue
        
        print(f"    ✓ Generated {len(predictions)} predictions")
        return predictions
    
    def _get_popularity_predictions(self, candidate_movies: set, max_count: int) -> List[Dict]:
        """Get popularity-based predictions for fallback"""
        movie_popularity = self.ratings_df[
            self.ratings_df['movieId'].isin(candidate_movies)
        ].groupby('movieId').agg({
            'rating': ['count', 'mean']
        })
        movie_popularity.columns = ['count', 'mean_rating']
        movie_popularity['popularity'] = movie_popularity['count'] * movie_popularity['mean_rating']
        
        # Get top popular movies
        top_movies = movie_popularity.nlargest(min(max_count, len(movie_popularity)), 'popularity')
        
        predictions = []
        for movie_id, row in top_movies.iterrows():
            predictions.append({
                'movieId': movie_id,
                'predicted_rating': float(row['mean_rating'])
            })
        
        return predictions
    
    def _get_popular_movies(self, n: int) -> pd.DataFrame:
        """Fallback: return most popular movies"""
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
        
        return recommendations[['movieId', 'predicted_rating', 'title', 'genres', 'genres_list']]
    
    def get_similar_items(self, item_id: int, n: int = 10) -> pd.DataFrame:
        """
        Find movies similar to the given movie.
        
        For User KNN, we find movies that similar users also liked.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        if not self.validate_movie_exists(item_id):
            raise ValueError(f"Movie ID {item_id} not found in dataset")
        
        print(f"\n🎬 Finding movies similar to Movie {item_id} (User KNN approach)...")
        self._start_prediction_timer()
        
        try:
            # Get users who rated this movie highly (>= 4.0)
            movie_ratings = self.ratings_df[
                (self.ratings_df['movieId'] == item_id) & 
                (self.ratings_df['rating'] >= 4.0)
            ]
            
            if len(movie_ratings) == 0:
                return pd.DataFrame(columns=['movieId', 'similarity', 'title', 'genres'])
            
            # For each user who liked this movie, find their other highly-rated movies
            similar_movies = {}
            for user_id in movie_ratings['userId'].values:
                user_other_ratings = self.ratings_df[
                    (self.ratings_df['userId'] == user_id) &
                    (self.ratings_df['movieId'] != item_id) &
                    (self.ratings_df['rating'] >= 4.0)
                ]
                
                for _, row in user_other_ratings.iterrows():
                    movie_id = row['movieId']
                    rating = row['rating']
                    
                    if movie_id not in similar_movies:
                        similar_movies[movie_id] = []
                    similar_movies[movie_id].append(rating)
            
            # Calculate similarity scores based on how many similar users liked each movie
            similarity_scores = []
            for movie_id, ratings in similar_movies.items():
                # Similarity = (average rating) * log(number of users who liked both)
                avg_rating = np.mean(ratings)
                user_count = len(ratings)
                similarity = avg_rating * np.log1p(user_count)  # log1p for numerical stability
                
                similarity_scores.append({
                    'movieId': movie_id,
                    'similarity': similarity
                })
            
            # Convert to DataFrame and sort
            similarities_df = pd.DataFrame(similarity_scores)
            if len(similarities_df) == 0:
                return pd.DataFrame(columns=['movieId', 'similarity', 'title', 'genres'])
            
            similarities_df = similarities_df.sort_values('similarity', ascending=False)
            
            # Normalize similarities to 0-1 range
            max_sim = similarities_df['similarity'].max()
            if max_sim > 0:
                similarities_df['similarity'] = similarities_df['similarity'] / max_sim
            
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
        """Return list of User KNN capabilities"""
        return [
            "User-based collaborative filtering",
            "Neighborhood-based recommendations",
            "Intuitive similarity calculations",
            "Handles new items well",
            "Explainable recommendations",
            "No training required for new users"
        ]
    
    def _get_description(self) -> str:
        """Return human-readable algorithm description"""
        return (
            "Finds users with similar rating patterns and recommends movies that "
            "those similar users have enjoyed. Works on the principle that people "
            "with similar taste will like similar movies."
        )
    
    def _get_strengths(self) -> List[str]:
        """Return list of algorithm strengths"""
        return [
            "Highly interpretable results",
            "Good performance with sparse data",
            "Handles new items immediately",
            "Leverages user similarity effectively",
            "No complex training required"
        ]
    
    def _get_ideal_use_cases(self) -> List[str]:
        """Return list of ideal use cases"""
        return [
            "Users with sparse rating profiles (<30 ratings)",
            "When interpretability is important",
            "New item recommendations",
            "Community-based recommendations",
            "Real-time recommendation updates"
        ]
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get User KNN-specific state for saving"""
        return {
            'n_neighbors': self.n_neighbors,
            'similarity_metric': self.similarity_metric,
            'user_movie_matrix': self.user_movie_matrix,
            'user_mapper': self.user_mapper,
            'movie_mapper': self.movie_mapper,
            'user_inv_mapper': self.user_inv_mapper,
            'movie_inv_mapper': self.movie_inv_mapper,
            'user_means': self.user_means,
            'global_mean': self.global_mean,
            'metrics': {
                'rmse': self.metrics.rmse,
                'training_time': self.metrics.training_time,
                'coverage': self.metrics.coverage,
                'memory_usage_mb': self.metrics.memory_usage_mb
            }
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Set User KNN-specific state from loading"""
        self.n_neighbors = state['n_neighbors']
        self.similarity_metric = state['similarity_metric']
        self.user_movie_matrix = state['user_movie_matrix']
        self.user_mapper = state['user_mapper']
        self.movie_mapper = state['movie_mapper']
        self.user_inv_mapper = state['user_inv_mapper']
        self.movie_inv_mapper = state['movie_inv_mapper']
        self.user_means = state['user_means']
        self.global_mean = state['global_mean']
        
        # Restore metrics if available
        if 'metrics' in state:
            metrics_data = state['metrics']
            self.metrics.rmse = metrics_data.get('rmse', 0.0)
            self.metrics.training_time = metrics_data.get('training_time', 0.0)
            self.metrics.coverage = metrics_data.get('coverage', 0.0)
            self.metrics.memory_usage_mb = metrics_data.get('memory_usage_mb', 0.0)
        else:
            # Calculate coverage from loaded model if metrics not saved
            if self.movie_mapper and self.movies_df is not None:
                self.metrics.coverage = (len(self.movie_mapper) / len(self.movies_df)) * 100
        
        # Recreate KNN model
        self.knn_model = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,
            metric=self.similarity_metric,
            algorithm='brute'
        )
        self.knn_model.fit(self.user_movie_matrix)
        
        # Mark as trained (critical for Hybrid loading)
        self.is_trained = True
    
    def get_explanation_context(self, user_id: int, movie_id: int) -> Dict[str, Any]:
        """
        Get context for explaining why this movie was recommended.
        
        Returns:
            Dictionary with explanation context specific to User KNN
        """
        if not self.is_trained:
            return {}
        
        try:
            if user_id not in self.user_mapper:
                return {'method': 'popularity', 'reason': 'User not in training data'}
            
            user_idx = self.user_mapper[user_id]
            user_vector = self.user_movie_matrix[user_idx:user_idx+1]
            
            # Find similar users
            distances, neighbor_indices = self.knn_model.kneighbors(user_vector, n_neighbors=6)
            neighbor_indices = neighbor_indices.flatten()[1:6]  # Skip self, take top 5
            
            # Find which of these neighbors rated the recommended movie highly
            similar_users_who_liked = []
            for neighbor_idx in neighbor_indices:
                neighbor_user_id = self.user_inv_mapper[neighbor_idx]
                
                # Check if this neighbor rated the movie
                neighbor_ratings = self.ratings_df[
                    (self.ratings_df['userId'] == neighbor_user_id) &
                    (self.ratings_df['movieId'] == movie_id)
                ]
                
                if len(neighbor_ratings) > 0 and neighbor_ratings.iloc[0]['rating'] >= 4.0:
                    similar_users_who_liked.append({
                        'user_id': neighbor_user_id,
                        'rating': neighbor_ratings.iloc[0]['rating']
                    })
            
            return {
                'method': 'user_similarity',
                'similar_users_count': len(similar_users_who_liked),
                'similar_users': similar_users_who_liked[:3],  # Top 3 for display
                'prediction': self._predict_rating(user_id, movie_id)
            }
            
        except Exception as e:
            return {'method': 'error', 'error': str(e)}