"""
CineMatch V2.1.0 - Item-Based KNN Recommender

K-Nearest Neighbors recommendation using item-based collaborative filtering.
Finds movies with similar rating patterns and recommends them to users.

REFACTORED: Now inherits from KNNBaseRecommender using Template Method pattern.
Eliminates ~125 lines of duplicated code.

Author: CineMatch Development Team
Date: November 12, 2025
Version: 2.1.0
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.algorithms.knn_base import KNNBaseRecommender


class ItemKNNRecommender(KNNBaseRecommender):
    """
    Item-Based K-Nearest Neighbors Recommender.
    
    Finds movies with similar rating patterns and recommends them 
    based on what the user has previously enjoyed.
    
    Inherits common KNN logic from KNNBaseRecommender.
    Implements item-specific matrix creation, similarity precomputation,
    and filtering based on minimum ratings.
    """
    
    def __init__(self, n_neighbors: int = 30, similarity_metric: str = 'cosine', 
                 min_ratings: int = 5, **kwargs):
        """
        Initialize Item KNN recommender.
        
        Args:
            n_neighbors: Number of similar items to consider (default: 30)
            similarity_metric: Similarity metric ('cosine' recommended)
            min_ratings: Minimum ratings per item to include in similarity (default: 5)
            **kwargs: Additional parameters
        """
        # Initialize parent with Template Method pattern
        super().__init__(
            name="KNN Item-Based",
            n_neighbors=n_neighbors,
            similarity_metric=similarity_metric,
            min_ratings=min_ratings,
            **kwargs
        )
        
        # Item-specific parameters
        self.min_ratings = min_ratings
        
        # Item-specific data structures
        self.item_user_matrix = None  # Alias for interaction_matrix
        self.item_means = None  # Mean ratings per item
        self.similarity_matrix = None  # Precomputed similarity matrix (optional)
        self.valid_items = set()  # Items with >= min_ratings
    
    def _get_matrix_description(self) -> str:
        """Return description of interaction matrix for logging"""
        return "item-user matrix"
    
    def _create_interaction_matrix(self, ratings_df: pd.DataFrame) -> None:
        """
        Create sparse item-user matrix for KNN.
        
        This method implements the abstract method from KNNBaseRecommender.
        Creates an item x user matrix and filters items by minimum rating threshold.
        """
        # Filter items with minimum ratings
        item_rating_counts = ratings_df['movieId'].value_counts()
        valid_items = item_rating_counts[item_rating_counts >= self.min_ratings].index
        self.valid_items = set(valid_items)
        
        # Filter ratings to only include valid items
        filtered_ratings = ratings_df[ratings_df['movieId'].isin(valid_items)]
        
        # Create sparse matrix (items x users)
        n_movies = len(filtered_ratings['movieId'].unique())
        n_users = len(filtered_ratings['userId'].unique())
        
        movie_indices = filtered_ratings['movieId'].map(self.movie_mapper).values
        user_indices = filtered_ratings['userId'].map(self.user_mapper).values
        ratings = filtered_ratings['rating'].values
        
        # Create sparse matrix (movies x users)
        self.item_user_matrix = csr_matrix(
            (ratings, (movie_indices, user_indices)),
            shape=(n_movies, n_users)
        )
        
        # Set base class interaction_matrix property
        self.interaction_matrix = self.item_user_matrix
        
        # Calculate item means for mean-centered predictions
        self.item_means = np.array(self.item_user_matrix.sum(axis=1) / 
                                 (self.item_user_matrix > 0).sum(axis=1)).flatten()
        self.item_means = np.nan_to_num(self.item_means)
        
        # Global mean for fallback
        self.global_mean = filtered_ratings['rating'].mean()
    
    def _post_fit_preprocessing(self) -> None:
        """
        Post-fit hook method: precompute similarity matrix.
        
        This optional hook method from KNNBaseRecommender allows us to
        precompute the item similarity matrix after KNN fitting for better performance.
        """
        print("  • Computing item similarity matrix...")
        self._compute_similarity_matrix()
    
    def _compute_similarity_matrix(self) -> None:
        """Precompute item similarity matrix for faster predictions"""
        try:
            if self.item_user_matrix.shape[0] <= 10000:  # Only for reasonable sizes
                print("    • Computing full similarity matrix...")
                # Use cosine similarity for better performance
                self.similarity_matrix = cosine_similarity(self.item_user_matrix)
                # Zero out self-similarities to avoid issues
                np.fill_diagonal(self.similarity_matrix, 0)
            else:
                print("    • Matrix too large, using on-demand similarity computation")
                self.similarity_matrix = None
        except MemoryError:
            print("    • Not enough memory for full similarity matrix, using on-demand computation")
            self.similarity_matrix = None
    
    def _calculate_coverage(self) -> float:
        """
        Override base class method to calculate coverage based on valid items.
        
        For item-based KNN, coverage is the percentage of items that meet
        the minimum rating threshold and can be recommended.
        """
        if len(self.movies_df) == 0:
            return 0.0
        return len(self.valid_items) / len(self.movies_df) * 100
    
    def _get_additional_memory_usage(self) -> float:
        """
        Override base class method to include similarity matrix memory.
        
        Returns additional memory usage in MB beyond the base interaction matrix.
        """
        if self.similarity_matrix is not None:
            return self.similarity_matrix.nbytes / (1024 * 1024)
        return 0.0
    
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
        # Check if movie is in our valid set
        if movie_id not in self.valid_items or movie_id not in self.movie_mapper:
            return self.global_mean
        
        # Check if user exists in our data
        if user_id not in self.user_mapper:
            # For new users, return item average
            movie_idx = self.movie_mapper[movie_id]
            return max(self.item_means[movie_idx], self.global_mean)
        
        movie_idx = self.movie_mapper[movie_id]
        user_idx = self.user_mapper[user_id]
        
        # Get user's ratings for calculating weighted prediction
        user_ratings = {}
        for mid, midx in self.movie_mapper.items():
            rating = self.item_user_matrix[midx, user_idx]
            if rating > 0:
                user_ratings[midx] = rating
        
        if not user_ratings:
            # User has no ratings in our filtered set
            return max(self.item_means[movie_idx], self.global_mean)
        
        # Find similar items to the target movie
        if self.similarity_matrix is not None:
            # Use precomputed similarity matrix
            similarities = self.similarity_matrix[movie_idx]
            # Get indices of most similar items that the user has rated
            similar_indices = []
            for rated_movie_idx, _ in user_ratings.items():
                if similarities[rated_movie_idx] > 0:
                    similar_indices.append((rated_movie_idx, similarities[rated_movie_idx]))
        else:
            # Compute similarity on-demand using KNN
            movie_vector = self.item_user_matrix[movie_idx:movie_idx+1]
            distances, neighbor_indices = self.knn_model.kneighbors(
                movie_vector, n_neighbors=min(self.n_neighbors + 1, self.item_user_matrix.shape[0])
            )
            
            # Convert distances to similarities and filter for rated items
            similar_indices = []
            for i, neighbor_idx in enumerate(neighbor_indices.flatten()[1:]):  # Skip self
                if neighbor_idx in user_ratings:
                    distance = distances.flatten()[i + 1]
                    similarity = 1 / (1 + distance) if distance > 0 else 1.0
                    similar_indices.append((neighbor_idx, similarity))
        
        # Sort by similarity and take top neighbors
        similar_indices.sort(key=lambda x: x[1], reverse=True)
        similar_indices = similar_indices[:self.n_neighbors]
        
        if not similar_indices:
            return max(self.item_means[movie_idx], self.global_mean)
        
        # Calculate weighted prediction
        numerator = 0.0
        denominator = 0.0
        item_mean = self.item_means[movie_idx]
        
        for similar_item_idx, similarity in similar_indices:
            user_rating = user_ratings[similar_item_idx]
            similar_item_mean = self.item_means[similar_item_idx]
            
            # Mean-centered rating
            centered_rating = user_rating - similar_item_mean
            
            numerator += similarity * centered_rating
            denominator += similarity
        
        # Return prediction
        if denominator > 0:
            prediction = item_mean + (numerator / denominator)
            return np.clip(prediction, 0.5, 5.0)
        else:
            return max(item_mean, self.global_mean)
    
    def get_recommendations(
        self,
        user_id: int,
        n: int = 10,
        exclude_rated: bool = True
    ) -> pd.DataFrame:
        """Generate top-N recommendations for a user"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        # Validate user exists (in our data or allow new users)
        user_exists = self.validate_user_exists(user_id)
        
        print(f"\n🎯 Generating {self.name} recommendations for User {user_id}...")
        print(f"  • User exists in training data: {user_exists}")
        
        self._start_prediction_timer()
        
        try:
            # Get user's rated movies if excluding them
            rated_movie_ids = set()
            if exclude_rated and user_exists:
                user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
                rated_movie_ids = set(user_ratings['movieId'].values)
                print(f"  • Excluding {len(rated_movie_ids)} already-rated movies")
            
            # Get candidate movies (only from our valid set)
            candidate_movies = self.valid_items - rated_movie_ids if exclude_rated else self.valid_items
            
            # OPTIMIZATION: For performance, use smart sampling for large candidate sets
            if len(candidate_movies) > 5000:
                # Use popularity-based sampling + random sampling for efficiency
                print(f"  • Large candidate set ({len(candidate_movies)}), using smart sampling...")
                candidate_movies = self._smart_sample_candidates(candidate_movies, max_candidates=5000)
                print(f"  ✓ Sampled down to {len(candidate_movies)} candidates")
            
            print(f"  • Evaluating {len(candidate_movies)} candidate movies...")
            
            # Use optimized batch prediction for Item KNN
            predictions = self._batch_predict_ratings(user_id, candidate_movies)
            
            # Convert to DataFrame and sort
            predictions_df = pd.DataFrame(predictions)
            if len(predictions_df) == 0:
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
        # PERFORMANCE FIX: Use pre-computed popularity to avoid slow .isin() on 32M rows
        
        # Use cached movie stats to avoid recomputing
        if not hasattr(self, '_all_movie_stats'):
            self._all_movie_stats = self.ratings_df.groupby('movieId').agg({
                'rating': ['count', 'mean']
            })
            self._all_movie_stats.columns = ['count', 'mean_rating']
            self._all_movie_stats['popularity'] = self._all_movie_stats['count'] * self._all_movie_stats['mean_rating']
        
        # Filter to candidate movies (filtering pre-computed stats is fast)
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
        """Optimized vectorized prediction for Item KNN"""
        predictions = []
        
        # Get user's rating profile if they exist
        if user_id not in self.user_mapper:
            # New user - use popularity-based recommendations
            return self._get_popularity_predictions(candidate_movies, 1000)
        
        user_idx = self.user_mapper[user_id]
        
        print(f"    → Getting user {user_id}'s rating history...")
        
        # Get user's ratings as a dictionary for fast lookup
        user_ratings = {}
        user_movies = self.item_user_matrix[:, user_idx]
        for movie_idx, rating in zip(user_movies.indices, user_movies.data):
            if rating > 0:
                movie_id = self.movie_inv_mapper[movie_idx]
                user_ratings[movie_idx] = rating
        
        if len(user_ratings) == 0:
            return self._get_popularity_predictions(candidate_movies, 1000)
        
        print(f"    → User has {len(user_ratings)} ratings")
        print(f"    → Generating predictions for {len(candidate_movies)} candidates...")
        
        # Convert candidate movies to indices and limit for performance
        valid_candidates = []
        candidate_indices = []
        
        for movie_id in candidate_movies:
            if movie_id in self.movie_mapper and movie_id in self.valid_items:
                movie_idx = self.movie_mapper[movie_id]
                valid_candidates.append(movie_id)
                candidate_indices.append(movie_idx)
        
        # Limit candidates for performance
        if len(valid_candidates) > 2000:
            # PERFORMANCE FIX: Use cached movie stats instead of filtering 32M rows
            # This prevents system crash/timeout from expensive .isin() operation
            if not hasattr(self, '_movie_rating_counts'):
                self._movie_rating_counts = self.ratings_df.groupby('movieId').size()
            
            # Filter cached stats (fast) instead of raw ratings (slow)
            # Use .loc to ensure we get a Series, not DataFrame
            available_stats = self._movie_rating_counts.loc[self._movie_rating_counts.index.isin(valid_candidates)]
            
            top_candidates = available_stats.nlargest(2000).index.tolist()
            valid_candidates = [mid for mid in valid_candidates if mid in top_candidates]
            candidate_indices = [self.movie_mapper[mid] for mid in valid_candidates]
            
            print(f"    → Optimized to top {len(valid_candidates)} popular candidates")
        
        # Vectorized prediction
        processed = 0
        for movie_id, movie_idx in zip(valid_candidates, candidate_indices):
            try:
                # Get movie vector
                movie_vector = self.item_user_matrix[movie_idx:movie_idx+1]
                
                if movie_vector.nnz == 0:
                    # No ratings for this movie - use global mean
                    pred_rating = self.global_mean
                else:
                    # Find similar items that user has rated
                    numerator = 0.0
                    denominator = 0.0
                    item_mean = self.item_means[movie_idx]
                    
                    # Quick similarity calculation with user's rated movies
                    similarities_found = 0
                    for rated_movie_idx, user_rating in user_ratings.items():
                        if similarities_found >= 30:  # Limit for speed
                            break
                            
                        # Calculate similarity using dot product (faster than KNN for single item)
                        rated_movie_vector = self.item_user_matrix[rated_movie_idx:rated_movie_idx+1]
                        
                        if rated_movie_vector.nnz > 0:
                            # Simple cosine similarity approximation
                            common_users = movie_vector.multiply(rated_movie_vector)
                            similarity = common_users.sum() / (movie_vector.nnz + rated_movie_vector.nnz)
                            
                            if similarity > 0.1:  # Only use meaningful similarities
                                rated_item_mean = self.item_means[rated_movie_idx]
                                centered_rating = user_rating - rated_item_mean
                                
                                numerator += similarity * centered_rating
                                denominator += similarity
                                similarities_found += 1
                    
                    # Calculate prediction
                    if denominator > 0:
                        pred_rating = item_mean + (numerator / denominator)
                        pred_rating = np.clip(pred_rating, 0.5, 5.0)
                    else:
                        # Fallback to item mean
                        pred_rating = max(item_mean, self.global_mean)
                
                predictions.append({
                    'movieId': movie_id,
                    'predicted_rating': float(pred_rating)
                })
                
                processed += 1
                if processed % 500 == 0:
                    print(f"      → Processed {processed}/{len(valid_candidates)} movies")
                
            except Exception as e:
                continue
        
        print(f"    ✓ Generated {len(predictions)} predictions")
        return predictions
    
    def _get_popularity_predictions(self, candidate_movies: set, max_count: int) -> List[Dict]:
        """Get popularity-based predictions for fallback"""
        valid_candidates = candidate_movies.intersection(self.valid_items)
        
        movie_popularity = self.ratings_df[
            self.ratings_df['movieId'].isin(valid_candidates)
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
        
        return predictions
    
    def _get_popular_movies(self, n: int) -> pd.DataFrame:
        """Fallback: return most popular movies from valid set"""
        valid_ratings = self.ratings_df[self.ratings_df['movieId'].isin(self.valid_items)]
        movie_ratings = valid_ratings.groupby('movieId').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        movie_ratings.columns = ['movieId', 'count', 'mean_rating']
        
        # Sort by popularity
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
        """Find movies similar to the given movie using item-based similarity"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        if not self.validate_movie_exists(item_id):
            raise ValueError(f"Movie ID {item_id} not found in dataset")
        
        if item_id not in self.valid_items:
            print(f"⚠️ Movie {item_id} has insufficient ratings for item-based similarity")
            return self._get_similar_by_genre(item_id, n)
        
        print(f"\n🎬 Finding movies similar to Movie {item_id} (Item KNN approach)...")
        self._start_prediction_timer()
        
        try:
            movie_idx = self.movie_mapper[item_id]
            
            if self.similarity_matrix is not None:
                # Use precomputed similarity matrix
                similarities = self.similarity_matrix[movie_idx]
                # Get top similar items
                similar_indices = np.argsort(similarities)[::-1][:n+1]  # +1 to exclude self
                similar_indices = similar_indices[similarities[similar_indices] > 0]  # Only positive similarities
                
                similarity_scores = []
                for idx in similar_indices:
                    if idx != movie_idx:  # Exclude the item itself
                        movie_id = self.movie_inv_mapper[idx]
                        similarity = similarities[idx]
                        similarity_scores.append({
                            'movieId': movie_id,
                            'similarity': float(similarity)
                        })
                        if len(similarity_scores) >= n:
                            break
            else:
                # Use KNN for on-demand similarity computation
                movie_vector = self.item_user_matrix[movie_idx:movie_idx+1]
                distances, neighbor_indices = self.knn_model.kneighbors(
                    movie_vector, n_neighbors=min(n+1, self.item_user_matrix.shape[0])
                )
                
                similarity_scores = []
                for i, neighbor_idx in enumerate(neighbor_indices.flatten()[1:]):  # Skip self
                    distance = distances.flatten()[i+1]
                    similarity = 1 / (1 + distance) if distance > 0 else 1.0
                    movie_id = self.movie_inv_mapper[neighbor_idx]
                    
                    similarity_scores.append({
                        'movieId': movie_id,
                        'similarity': float(similarity)
                    })
            
            # Convert to DataFrame
            if not similarity_scores:
                return self._get_similar_by_genre(item_id, n)
            
            similarities_df = pd.DataFrame(similarity_scores)
            
            # Merge with movie info
            similar_movies = similarities_df.merge(
                self.movies_df[['movieId', 'title', 'genres']],
                on='movieId',
                how='left'
            )
            
            print(f"✓ Found {len(similar_movies)} similar movies")
            return similar_movies
            
        finally:
            self._end_prediction_timer()
    
    def _get_similar_by_genre(self, item_id: int, n: int) -> pd.DataFrame:
        """Fallback: find similar movies by genre when item-based similarity fails"""
        target_movie = self.movies_df[self.movies_df['movieId'] == item_id].iloc[0]
        target_genres = set(target_movie['genres'].split('|'))
        
        # Find movies with overlapping genres
        similar_movies = []
        for _, movie in self.movies_df.iterrows():
            if movie['movieId'] == item_id:
                continue
            
            movie_genres = set(movie['genres'].split('|'))
            overlap = len(target_genres & movie_genres)
            
            if overlap > 0:
                similarity = overlap / len(target_genres | movie_genres)  # Jaccard similarity
                similar_movies.append({
                    'movieId': movie['movieId'],
                    'similarity': similarity,
                    'title': movie['title'],
                    'genres': movie['genres']
                })
        
        # Sort by similarity and return top N
        similar_movies.sort(key=lambda x: x['similarity'], reverse=True)
        return pd.DataFrame(similar_movies[:n])
    
    def _get_capabilities(self) -> List[str]:
        """Return list of Item KNN capabilities"""
        return [
            "Item-based collaborative filtering",
            "Movie similarity analysis",
            "Pattern recognition in ratings",
            "Efficient for users with many ratings",
            "Stable recommendations",
            "Good performance with dense data"
        ]
    
    def _get_description(self) -> str:
        """Return human-readable algorithm description"""
        return (
            "Analyzes movies with similar rating patterns and recommends items "
            "that are similar to what you've enjoyed before. Works by finding "
            "movies that users tend to rate similarly."
        )
    
    def _get_strengths(self) -> List[str]:
        """Return list of algorithm strengths"""
        return [
            "Stable and consistent recommendations",
            "Works well for users with many ratings",
            "Pre-computed similarities for speed",
            "Less susceptible to new user ratings",
            "Good performance with established movies"
        ]
    
    def _get_ideal_use_cases(self) -> List[str]:
        """Return list of ideal use cases"""
        return [
            "Users with 30+ ratings",
            "When recommendation stability is important",
            "Discovering movies similar to favorites",
            "Users who like specific genres/styles",
            "Established movie catalog browsing"
        ]
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Get Item KNN-specific state for saving"""
        return {
            'n_neighbors': self.n_neighbors,
            'similarity_metric': self.similarity_metric,
            'min_ratings': self.min_ratings,
            'item_user_matrix': self.item_user_matrix,
            'user_mapper': self.user_mapper,
            'movie_mapper': self.movie_mapper,
            'movie_inv_mapper': self.movie_inv_mapper,
            'item_means': self.item_means,
            'global_mean': self.global_mean,
            'similarity_matrix': self.similarity_matrix,
            'valid_items': self.valid_items,
            'metrics': {
                'rmse': self.metrics.rmse,
                'training_time': self.metrics.training_time,
                'coverage': self.metrics.coverage,
                'memory_usage_mb': self.metrics.memory_usage_mb
            }
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Set Item KNN-specific state from loading"""
        self.n_neighbors = state['n_neighbors']
        self.similarity_metric = state['similarity_metric']
        self.min_ratings = state['min_ratings']
        self.item_user_matrix = state['item_user_matrix']
        self.user_mapper = state['user_mapper']
        self.movie_mapper = state['movie_mapper']
        self.movie_inv_mapper = state['movie_inv_mapper']
        self.item_means = state['item_means']
        self.global_mean = state['global_mean']
        self.similarity_matrix = state.get('similarity_matrix')
        self.valid_items = state['valid_items']
        
        # Restore metrics if available
        if 'metrics' in state:
            metrics_data = state['metrics']
            self.metrics.rmse = metrics_data.get('rmse', 0.0)
            self.metrics.training_time = metrics_data.get('training_time', 0.0)
            self.metrics.coverage = metrics_data.get('coverage', 0.0)
            self.metrics.memory_usage_mb = metrics_data.get('memory_usage_mb', 0.0)
        else:
            # Calculate coverage from loaded model if metrics not saved
            if self.valid_items and self.movies_df is not None:
                self.metrics.coverage = (len(self.valid_items) / len(self.movies_df)) * 100
        
        # Recreate KNN model
        self.knn_model = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,
            metric=self.similarity_metric,
            algorithm='brute'
        )
        if self.item_user_matrix is not None:
            self.knn_model.fit(self.item_user_matrix)
        
        # Mark as trained (critical for Hybrid loading)
        self.is_trained = True
    
    def get_explanation_context(self, user_id: int, movie_id: int) -> Dict[str, Any]:
        """
        Get context for explaining why this movie was recommended.
        
        Returns:
            Dictionary with explanation context specific to Item KNN
        """
        if not self.is_trained:
            return {}
        
        try:
            if movie_id not in self.valid_items or movie_id not in self.movie_mapper:
                return {'method': 'popularity', 'reason': 'Movie not in item similarity model'}
            
            if user_id not in self.user_mapper:
                return {'method': 'item_average', 'reason': 'User not in training data'}
            
            movie_idx = self.movie_mapper[movie_id]
            user_idx = self.user_mapper[user_id]
            
            # Get user's rated movies that are similar to this recommendation
            user_ratings = {}
            for mid, midx in self.movie_mapper.items():
                rating = self.item_user_matrix[midx, user_idx]
                if rating > 0:
                    user_ratings[midx] = (mid, rating)
            
            if not user_ratings:
                return {'method': 'item_average', 'reason': 'User has no ratings in model'}
            
            # Find similar items that the user has rated
            similar_rated_movies = []
            if self.similarity_matrix is not None:
                similarities = self.similarity_matrix[movie_idx]
                for rated_idx, (rated_movie_id, rating) in user_ratings.items():
                    similarity = similarities[rated_idx]
                    if similarity > 0.1:  # Minimum similarity threshold
                        similar_rated_movies.append({
                            'movie_id': rated_movie_id,
                            'user_rating': rating,
                            'similarity': float(similarity)
                        })
                
                # Sort by similarity and take top 3
                similar_rated_movies.sort(key=lambda x: x['similarity'], reverse=True)
            
            return {
                'method': 'item_similarity',
                'similar_movies_count': len(similar_rated_movies),
                'similar_movies': similar_rated_movies[:3],
                'prediction': self._predict_rating(user_id, movie_id)
            }
            
        except Exception as e:
            return {'method': 'error', 'error': str(e)}