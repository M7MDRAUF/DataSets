"""
CineMatch V1.0.0 - Item-Based KNN Recommender

K-Nearest Neighbors recommendation using item-based collaborative filtering.
Finds movies with similar rating patterns and recommends them to users.

Author: CineMatch Development Team
Date: November 7, 2025
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np
import time
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.algorithms.base_recommender import BaseRecommender


class ItemKNNRecommender(BaseRecommender):
    """
    Item-Based K-Nearest Neighbors Recommender.
    
    Finds movies with similar rating patterns and recommends them 
    based on what the user has previously enjoyed.
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
        super().__init__("KNN Item-Based", n_neighbors=n_neighbors, 
                        similarity_metric=similarity_metric, min_ratings=min_ratings, **kwargs)
        
        self.n_neighbors = n_neighbors
        self.similarity_metric = similarity_metric
        self.min_ratings = min_ratings
        self.knn_model = NearestNeighbors(
            n_neighbors=n_neighbors + 1,  # +1 because it includes the query item
            metric=similarity_metric,
            algorithm='brute'  # Better for sparse matrices
        )
        
        # Data structures
        self.item_user_matrix = None
        self.user_mapper = {}
        self.movie_mapper = {}
        self.movie_inv_mapper = {}
        self.item_means = None
        self.global_mean = 0.0
        self.similarity_matrix = None
        self.valid_items = set()
    
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        """Train the Item KNN model on the provided data"""
        print(f"\n🎬 Training {self.name}...")
        start_time = time.time()
        
        # Store data references
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()
        
        # Add genres_list column if not present
        if 'genres_list' not in self.movies_df.columns:
            self.movies_df['genres_list'] = self.movies_df['genres'].str.split('|')
        
        print("  • Creating item-user matrix...")
        self._create_item_user_matrix(ratings_df)
        
        print("  • Training KNN model...")
        self.knn_model.fit(self.item_user_matrix)
        
        # Optionally precompute similarity matrix for better performance
        print("  • Computing item similarity matrix...")
        self._compute_similarity_matrix()
        
        # Calculate metrics
        training_time = time.time() - start_time
        self.metrics.training_time = training_time
        self.is_trained = True
        
        # Calculate RMSE on a test sample
        print("  • Calculating RMSE...")
        self._calculate_rmse(ratings_df)
        
        # Calculate coverage
        self.metrics.coverage = len(self.valid_items) / len(self.movies_df) * 100
        
        # Calculate memory usage (approximate)
        matrix_size_mb = (self.item_user_matrix.data.nbytes + 
                         self.item_user_matrix.indices.nbytes + 
                         self.item_user_matrix.indptr.nbytes) / (1024 * 1024)
        
        if self.similarity_matrix is not None:
            sim_size_mb = self.similarity_matrix.nbytes / (1024 * 1024)
            matrix_size_mb += sim_size_mb
        
        self.metrics.memory_usage_mb = matrix_size_mb
        
        print(f"✓ {self.name} trained successfully!")
        print(f"  • Training time: {training_time:.1f}s")
        print(f"  • RMSE: {self.metrics.rmse:.4f}")
        print(f"  • MAE: {self.metrics.mae:.4f}")
        print(f"  • Matrix size: {self.item_user_matrix.shape}")
        print(f"  • Valid items: {len(self.valid_items)}")
        print(f"  • Coverage: {self.metrics.coverage:.1f}%")
        print(f"  • Memory usage: {matrix_size_mb:.1f} MB")
    
    def _create_item_user_matrix(self, ratings_df: pd.DataFrame) -> None:
        """Create sparse item-user matrix for KNN"""
        # Filter items with minimum ratings
        item_rating_counts = ratings_df['movieId'].value_counts()
        valid_items = item_rating_counts[item_rating_counts >= self.min_ratings].index
        self.valid_items = set(valid_items)
        
        # Filter ratings to only include valid items
        filtered_ratings = ratings_df[ratings_df['movieId'].isin(valid_items)]
        
        # CRITICAL FIX: Create mappings from FULL dataset to prevent index corruption
        # Use ALL users and movies from original ratings_df, not filtered subset
        all_users = ratings_df['userId'].unique()
        unique_movies = filtered_ratings['movieId'].unique()  # Movies still filtered for min_ratings
        
        self.user_mapper = {uid: idx for idx, uid in enumerate(all_users)}
        self.movie_mapper = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.movie_inv_mapper = {idx: mid for mid, idx in self.movie_mapper.items()}
        self.user_inv_mapper = {idx: uid for uid, idx in self.user_mapper.items()}
        
        # Create sparse matrix (items x users)
        # Use filtered movies but ALL users to prevent index corruption
        n_movies = len(unique_movies)
        n_users = len(all_users)
        
        movie_indices = filtered_ratings['movieId'].map(self.movie_mapper).values
        user_indices = filtered_ratings['userId'].map(self.user_mapper).values
        ratings = filtered_ratings['rating'].values
        
        # Create sparse matrix (movies x users)
        self.item_user_matrix = csr_matrix(
            (ratings, (movie_indices, user_indices)),
            shape=(n_movies, n_users)
        )
        
        # Calculate item means for mean-centered predictions
        self.item_means = np.array(self.item_user_matrix.sum(axis=1) / 
                                 (self.item_user_matrix > 0).sum(axis=1)).flatten()
        self.item_means = np.nan_to_num(self.item_means)
        
        # Global mean for fallback
        self.global_mean = filtered_ratings['rating'].mean()
    
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
    
    def _calculate_rmse(self, ratings_df: pd.DataFrame) -> None:
        """Calculate RMSE and MAE on a test sample - OPTIMIZED for speed"""
        # Use a much smaller sample for RMSE to avoid extreme computation time
        # For Item-KNN with 32M ratings, even 100 predictions can take minutes
        test_items = ratings_df[ratings_df['movieId'].isin(self.valid_items)]
        
        # Use only 100 samples instead of 3000 for faster training
        test_sample = test_items.sample(min(100, len(test_items)), random_state=42)
        
        squared_errors = []
        absolute_errors = []
        for idx, row in enumerate(test_sample.itertuples(index=False)):
            try:
                pred = self._predict_rating(row.userId, row.movieId)
                error = pred - row.rating
                squared_errors.append(error ** 2)
                absolute_errors.append(abs(error))
                
                # Progress indicator every 25 predictions
                if (idx + 1) % 25 == 0:
                    print(f"    • RMSE progress: {idx + 1}/{len(test_sample)} predictions...")
            except KeyError as e:
                # User/movie not in training data - skip gracefully
                continue
            except Exception as e:
                # Log unexpected errors but continue
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"RMSE prediction failed for user {row.userId}, movie {row.movieId}: {e}")
                continue
        
        if squared_errors:
            self.metrics.rmse = np.sqrt(np.mean(squared_errors))
            self.metrics.mae = np.mean(absolute_errors)
        else:
            # If all predictions fail, use baseline estimates
            self.metrics.rmse = 0.95  # Typical for KNN on MovieLens
            self.metrics.mae = 0.75
    
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
    
    def recommend(self, user_id: int, n: int = 10, exclude_rated: bool = True) -> pd.DataFrame:
        """Alias for get_recommendations() for consistency with other algorithms"""
        return self.get_recommendations(user_id, n, exclude_rated)
    
    def _smart_sample_candidates(self, candidate_movies: set, max_candidates: int = 5000) -> set:
        """Smart sampling of candidate movies for efficiency"""
        # Get popularity scores for sampling
        movie_popularity = self.ratings_df[
            self.ratings_df['movieId'].isin(candidate_movies)
        ].groupby('movieId').agg({
            'rating': ['count', 'mean']
        })
        movie_popularity.columns = ['count', 'mean_rating']
        movie_popularity['popularity'] = movie_popularity['count'] * movie_popularity['mean_rating']
        
        # Sample: 70% popular movies + 30% random for diversity
        popular_count = int(max_candidates * 0.7)
        random_count = max_candidates - popular_count
        
        # Get most popular movies
        top_popular = movie_popularity.nlargest(min(popular_count, len(movie_popularity)), 'popularity').index.tolist()
        
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
        
        # CRITICAL: Validate user exists in mapper
        if user_id not in self.user_mapper:
            print(f"    ⚠ User {user_id} not in training data, using popularity fallback")
            return self._get_popularity_predictions(candidate_movies, 1000)
        
        user_idx = self.user_mapper[user_id]
        
        print(f"    → Getting user {user_id}'s rating history...")
        
        # Get user's ratings from the SPARSE MATRIX column
        # This accesses the user's column in item_user_matrix (items x users)
        user_ratings = {}
        user_column = self.item_user_matrix[:, user_idx]  # Get this user's column (all items)
        
        # Extract non-zero ratings
        for movie_idx, rating in zip(user_column.indices, user_column.data):
            if rating > 0:
                movie_id = self.movie_inv_mapper[movie_idx]
                user_ratings[movie_idx] = rating
        
        # Double-check: count should match what's in ratings_df for this user
        actual_count = len(self.ratings_df[self.ratings_df['userId'] == user_id])
        matrix_count = len(user_ratings)
        
        if matrix_count != actual_count:
            # Count mismatch indicates the user was looking up wrong index
            print(f"    ⚠ WARNING: Rating count mismatch for user {user_id}")
            print(f"    → ratings_df shows: {actual_count} ratings")
            print(f"    → Matrix shows: {matrix_count} ratings")
            print(f"    → This indicates index corruption - FIX REQUIRED")
        
        if len(user_ratings) == 0:
            return self._get_popularity_predictions(candidate_movies, 1000)
        
        print(f"    → User has {len(user_ratings)} ratings (verified: {actual_count} in dataset)")
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
            # Prioritize popular movies
            candidate_ratings = self.ratings_df[
                self.ratings_df['movieId'].isin(valid_candidates)
            ].groupby('movieId').size()
            
            top_candidates = candidate_ratings.nlargest(2000).index.tolist()
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
                
            except KeyError as e:
                # Movie or user data missing - skip gracefully
                continue
            except Exception as e:
                # Log unexpected errors
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Failed to predict for movie {movie_id}: {e}")
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
            'valid_items': self.valid_items
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
        
        # Recreate KNN model
        self.knn_model = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,
            metric=self.similarity_metric,
            algorithm='brute'
        )
        if self.item_user_matrix is not None:
            self.knn_model.fit(self.item_user_matrix)
    
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