"""
CineMatch V2.1.0 - Content-Based Filtering Recommender

Content-Based Filtering using movie features (genres, tags, titles).
Recommends movies with similar characteristics to those the user has enjoyed.

REFACTORED (Phase 1B): Feature engineering extracted to separate modules.

Features:
- TF-IDF vectorization for genres, tags, and titles (via MovieFeatureExtractor)
- Cosine similarity for movie-movie similarity (via SimilarityMatrixBuilder)
- User profile building from rating history (via UserProfileBuilder)
- Cold-start handling for new users
- Memory-efficient sparse matrix operations

Author: CineMatch Development Team
Date: November 12, 2025
Version: 2.1.0
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set
import pandas as pd
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.algorithms.base_recommender import BaseRecommender
from src.algorithms.feature_engineering import (
    MovieFeatureExtractor,
    SimilarityMatrixBuilder,
    UserProfileBuilder
)


class ContentBasedRecommender(BaseRecommender):
    """
    Content-Based Filtering Recommender.
    
    Uses movie features (genres, tags, titles) to find movies similar to
    those the user has rated highly. Builds a user profile based on their
    rating history and recommends movies with similar feature profiles.
    
    Algorithm Flow:
    1. Extract features from movies (genres, tags, titles)
    2. Create TF-IDF feature vectors for each movie
    3. Compute movie-movie similarity matrix
    4. Build user profile from rated movies
    5. Recommend movies similar to user's profile
    """
    
    def __init__(
        self, 
        genre_weight: float = 0.5,
        tag_weight: float = 0.3, 
        title_weight: float = 0.2,
        min_similarity: float = 0.01,
        **kwargs
    ):
        """
        Initialize Content-Based recommender.
        
        Args:
            genre_weight: Weight for genre features (default: 0.5)
            tag_weight: Weight for tag features (default: 0.3)
            title_weight: Weight for title features (default: 0.2)
            min_similarity: Minimum similarity threshold (default: 0.01)
            **kwargs: Additional parameters
        """
        super().__init__(
            "Content-Based Filtering",
            genre_weight=genre_weight,
            tag_weight=tag_weight,
            title_weight=title_weight,
            min_similarity=min_similarity,
            **kwargs
        )
        
        self.genre_weight = genre_weight
        self.tag_weight = tag_weight
        self.title_weight = title_weight
        self.min_similarity = min_similarity
        
        # Feature Engineering Components (Phase 1B Refactoring)
        self.feature_extractor = MovieFeatureExtractor(
            genre_weight=genre_weight,
            tag_weight=tag_weight,
            title_weight=title_weight
        )
        self.similarity_builder = SimilarityMatrixBuilder(
            min_similarity=min_similarity
        )
        self.profile_builder = UserProfileBuilder()
        
        # Convenience references to feature matrices (for backward compatibility)
        self.combined_features = None
        self.similarity_matrix = None
        self.movie_mapper = {}
        self.movie_inv_mapper = {}
        
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        """Train the Content-Based model on the provided data"""
        print(f"\n🎬 Training {self.name}...")
        start_time = time.time()
        
        # Store data references
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()
        
        # Add genres_list column if not present
        if 'genres_list' not in self.movies_df.columns:
            self.movies_df['genres_list'] = self.movies_df['genres'].str.split('|')
        
        # Load tags data (via feature extractor)
        print("  • Loading tags data...")
        self.movies_df = self.feature_extractor.load_and_preprocess_tags(self.movies_df)
        
        # Build feature matrices (via feature extractor)
        print("  • Extracting movie features...")
        self.combined_features, self.movie_mapper, self.movie_inv_mapper = \
            self.feature_extractor.build_feature_matrix(self.movies_df)
        
        # Compute similarity matrix (via similarity builder)
        print("  • Computing movie-movie similarity matrix...")
        self.similarity_matrix = self.similarity_builder.compute_similarity_matrix(
            self.combined_features
        )
        
        # Calculate metrics
        training_time = time.time() - start_time
        self.metrics.training_time = training_time
        self.is_trained = True
        
        # Calculate RMSE on a test sample
        print("  • Calculating RMSE...")
        self._calculate_rmse(ratings_df)
        
        # Calculate coverage
        self.metrics.coverage = 100.0  # Can recommend any movie with features
        
        # Calculate memory usage
        matrix_size_mb = self._calculate_memory_usage()
        self.metrics.memory_usage_mb = matrix_size_mb
        
        print(f"✓ {self.name} trained successfully!")
        print(f"  • Training time: {training_time:.1f}s")
        print(f"  • RMSE: {self.metrics.rmse:.4f}")
        print(f"  • Feature dimensions: {self.combined_features.shape}")
        if self.similarity_matrix is not None:
            print(f"  • Similarity matrix size: {self.similarity_matrix.shape}")
        else:
            print(f"  • Similarity computation: On-demand (memory-optimized)")
        print(f"  • Memory usage: {matrix_size_mb:.1f} MB")
        print(f"  • Coverage: {self.metrics.coverage:.1f}%")
    
    # OLD METHODS REMOVED (Phase 1B Refactoring):
    # - _load_tags_data() -> Moved to MovieFeatureExtractor.load_and_preprocess_tags()
    # - _build_feature_matrix() -> Moved to MovieFeatureExtractor.build_feature_matrix()
    # - _compute_similarity_matrix() -> Moved to SimilarityMatrixBuilder.compute_similarity_matrix()
    # - _build_user_profile() -> Moved to UserProfileBuilder.build_profile()
    
    def _build_user_profile(self, user_id: int) -> Optional[np.ndarray]:
        """
        Build user profile from their rating history.
        
        Uses UserProfileBuilder for profile creation and caching.
        
        Args:
            user_id: User ID
            
        Returns:
            User profile vector (weighted average of rated movie features)
        """
        return self.profile_builder.build_profile(
            user_id=user_id,
            ratings_df=self.ratings_df,
            feature_matrix=self.combined_features,
            movie_mapper=self.movie_mapper
        )
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for a specific user-movie pair.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating (1.0-5.0)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        self._start_prediction_timer()
        
        try:
            # Build user profile
            user_profile = self._build_user_profile(user_id)
            
            if user_profile is None:
                # New user: return global mean
                return self.ratings_df['rating'].mean()
            
            # Get movie features
            if movie_id not in self.movie_mapper:
                # Movie not in dataset: return user's mean rating
                user_mean = self.ratings_df[
                    self.ratings_df['userId'] == user_id
                ]['rating'].mean()
                return user_mean if not pd.isna(user_mean) else 3.5
            
            movie_idx = self.movie_mapper[movie_id]
            movie_features = self.combined_features[movie_idx].toarray().flatten()
            
            # Compute similarity between user profile and movie
            similarity = np.dot(user_profile, movie_features)
            
            # Convert similarity to rating (scale to 1-5)
            # Similarity is in [0, 1], map to rating range
            user_mean = self.ratings_df[
                self.ratings_df['userId'] == user_id
            ]['rating'].mean()
            
            # Predicted rating: user_mean + similarity_boost
            # Higher similarity = higher rating boost
            predicted_rating = user_mean + (similarity - 0.5) * 4
            
            # Clamp to [1, 5]
            predicted_rating = np.clip(predicted_rating, 1.0, 5.0)
            
            return float(predicted_rating)
            
        finally:
            self._end_prediction_timer()
    
    def get_recommendations(
        self,
        user_id: int,
        n: int = 10,
        exclude_rated: bool = True
    ) -> pd.DataFrame:
        """
        Generate top-N recommendations for a user.
        
        Args:
            user_id: User ID
            n: Number of recommendations
            exclude_rated: Exclude movies user has rated
            
        Returns:
            DataFrame with recommendations
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        print(f"\n🎬 Generating {self.name} recommendations for User {user_id}...")
        self._start_prediction_timer()
        
        try:
            # Build user profile
            user_profile = self._build_user_profile(user_id)
            
            if user_profile is None:
                # New user: return popular movies
                print(f"  • User {user_id} has no ratings - generating popular recommendations...")
                return self._get_popular_movies(n)
            
            # Get user's rated movies
            user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
            rated_movie_ids = set(user_ratings['movieId'].values)
            
            print(f"  • User has rated {len(rated_movie_ids)} movies")
            
            # Get candidate movies
            if exclude_rated:
                candidate_movie_ids = [
                    mid for mid in self.movie_mapper.keys()
                    if mid not in rated_movie_ids
                ]
            else:
                candidate_movie_ids = list(self.movie_mapper.keys())
            
            print(f"  • Evaluating {len(candidate_movie_ids)} candidate movies...")
            
            # Compute similarity scores for all candidates
            candidate_indices = [self.movie_mapper[mid] for mid in candidate_movie_ids]
            candidate_features = self.combined_features[candidate_indices]
            
            # Compute similarity with user profile (vectorized)
            similarities = candidate_features @ user_profile
            
            # Get user's mean rating for scaling
            user_mean = user_ratings['rating'].mean()
            
            # Convert similarities to predicted ratings
            predicted_ratings = user_mean + (similarities - 0.5) * 4
            predicted_ratings = np.clip(predicted_ratings, 1.0, 5.0)
            
            # Create DataFrame with predictions
            predictions = pd.DataFrame({
                'movieId': candidate_movie_ids,
                'predicted_rating': predicted_ratings,
                'similarity': similarities
            })
            
            # Sort by predicted rating
            predictions = predictions.sort_values('predicted_rating', ascending=False)
            
            # Get top N
            top_predictions = predictions.head(n)
            
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
    
    def get_similar_items(self, item_id: int, n: int = 10) -> pd.DataFrame:
        """
        Find movies similar to the given movie.
        
        Args:
            item_id: Movie ID
            n: Number of similar movies
            
        Returns:
            DataFrame with similar movies
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")
        
        if item_id not in self.movie_mapper:
            raise ValueError(f"Movie ID {item_id} not found in dataset")
        
        print(f"\n🔍 Finding movies similar to Movie {item_id}...")
        self._start_prediction_timer()
        
        try:
            # Get movie index
            movie_idx = self.movie_mapper[item_id]
            
            # Get movie features
            movie_features = self.combined_features[movie_idx]
            
            # Compute similarities on-demand or use pre-computed matrix
            if self.similarity_matrix is not None:
                # Use pre-computed matrix (small dataset)
                similarity_scores = self.similarity_matrix[movie_idx].toarray().flatten()
            else:
                # Compute on-demand (large dataset)
                # Since features are normalized, dot product = cosine similarity
                similarity_scores = self.combined_features.dot(movie_features.T).toarray().flatten()
            
            # Get top N similar movies (excluding itself)
            # Set self-similarity to 0
            similarity_scores[movie_idx] = 0
            
            # Get indices of top N
            top_indices = np.argsort(similarity_scores)[::-1][:n]
            top_scores = similarity_scores[top_indices]
            
            # Convert indices to movie IDs
            similar_movie_ids = [self.movie_inv_mapper[idx] for idx in top_indices]
            
            # Create DataFrame
            similar_movies = pd.DataFrame({
                'movieId': similar_movie_ids,
                'similarity': top_scores
            })
            
            # Merge with movie info
            similar_movies = similar_movies.merge(
                self.movies_df[['movieId', 'title', 'genres', 'genres_list']],
                on='movieId',
                how='left'
            )
            
            print(f"✓ Found {len(similar_movies)} similar movies")
            
            return similar_movies
            
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
        recommendations['similarity'] = 1.0
        
        print(f"✓ Generated {len(recommendations)} popular movie recommendations")
        return recommendations[['movieId', 'predicted_rating', 'similarity', 'title', 'genres', 'genres_list']]
    
    def _calculate_rmse(self, ratings_df: pd.DataFrame) -> None:
        """Calculate RMSE on a test sample"""
        test_sample = ratings_df.sample(min(5000, len(ratings_df)), random_state=42)
        
        squared_errors = []
        for _, row in test_sample.iterrows():
            try:
                pred = self.predict(row['userId'], row['movieId'])
                squared_errors.append((pred - row['rating']) ** 2)
            except:
                continue
        
        if squared_errors:
            self.metrics.rmse = np.sqrt(np.mean(squared_errors))
        else:
            self.metrics.rmse = 0.0
    
    def _calculate_memory_usage(self) -> float:
        """Calculate approximate memory usage in MB"""
        total_bytes = 0
        
        # Feature matrices
        if self.combined_features is not None:
            total_bytes += self.combined_features.data.nbytes
            total_bytes += self.combined_features.indices.nbytes
            total_bytes += self.combined_features.indptr.nbytes
        
        # Similarity matrix (via similarity_builder)
        total_mb = total_bytes / (1024 * 1024)
        total_mb += self.similarity_builder.get_memory_usage()
        
        return total_mb
    
    def get_feature_importance(self, movie_id: int) -> Dict[str, Any]:
        """
        Get feature importance for a specific movie.
        
        Args:
            movie_id: Movie ID
            
        Returns:
            Dictionary with feature analysis
        """
        if movie_id not in self.movie_mapper:
            return {'error': 'Movie not found'}
        
        movie_idx = self.movie_mapper[movie_id]
        
        # Get movie info
        movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
        
        # Get feature vectors (from feature_extractor)
        genre_vec = self.feature_extractor.genre_features[movie_idx].toarray().flatten()
        tag_vec = self.feature_extractor.tag_features[movie_idx].toarray().flatten()
        title_vec = self.feature_extractor.title_features[movie_idx].toarray().flatten()
        
        # Get feature names (from feature_extractor vectorizers)
        genre_features = self.feature_extractor.genre_vectorizer.get_feature_names_out() if hasattr(
            self.feature_extractor.genre_vectorizer, 'get_feature_names_out'
        ) else []
        
        tag_features = self.feature_extractor.tag_vectorizer.get_feature_names_out() if hasattr(
            self.feature_extractor.tag_vectorizer, 'get_feature_names_out'
        ) and len(tag_vec) > 0 else []
        
        title_features = self.feature_extractor.title_vectorizer.get_feature_names_out() if hasattr(
            self.feature_extractor.title_vectorizer, 'get_feature_names_out'
        ) else []
        
        # Get top features for each type
        top_genres = []
        if len(genre_features) > 0 and len(genre_vec) > 0:
            top_genre_indices = np.argsort(genre_vec)[::-1][:5]
            top_genres = [
                (genre_features[i], float(genre_vec[i])) 
                for i in top_genre_indices if genre_vec[i] > 0
            ]
        
        top_tags = []
        if len(tag_features) > 0 and len(tag_vec) > 0:
            top_tag_indices = np.argsort(tag_vec)[::-1][:5]
            top_tags = [
                (tag_features[i], float(tag_vec[i])) 
                for i in top_tag_indices if tag_vec[i] > 0
            ]
        
        top_title_words = []
        if len(title_features) > 0 and len(title_vec) > 0:
            top_title_indices = np.argsort(title_vec)[::-1][:5]
            top_title_words = [
                (title_features[i], float(title_vec[i])) 
                for i in top_title_indices if title_vec[i] > 0
            ]
        
        return {
            'movie_id': movie_id,
            'title': movie_info['title'],
            'genres': movie_info['genres'],
            'top_genre_features': top_genres,
            'top_tag_features': top_tags,
            'top_title_features': top_title_words,
            'genre_weight': self.genre_weight,
            'tag_weight': self.tag_weight,
            'title_weight': self.title_weight
        }
    
    def explain_recommendation(
        self, 
        user_id: int, 
        movie_id: int
    ) -> Dict[str, Any]:
        """
        Explain why a movie was recommended to a user.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Dictionary with explanation details
        """
        # Build user profile
        user_profile = self._build_user_profile(user_id)
        
        if user_profile is None:
            return {'explanation': 'User has no rating history'}
        
        # Get movie features
        if movie_id not in self.movie_mapper:
            return {'explanation': 'Movie not found'}
        
        movie_idx = self.movie_mapper[movie_id]
        movie_features = self.combined_features[movie_idx].toarray().flatten()
        
        # Compute similarity
        similarity = float(np.dot(user_profile, movie_features))
        
        # Get user's top-rated movies
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        top_rated = user_ratings.nlargest(5, 'rating')
        
        top_rated_movies = top_rated.merge(
            self.movies_df[['movieId', 'title', 'genres']],
            on='movieId',
            how='left'
        )
        
        # Get recommended movie info
        rec_movie = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
        
        return {
            'similarity_score': similarity,
            'recommended_movie': {
                'id': movie_id,
                'title': rec_movie['title'],
                'genres': rec_movie['genres']
            },
            'based_on_movies': top_rated_movies[['title', 'genres', 'rating']].to_dict('records'),
            'explanation': f"This movie is recommended because it shares similar features "
                          f"(genres, tags, themes) with movies you've rated highly. "
                          f"Similarity score: {similarity:.3f}"
        }
    
    def load_model(self, model_path: Path) -> None:
        """
        Load pre-trained model from disk.
        
        Args:
            model_path: Path to the saved model file
        """
        import pickle
        
        print(f"📂 Loading Content-Based model from {model_path}")
        start_time = time.time()
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Extract model
        if isinstance(model_data, dict) and 'model' in model_data:
            loaded_model = model_data['model']
            metadata = model_data.get('metadata', {})
            metrics = model_data.get('metrics', {})
            
            print(f"   • Model version: {metadata.get('version', 'Unknown')}")
            print(f"   • Trained on: {metadata.get('trained_on', 'Unknown')}")
            print(f"   • Number of movies: {metadata.get('n_movies', 'Unknown')}")
        else:
            loaded_model = model_data
        
        # Copy all attributes from loaded model
        self.__dict__.update(loaded_model.__dict__)
        
        # Ensure is_trained flag is set
        self.is_trained = True
        
        load_time = time.time() - start_time
        
        print(f"   ✓ Model loaded successfully in {load_time:.2f}s")
        print(f"   ✓ Feature dimensions: {self.combined_features.shape}")
        if self.similarity_matrix is not None:
            print(f"   ✓ Similarity matrix: {self.similarity_matrix.shape}")
        else:
            print(f"   ✓ Similarity computation: On-demand (memory-optimized)")
    
    def save_model(self, model_path: Path) -> None:
        """
        Save trained model to disk.
        
        Args:
            model_path: Path to save the model file
        """
        import pickle
        
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        # Create parent directory if it doesn't exist
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare model data
        model_data = {
            'model': self,
            'metrics': {
                'rmse': self.metrics.rmse,
                'training_time': self.metrics.training_time,
                'coverage': self.metrics.coverage,
                'memory_mb': self.metrics.memory_usage_mb
            },
            'metadata': {
                'trained_on': time.strftime("%Y-%m-%d %H:%M:%S"),
                'version': '2.1.0',
                'n_movies': len(self.movie_mapper),
                'feature_dimensions': self.combined_features.shape,
                'similarity_matrix_shape': self.similarity_matrix.shape if self.similarity_matrix is not None else None,
                'on_demand_computation': self.similarity_matrix is None,
                'params': self.params
            }
        }
        
        print(f"💾 Saving Content-Based model to {model_path}")
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"   ✓ Model saved successfully ({file_size_mb:.1f} MB)")
    
    # ==================== ABSTRACT METHOD IMPLEMENTATIONS ====================
    
    def _get_capabilities(self) -> List[str]:
        """Return list of algorithm capabilities."""
        return [
            "Content-based filtering",
            "Feature similarity",
            "Cold-start for new items",
            "Explainable recommendations",
            "Genre-based matching",
            "Tag-based matching",
            "Title-based matching"
        ]
    
    def _get_description(self) -> str:
        """Return algorithm description."""
        return ("Content-Based Filtering using TF-IDF feature extraction. "
                "Recommends movies with similar characteristics (genres, tags, titles) "
                "to those the user has rated highly. No dependency on other users' ratings.")
    
    def _get_strengths(self) -> List[str]:
        """Return algorithm strengths."""
        return [
            "No cold-start problem for new items",
            "Highly explainable recommendations",
            "Independent of other users",
            "Can recommend niche items",
            "Privacy-friendly (user-specific)"
        ]
    
    def _get_ideal_use_cases(self) -> List[str]:
        """Return ideal use cases."""
        return [
            "New movie recommendations",
            "Genre exploration",
            "Discovering similar movies",
            "Privacy-sensitive applications",
            "Small user base scenarios"
        ]
    
    def _get_model_state(self) -> Dict[str, Any]:
        """Return current model state for serialization."""
        if not self.is_trained:
            return {}
        
        return {
            # Feature engineering modules (Phase 1B)
            'feature_extractor': self.feature_extractor,
            'similarity_builder': self.similarity_builder,
            'profile_builder': self.profile_builder,
            # Convenience references
            'combined_features': self.combined_features,
            'similarity_matrix': self.similarity_matrix,
            'movie_mapper': self.movie_mapper,
            'movie_inv_mapper': self.movie_inv_mapper,
            'params': self.params,
            'metrics': {
                'rmse': self.metrics.rmse if hasattr(self, 'metrics') else None,
                'training_time': self.metrics.training_time if hasattr(self, 'metrics') else None,
                'coverage': self.metrics.coverage if hasattr(self, 'metrics') else None,
                'memory_mb': self.metrics.memory_usage_mb if hasattr(self, 'metrics') else None
            }
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        """Restore model state from serialized data."""
        if not state:
            return
        
        # Restore feature engineering modules (Phase 1B)
        self.feature_extractor = state.get('feature_extractor')
        self.similarity_builder = state.get('similarity_builder')
        self.profile_builder = state.get('profile_builder')
        
        # Restore convenience references
        self.combined_features = state.get('combined_features')
        self.similarity_matrix = state.get('similarity_matrix')
        self.movie_mapper = state.get('movie_mapper', {})
        self.movie_inv_mapper = state.get('movie_inv_mapper', {})
        self.params = state.get('params', self.params)
        
        # Restore metrics
        metrics_data = state.get('metrics', {})
        if metrics_data:
            self.metrics.rmse = metrics_data.get('rmse', 0.0)
            self.metrics.training_time = metrics_data.get('training_time', 0.0)
            self.metrics.coverage = metrics_data.get('coverage', 100.0)  # Default to 100% for content-based
            self.metrics.memory_usage_mb = metrics_data.get('memory_mb', 0.0)
        else:
            # If no metrics saved, calculate them now
            self.metrics.coverage = 100.0
            if self.combined_features is not None:
                # Calculate memory usage
                total_bytes = 0
                total_bytes += self.combined_features.data.nbytes
                total_bytes += self.combined_features.indices.nbytes
                total_bytes += self.combined_features.indptr.nbytes
                if self.similarity_matrix is not None:
                    total_bytes += self.similarity_matrix.data.nbytes
                    total_bytes += self.similarity_matrix.indices.nbytes
                    total_bytes += self.similarity_matrix.indptr.nbytes
                self.metrics.memory_usage_mb = total_bytes / (1024 * 1024)
        
        self.is_trained = True
    
    def __getstate__(self):
        """Prepare object for pickling.
        
        Phase 1B: Feature engineering modules (MovieFeatureExtractor, SimilarityMatrixBuilder,
        UserProfileBuilder) are now picklable as proper classes. No special handling needed.
        """
        state = self.__dict__.copy()
        return state
    
    def __setstate__(self, state):
        """Restore object from pickled state.
        
        Phase 1B: Feature engineering modules are restored directly from pickle.
        """
        # Restore basic state
        self.__dict__.update(state)
        
        # Mark as trained (critical for Hybrid loading)
        self.is_trained = True
