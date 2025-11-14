"""
CineMatch V2.1.0 - Content-Based Filtering Recommender

Content-Based Filtering using movie features (genres, tags, titles).
Recommends movies with similar characteristics to those the user has enjoyed.

Features:
- TF-IDF vectorization for genres, tags, and titles
- Cosine similarity for movie-movie similarity
- User profile building from rating history
- Cold-start handling for new users
- Memory-efficient sparse matrix operations

Author: CineMatch Development Team
Date: November 11, 2025
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set
import pandas as pd
import numpy as np
import time
import warnings
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import re

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.algorithms.base_recommender import BaseRecommender


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
        
        # Feature extractors
        self.genre_vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            lowercase=False,
            preprocessor=lambda x: x
        )
        self.tag_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.title_vectorizer = TfidfVectorizer(
            max_features=300,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=3
        )
        
        # Feature matrices
        self.genre_features = None
        self.tag_features = None
        self.title_features = None
        self.combined_features = None
        
        # Similarity matrix
        self.similarity_matrix = None
        
        # Movie mapping
        self.movie_mapper = {}
        self.movie_inv_mapper = {}
        
        # User profiles cache
        self.user_profiles = {}
        
        # Tags data
        self.tags_df = None
        
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
        
        # Load tags data
        print("  • Loading tags data...")
        self._load_tags_data()
        
        # Build feature matrices
        print("  • Extracting movie features...")
        self._build_feature_matrix()
        
        # Compute similarity matrix
        print("  • Computing movie-movie similarity matrix...")
        self._compute_similarity_matrix()
        
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
    
    def _load_tags_data(self) -> None:
        """Load and preprocess tags data"""
        try:
            data_path = Path(__file__).parent.parent.parent / 'data' / 'ml-32m'
            tags_path = data_path / 'tags.csv'
            
            if tags_path.exists():
                # Load tags (sample for memory efficiency)
                self.tags_df = pd.read_csv(
                    tags_path,
                    usecols=['userId', 'movieId', 'tag']
                )
                
                # Clean tags: lowercase, remove special chars
                self.tags_df['tag'] = self.tags_df['tag'].str.lower()
                self.tags_df['tag'] = self.tags_df['tag'].str.replace(r'[^a-z0-9\s]', '', regex=True)
                
                # Aggregate tags by movie
                movie_tags = self.tags_df.groupby('movieId')['tag'].apply(
                    lambda x: ' '.join(x.unique())
                ).reset_index()
                movie_tags.columns = ['movieId', 'tags_text']
                
                # Merge with movies
                self.movies_df = self.movies_df.merge(
                    movie_tags,
                    on='movieId',
                    how='left'
                )
                self.movies_df['tags_text'].fillna('', inplace=True)
                
                print(f"    ✓ Loaded {len(self.tags_df)} tags for {len(movie_tags)} movies")
            else:
                print(f"    ⚠ Tags file not found, using genres and titles only")
                self.movies_df['tags_text'] = ''
                
        except Exception as e:
            print(f"    ⚠ Error loading tags: {e}")
            self.movies_df['tags_text'] = ''
    
    def _build_feature_matrix(self) -> None:
        """Build TF-IDF feature matrix from movie metadata"""
        # Create movie index mapping
        self.movie_mapper = {
            mid: idx for idx, mid in enumerate(self.movies_df['movieId'].values)
        }
        self.movie_inv_mapper = {
            idx: mid for mid, idx in self.movie_mapper.items()
        }
        
        # 1. Genre Features (Multi-hot encoding with TF-IDF)
        print("    • Processing genre features...")
        genre_lists = self.movies_df['genres_list'].apply(lambda x: x if isinstance(x, list) else []).tolist()
        self.genre_features = self.genre_vectorizer.fit_transform(genre_lists)
        
        # 2. Tag Features (TF-IDF on aggregated tags)
        print("    • Processing tag features...")
        tags_text = self.movies_df['tags_text'].fillna('').tolist()
        
        # Only fit if we have tags
        if any(tags_text):
            self.tag_features = self.tag_vectorizer.fit_transform(tags_text)
        else:
            # Create empty sparse matrix if no tags
            self.tag_features = csr_matrix((len(self.movies_df), 1))
        
        # 3. Title Features (TF-IDF on title words)
        print("    • Processing title features...")
        titles = self.movies_df['title'].fillna('').tolist()
        
        # Extract title text (remove year in parentheses)
        title_texts = [re.sub(r'\s*\(\d{4}\)\s*', '', title) for title in titles]
        self.title_features = self.title_vectorizer.fit_transform(title_texts)
        
        # 4. Combine features with weights
        print("    • Combining features with weights...")
        
        # Normalize each feature type
        genre_features_norm = normalize(self.genre_features, norm='l2', axis=1)
        tag_features_norm = normalize(self.tag_features, norm='l2', axis=1)
        title_features_norm = normalize(self.title_features, norm='l2', axis=1)
        
        # Apply weights
        genre_features_weighted = genre_features_norm * self.genre_weight
        tag_features_weighted = tag_features_norm * self.tag_weight
        title_features_weighted = title_features_norm * self.title_weight
        
        # Combine horizontally (concatenate features)
        self.combined_features = hstack([
            genre_features_weighted,
            tag_features_weighted,
            title_features_weighted
        ]).tocsr()
        
        print(f"    ✓ Genre features: {self.genre_features.shape[1]} dimensions")
        print(f"    ✓ Tag features: {self.tag_features.shape[1]} dimensions")
        print(f"    ✓ Title features: {self.title_features.shape[1]} dimensions")
        print(f"    ✓ Combined features: {self.combined_features.shape}")
    
    def _compute_similarity_matrix(self) -> None:
        """Compute movie-movie similarity matrix using cosine similarity"""
        n_movies = self.combined_features.shape[0]
        
        print(f"    • Preparing for on-demand similarity computation ({n_movies} movies)")
        
        # For large datasets, we DON'T pre-compute the full similarity matrix
        # Instead, we just normalize features and compute similarities on-demand
        # This saves memory: 87K x 87K matrix = ~60GB, but normalized features = ~300MB
        
        if n_movies <= 5000:
            # Small dataset: pre-compute full similarity matrix
            print(f"    • Small dataset detected - pre-computing full similarity matrix...")
            self.similarity_matrix = cosine_similarity(
                self.combined_features,
                dense_output=False
            )
            
            # Apply minimum similarity threshold (sparsify)
            self.similarity_matrix.data[
                self.similarity_matrix.data < self.min_similarity
            ] = 0
            self.similarity_matrix.eliminate_zeros()
            
            # Calculate sparsity
            sparsity = (1 - self.similarity_matrix.nnz / 
                       (self.similarity_matrix.shape[0] * self.similarity_matrix.shape[1])) * 100
            
            print(f"    ✓ Similarity matrix computed: {self.similarity_matrix.shape}")
            print(f"    ✓ Sparsity: {sparsity:.2f}%")
            print(f"    ✓ Non-zero entries: {self.similarity_matrix.nnz:,}")
        else:
            # Large dataset: use on-demand computation (no pre-computed matrix)
            print(f"    • Large dataset detected - using on-demand similarity computation")
            print(f"    • This saves memory: ~{(n_movies * n_movies * 8) / (1024**3):.1f} GB avoided")
            
            # Normalize features for efficient cosine similarity computation
            # Cosine similarity = dot product of normalized vectors
            self.combined_features = normalize(self.combined_features, norm='l2', axis=1)
            
            # Set similarity_matrix to None to indicate on-demand mode
            self.similarity_matrix = None
            
            print(f"    ✓ Features normalized for on-demand similarity computation")
            print(f"    ✓ Memory usage: {self._calculate_memory_usage():.1f} MB")
    
    def _build_user_profile(self, user_id: int) -> Optional[np.ndarray]:
        """
        Build user profile from their rating history.
        
        Args:
            user_id: User ID
            
        Returns:
            User profile vector (weighted average of rated movie features)
        """
        # Check cache first
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        
        # Get user's ratings
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        
        if len(user_ratings) == 0:
            return None
        
        # Get movie indices for rated movies
        rated_movie_ids = user_ratings['movieId'].values
        rated_movie_indices = []
        ratings = []
        
        for mid, rating in zip(rated_movie_ids, user_ratings['rating'].values):
            if mid in self.movie_mapper:
                rated_movie_indices.append(self.movie_mapper[mid])
                ratings.append(rating)
        
        if len(rated_movie_indices) == 0:
            return None
        
        # Get features for rated movies
        rated_features = self.combined_features[rated_movie_indices]
        
        # Weight by ratings (higher ratings = more influence)
        # Normalize ratings to 0-1 range
        ratings_norm = np.array(ratings) / 5.0
        
        # Create weighted profile (ratings × features)
        user_profile = (rated_features.T @ ratings_norm).T / len(ratings)
        
        # Normalize profile
        user_profile = normalize(user_profile.reshape(1, -1), norm='l2')[0]
        
        # Cache profile
        self.user_profiles[user_id] = user_profile
        
        return user_profile
    
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
        
        # Similarity matrix
        if self.similarity_matrix is not None:
            total_bytes += self.similarity_matrix.data.nbytes
            total_bytes += self.similarity_matrix.indices.nbytes
            total_bytes += self.similarity_matrix.indptr.nbytes
        
        return total_bytes / (1024 * 1024)
    
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
        
        # Get feature vectors
        genre_vec = self.genre_features[movie_idx].toarray().flatten()
        tag_vec = self.tag_features[movie_idx].toarray().flatten()
        title_vec = self.title_features[movie_idx].toarray().flatten()
        
        # Get feature names
        genre_features = self.genre_vectorizer.get_feature_names_out() if hasattr(
            self.genre_vectorizer, 'get_feature_names_out'
        ) else []
        
        tag_features = self.tag_vectorizer.get_feature_names_out() if hasattr(
            self.tag_vectorizer, 'get_feature_names_out'
        ) and len(tag_vec) > 0 else []
        
        title_features = self.title_vectorizer.get_feature_names_out() if hasattr(
            self.title_vectorizer, 'get_feature_names_out'
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
            'combined_features': self.combined_features,
            'similarity_matrix': self.similarity_matrix,
            'genre_vectorizer': self.genre_vectorizer,
            'tag_vectorizer': self.tag_vectorizer,
            'title_vectorizer': self.title_vectorizer,
            'genre_features': self.genre_features,
            'tag_features': self.tag_features,
            'title_features': self.title_features,
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
        
        self.combined_features = state.get('combined_features')
        self.similarity_matrix = state.get('similarity_matrix')
        self.genre_vectorizer = state.get('genre_vectorizer')
        self.tag_vectorizer = state.get('tag_vectorizer')
        self.title_vectorizer = state.get('title_vectorizer')
        self.genre_features = state.get('genre_features')
        self.tag_features = state.get('tag_features')
        self.title_features = state.get('title_features')
        self.movie_mapper = state.get('movie_mapper', {})
        self.movie_inv_mapper = state.get('movie_inv_mapper', {})
        self.params = state.get('params', self.params)
        
        # Restore metrics
        metrics_data = state.get('metrics', {})
        if metrics_data:
            self.metrics.rmse = metrics_data.get('rmse', 0.0)
            self.metrics.training_time = metrics_data.get('training_time', 0.0)
            self.metrics.coverage = metrics_data.get('coverage', 0.0)
            self.metrics.memory_usage_mb = metrics_data.get('memory_mb', 0.0)
        
        self.is_trained = True
    
    def __getstate__(self):
        """Prepare object for pickling by removing unpicklable lambda functions."""
        state = self.__dict__.copy()
        # Store vectorizer vocabulary and parameters instead of the vectorizer objects
        if hasattr(self, 'genre_vectorizer') and self.genre_vectorizer is not None:
            state['_genre_vocab'] = self.genre_vectorizer.vocabulary_
            state['_genre_idf'] = self.genre_vectorizer.idf_ if hasattr(self.genre_vectorizer, 'idf_') else None
            del state['genre_vectorizer']
        return state
    
    def __setstate__(self, state):
        """Restore object from pickled state by recreating vectorizers."""
        # Restore basic state
        self.__dict__.update(state)
        
        # Recreate genre_vectorizer from saved vocabulary
        if '_genre_vocab' in state:
            self.genre_vectorizer = TfidfVectorizer(
                tokenizer=lambda x: x,
                lowercase=False,
                preprocessor=lambda x: x
            )
            self.genre_vectorizer.vocabulary_ = state['_genre_vocab']
            if state.get('_genre_idf') is not None:
                self.genre_vectorizer.idf_ = state['_genre_idf']
            # Clean up temporary keys
            del self._genre_vocab
            if hasattr(self, '_genre_idf'):
                del self._genre_idf
    
    def get_explanation_context(self, user_id: int, movie_id: int) -> Dict[str, Any]:
        """
        Get context for explaining why this movie was recommended.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Dictionary with explanation context specific to Content-Based filtering
        """
        if not self.is_trained:
            return {}
        
        try:
            # Get movie details
            if movie_id not in self.movies_df['movieId'].values:
                return {'method': 'fallback', 'reason': 'Movie not in database'}
            
            movie_data = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
            
            # Get feature importance for this movie
            feature_info = self.get_feature_importance(movie_id)
            
            # Get user's rated movies to find similar ones
            user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
            
            if len(user_ratings) == 0:
                return {
                    'method': 'content_based',
                    'reason': 'New user - recommended based on movie features',
                    'movie_title': movie_data['title'],
                    'genres': movie_data.get('genres', '').split('|'),
                    'feature_weights': feature_info
                }
            
            # Find most similar movies the user liked
            highly_rated = user_ratings[user_ratings['rating'] >= 4.0]['movieId'].tolist()
            
            similar_movies = []
            if len(highly_rated) > 0:
                # Get similarities to user's highly rated movies
                for rated_movie_id in highly_rated[:5]:  # Top 5 rated movies
                    if rated_movie_id in self.movie_similarity_cache:
                        sim_score = self.movie_similarity_cache[rated_movie_id].get(movie_id, 0)
                        if sim_score > 0.1:  # Threshold for relevance
                            rated_movie = self.movies_df[self.movies_df['movieId'] == rated_movie_id].iloc[0]
                            similar_movies.append({
                                'title': rated_movie['title'],
                                'similarity': float(sim_score),
                                'user_rating': float(user_ratings[user_ratings['movieId'] == rated_movie_id]['rating'].iloc[0])
                            })
            
            # Sort by similarity
            similar_movies.sort(key=lambda x: x['similarity'], reverse=True)
            
            return {
                'method': 'content_based',
                'movie_title': movie_data['title'],
                'genres': movie_data.get('genres', '').split('|'),
                'similar_to_liked': similar_movies[:3],  # Top 3 similar movies
                'feature_weights': feature_info,
                'user_rated_count': len(user_ratings)
            }
            
        except Exception as e:
            return {
                'method': 'error',
                'reason': f'Error generating explanation: {str(e)}'
            }
