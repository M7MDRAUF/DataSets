"""
CineMatch V1.0.0 - sklearn-based SVD Model Class

Shared SimpleSVDRecommender class for training and inference.
Windows-compatible alternative to scikit-surprise.

Author: CineMatch Team
Date: October 28, 2025
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from typing import Tuple


class SimpleSVDRecommender:
    """
    Simple SVD-based recommender using sklearn.
    Windows-compatible alternative to scikit-surprise.
    """
    
    def __init__(self, n_components=100):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_ids = None
        self.movie_ids = None
        self.user_mapper = {}
        self.movie_mapper = {}
        self.user_inv_mapper = {}
        self.movie_inv_mapper = {}
        self.user_factors = None
        self.movie_factors = None
        self.global_mean = 0
        self.user_bias = {}
        self.movie_bias = {}
        
    def fit(self, ratings_df: pd.DataFrame):
        """Train the SVD model"""
        print("\nPreparing data for SVD...")
        
        # Create user and movie mappings
        self.user_ids = ratings_df['userId'].unique()
        self.movie_ids = ratings_df['movieId'].unique()
        
        self.user_mapper = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.movie_mapper = {mid: idx for idx, mid in enumerate(self.movie_ids)}
        self.user_inv_mapper = {idx: uid for uid, idx in self.user_mapper.items()}
        self.movie_inv_mapper = {idx: mid for mid, idx in self.movie_mapper.items()}
        
        # Map IDs to indices
        ratings_df = ratings_df.copy()
        ratings_df['user_idx'] = ratings_df['userId'].map(self.user_mapper)
        ratings_df['movie_idx'] = ratings_df['movieId'].map(self.movie_mapper)
        
        # Calculate global mean and biases
        self.global_mean = ratings_df['rating'].mean()
        
        # Calculate user biases
        user_means = ratings_df.groupby('userId')['rating'].mean()
        self.user_bias = (user_means - self.global_mean).to_dict()
        
        # Calculate movie biases
        movie_means = ratings_df.groupby('movieId')['rating'].mean()
        self.movie_bias = (movie_means - self.global_mean).to_dict()
        
        # Create sparse matrix
        n_users = len(self.user_ids)
        n_movies = len(self.movie_ids)
        
        print(f"  • Creating sparse matrix: {n_users} users × {n_movies} movies")
        
        # Subtract biases
        ratings_df['centered_rating'] = (
            ratings_df['rating'] - self.global_mean - 
            ratings_df['userId'].map(self.user_bias) - 
            ratings_df['movieId'].map(self.movie_bias)
        )
        
        sparse_matrix = csr_matrix(
            (ratings_df['centered_rating'], 
             (ratings_df['user_idx'], ratings_df['movie_idx'])),
            shape=(n_users, n_movies)
        )
        
        # Train SVD
        print(f"  • Training SVD with {self.n_components} components...")
        self.svd.fit(sparse_matrix)
        
        # Store user and movie factors
        self.user_factors = self.svd.transform(sparse_matrix)
        self.movie_factors = self.svd.components_.T
        
        explained_var = self.svd.explained_variance_ratio_.sum() * 100
        print(f"  ✓ SVD training complete")
        print(f"    - Explained variance: {explained_var:.2f}%")
        
    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating for a user-movie pair"""
        if user_id not in self.user_mapper:
            return self.global_mean
        if movie_id not in self.movie_mapper:
            return self.global_mean
            
        user_idx = self.user_mapper[user_id]
        movie_idx = self.movie_mapper[movie_id]
        
        # Prediction = global_mean + user_bias + movie_bias + user_factors · movie_factors
        pred = (
            self.global_mean + 
            self.user_bias.get(user_id, 0) +
            self.movie_bias.get(movie_id, 0) +
            np.dot(self.user_factors[user_idx], self.movie_factors[movie_idx])
        )
        
        # Clip to valid rating range
        return np.clip(pred, 0.5, 5.0)
    
    def test(self, test_df: pd.DataFrame) -> float:
        """Calculate RMSE on test set"""
        predictions = []
        actuals = []
        
        for _, row in test_df.iterrows():
            pred = self.predict(row['userId'], row['movieId'])
            predictions.append(pred)
            actuals.append(row['rating'])
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        
        return rmse
    
    def get_top_n_recommendations(
        self,
        user_id: int,
        n: int = 10,
        exclude_movies: set = None
    ) -> list:
        """
        Get top-N movie recommendations for a user.
        
        Args:
            user_id: User ID
            n: Number of recommendations
            exclude_movies: Set of movie IDs to exclude
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if user_id not in self.user_mapper:
            # New user - return popular movies
            movie_ratings = []
            for movie_id in self.movie_ids:
                if exclude_movies and movie_id in exclude_movies:
                    continue
                rating = self.global_mean + self.movie_bias.get(movie_id, 0)
                movie_ratings.append((movie_id, rating))
        else:
            # Existing user - predict all movies
            movie_ratings = []
            for movie_id in self.movie_ids:
                if exclude_movies and movie_id in exclude_movies:
                    continue
                rating = self.predict(user_id, movie_id)
                movie_ratings.append((movie_id, rating))
        
        # Sort by rating and return top N
        movie_ratings.sort(key=lambda x: x[1], reverse=True)
        return movie_ratings[:n]
