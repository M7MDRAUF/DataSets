"""
CineMatch V2.1.0 - User Profile Builder

Builds user profiles from rating history for content-based filtering.
Creates weighted feature vectors representing user preferences.

Author: CineMatch Development Team
Date: November 12, 2025
Version: 2.1.0
"""

from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


class UserProfileBuilder:
    """
    Builds user profiles from rating history.
    
    Creates a weighted average of movie features based on user ratings,
    where movies with higher ratings contribute more to the profile.
    Profiles are cached for efficiency.
    """
    
    def __init__(self):
        """Initialize user profile builder."""
        self.user_profiles = {}  # Cache for computed profiles
    
    def build_profile(
        self,
        user_id: int,
        ratings_df: pd.DataFrame,
        feature_matrix: csr_matrix,
        movie_mapper: Dict[int, int]
    ) -> Optional[np.ndarray]:
        """
        Build user profile from their rating history.
        
        Creates a weighted average of rated movie features, where
        higher-rated movies have more influence on the profile.
        
        Args:
            user_id: User ID
            ratings_df: DataFrame with user ratings
            feature_matrix: Combined feature matrix (movies x features)
            movie_mapper: Mapping from movieId to matrix index
            
        Returns:
            User profile vector (weighted average of rated movie features)
            or None if user has no valid ratings
        """
        # Check cache first
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        
        # Get user's ratings
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        
        if len(user_ratings) == 0:
            return None
        
        # Get movie indices for rated movies
        rated_movie_ids = user_ratings['movieId'].values
        rated_movie_indices = []
        ratings = []
        
        for mid, rating in zip(rated_movie_ids, user_ratings['rating'].values):
            if mid in movie_mapper:
                rated_movie_indices.append(movie_mapper[mid])
                ratings.append(rating)
        
        if len(rated_movie_indices) == 0:
            return None
        
        # Get features for rated movies
        rated_features = feature_matrix[rated_movie_indices]
        
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
    
    def update_profile(
        self,
        user_id: int,
        new_movie_id: int,
        new_rating: float,
        feature_matrix: csr_matrix,
        movie_mapper: Dict[int, int]
    ) -> Optional[np.ndarray]:
        """
        Update user profile with a new rating.
        
        This is more efficient than rebuilding from scratch when
        adding a single new rating.
        
        Args:
            user_id: User ID
            new_movie_id: ID of newly rated movie
            new_rating: Rating value
            feature_matrix: Combined feature matrix
            movie_mapper: Mapping from movieId to matrix index
            
        Returns:
            Updated user profile or None if movie not in mapper
        """
        if new_movie_id not in movie_mapper:
            return None
        
        movie_idx = movie_mapper[new_movie_id]
        movie_features = feature_matrix[movie_idx].toarray().flatten()
        
        # Get existing profile
        existing_profile = self.user_profiles.get(user_id)
        
        if existing_profile is None:
            # No existing profile, create new one with just this rating
            user_profile = movie_features * (new_rating / 5.0)
        else:
            # Update existing profile
            # Simple exponential moving average (weight recent ratings more)
            alpha = 0.1  # Learning rate
            user_profile = (1 - alpha) * existing_profile + alpha * movie_features * (new_rating / 5.0)
        
        # Normalize profile
        user_profile = user_profile / (np.linalg.norm(user_profile) + 1e-10)
        
        # Update cache
        self.user_profiles[user_id] = user_profile
        
        return user_profile
    
    def get_profile(self, user_id: int) -> Optional[np.ndarray]:
        """
        Get cached user profile.
        
        Args:
            user_id: User ID
            
        Returns:
            User profile if cached, None otherwise
        """
        return self.user_profiles.get(user_id)
    
    def has_profile(self, user_id: int) -> bool:
        """
        Check if user has a cached profile.
        
        Args:
            user_id: User ID
            
        Returns:
            True if profile is cached
        """
        return user_id in self.user_profiles
    
    def clear_cache(self, user_id: Optional[int] = None):
        """
        Clear profile cache.
        
        Args:
            user_id: If specified, clear only this user's profile.
                    If None, clear all profiles.
        """
        if user_id is not None:
            self.user_profiles.pop(user_id, None)
        else:
            self.user_profiles.clear()
    
    def get_cache_size(self) -> int:
        """
        Get number of cached profiles.
        
        Returns:
            Number of profiles in cache
        """
        return len(self.user_profiles)
    
    def get_profile_stats(self, user_id: int) -> Optional[Dict[str, float]]:
        """
        Get statistics about a user profile.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with profile statistics or None if not cached
        """
        profile = self.user_profiles.get(user_id)
        
        if profile is None:
            return None
        
        return {
            'dimensions': len(profile),
            'mean': float(np.mean(profile)),
            'std': float(np.std(profile)),
            'min': float(np.min(profile)),
            'max': float(np.max(profile)),
            'norm': float(np.linalg.norm(profile)),
            'non_zero': int(np.count_nonzero(profile)),
            'sparsity': float((1 - np.count_nonzero(profile) / len(profile)) * 100)
        }
