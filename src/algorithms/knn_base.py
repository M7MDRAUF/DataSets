"""
CineMatch V2.1.0 - KNN Base Recommender

Abstract base class for all K-Nearest Neighbors recommendation algorithms.
Implements Template Method pattern to eliminate code duplication between
User-Based and Item-Based KNN implementations.

Author: CineMatch Development Team
Date: November 12, 2025
Version: 2.1.0
"""

import sys
from pathlib import Path
from typing import Dict, Optional
from abc import abstractmethod
import pandas as pd
import numpy as np
import time
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.algorithms.base_recommender import BaseRecommender


class KNNBaseRecommender(BaseRecommender):
    """
    Abstract base class for K-Nearest Neighbors recommenders.
    
    Implements the Template Method pattern to provide common functionality
    for both User-Based and Item-Based KNN algorithms.
    
    Template Method Pattern:
    - fit(): Common training flow (calls abstract methods for specifics)
    - _create_matrix(): Abstract (user-item vs item-user)
    - _compute_similarity(): Abstract (user vs item similarity)
    - _calculate_metrics(): Common metrics calculation
    
    Benefits:
    - Eliminates ~210 lines of code duplication
    - Single source of truth for KNN logic
    - Easier to maintain and extend
    - Consistent behavior across KNN variants
    """
    
    def __init__(self, name: str, n_neighbors: int, similarity_metric: str = 'cosine', **kwargs):
        """
        Initialize KNN base recommender.
        
        Args:
            name: Algorithm name (e.g., "KNN User-Based")
            n_neighbors: Number of nearest neighbors to consider
            similarity_metric: Similarity metric ('cosine', 'euclidean', etc.)
            **kwargs: Additional parameters passed to parent
        """
        super().__init__(name, n_neighbors=n_neighbors, 
                        similarity_metric=similarity_metric, **kwargs)
        
        # Common KNN parameters
        self.n_neighbors = n_neighbors
        self.similarity_metric = similarity_metric
        
        # KNN model (scikit-learn NearestNeighbors)
        self.knn_model = NearestNeighbors(
            n_neighbors=n_neighbors + 1,  # +1 because it includes the query point
            metric=similarity_metric,
            algorithm='brute'  # Better for sparse matrices
        )
        
        # Common data structures (subclasses use these)
        self.interaction_matrix = None  # Will be user-item or item-user
        self.user_mapper = {}
        self.user_inv_mapper = {}
        self.movie_mapper = {}
        self.movie_inv_mapper = {}
        self.global_mean = 0.0
        
        # Cache for storing computed values
        self._cache = {}
    
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        """
        Train the KNN model (Template Method).
        
        This method defines the common training algorithm flow.
        Subclasses implement specific steps via abstract methods.
        
        Algorithm:
        1. Store data references
        2. Create interaction matrix (user-item or item-user)
        3. Train KNN model on the matrix
        4. Calculate performance metrics
        5. Report training results
        """
        print(f"\n🎯 Training {self.name}...")
        start_time = time.time()
        
        # Store data references
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()
        
        # Add genres_list column if not present (as tuples for hashability)
        if 'genres_list' not in self.movies_df.columns:
            self.movies_df['genres_list'] = self.movies_df['genres'].str.split('|').apply(tuple)
        
        # Calculate global mean for fallback predictions
        self.global_mean = ratings_df['rating'].mean()
        
        # Step 1: Create interaction matrix (subclass-specific)
        print(f"  • Creating {self._get_matrix_description()}...")
        self._create_interaction_matrix(ratings_df)
        
        # Step 2: Train KNN model
        print("  • Training KNN model...")
        self.knn_model.fit(self.interaction_matrix)
        
        # Step 3: Perform any additional preprocessing (optional)
        self._post_fit_preprocessing()
        
        # Step 4: Calculate metrics
        training_time = time.time() - start_time
        self.metrics.training_time = training_time
        self.is_trained = True
        
        print("  • Calculating performance metrics...")
        self._calculate_metrics(ratings_df, training_time)
        
        # Step 5: Report results
        self._print_training_summary()
    
    @abstractmethod
    def _create_interaction_matrix(self, ratings_df: pd.DataFrame) -> None:
        """
        Create the interaction matrix for KNN (user-item or item-user).
        
        Subclasses must implement this to create either:
        - User-item matrix (users x movies) for User-Based KNN
        - Item-user matrix (movies x users) for Item-Based KNN
        
        Must set: self.interaction_matrix, mappings, and any specific data structures
        """
        pass
    
    @abstractmethod
    def _get_matrix_description(self) -> str:
        """Return description of matrix type for logging"""
        pass
    
    def _post_fit_preprocessing(self) -> None:
        """
        Optional post-fit preprocessing (template hook).
        
        Subclasses can override this to perform additional preprocessing
        after KNN model training (e.g., precompute similarity matrix).
        
        Default: No additional preprocessing
        """
        pass
    
    def _calculate_metrics(self, ratings_df: pd.DataFrame, training_time: float) -> None:
        """
        Calculate common performance metrics for KNN models.
        
        Metrics calculated:
        - RMSE: Root Mean Squared Error on test sample
        - Coverage: Percentage of items that can be recommended
        - Memory Usage: Size of sparse matrices
        - Sparsity: Percentage of empty cells in matrix
        """
        # Store training time
        self.metrics.training_time = training_time
        
        # Calculate RMSE on test sample
        self._calculate_rmse(ratings_df)
        
        # Calculate coverage (subclass-specific, default 100%)
        self.metrics.coverage = self._calculate_coverage()
        
        # Calculate memory usage
        matrix_size_mb = self._calculate_memory_usage()
        self.metrics.memory_usage_mb = matrix_size_mb
    
    def _calculate_rmse(self, ratings_df: pd.DataFrame) -> None:
        """Calculate RMSE on a test sample (common logic)"""
        test_sample = ratings_df.sample(min(5000, len(ratings_df)), random_state=42)
        
        squared_errors = []
        for _, row in test_sample.iterrows():
            try:
                pred = self.predict(row['userId'], row['movieId'])
                if not np.isnan(pred):
                    squared_errors.append((pred - row['rating']) ** 2)
            except:
                continue
        
        if squared_errors:
            self.metrics.rmse = np.sqrt(np.mean(squared_errors))
        else:
            self.metrics.rmse = np.nan
    
    def _calculate_coverage(self) -> float:
        """
        Calculate coverage metric (can be overridden by subclasses).
        
        Default: 100% (all items can be recommended)
        Item-Based KNN may override with actual valid items percentage
        """
        return 100.0
    
    def _calculate_memory_usage(self) -> float:
        """Calculate memory usage of sparse matrices"""
        if self.interaction_matrix is None:
            return 0.0
        
        # Calculate sparse matrix memory
        matrix_size_mb = (
            self.interaction_matrix.data.nbytes + 
            self.interaction_matrix.indices.nbytes + 
            self.interaction_matrix.indptr.nbytes
        ) / (1024 * 1024)
        
        # Subclasses can add additional memory (e.g., similarity matrix)
        matrix_size_mb += self._get_additional_memory_usage()
        
        return matrix_size_mb
    
    def _get_additional_memory_usage(self) -> float:
        """Hook for subclasses to add additional memory usage (default: 0)"""
        return 0.0
    
    def _print_training_summary(self) -> None:
        """Print training summary (common format)"""
        print(f"✓ {self.name} trained successfully!")
        print(f"  • Training time: {self.metrics.training_time:.1f}s")
        print(f"  • RMSE: {self.metrics.rmse:.4f}")
        print(f"  • Matrix size: {self.interaction_matrix.shape}")
        print(f"  • Sparsity: {self._calculate_sparsity():.2f}%")
        print(f"  • Coverage: {self.metrics.coverage:.1f}%")
        print(f"  • Memory usage: {self.metrics.memory_usage_mb:.1f} MB")
    
    def _calculate_sparsity(self) -> float:
        """Calculate matrix sparsity percentage"""
        if self.interaction_matrix is None:
            return 0.0
        
        total_cells = np.prod(self.interaction_matrix.shape)
        non_zero_cells = self.interaction_matrix.nnz
        sparsity = (1 - non_zero_cells / total_cells) * 100
        
        return sparsity
    
    def _get_from_cache(self, key: str) -> Optional[any]:
        """Get value from cache (thread-safe)"""
        return self._cache.get(key)
    
    def _add_to_cache(self, key: str, value: any, max_cache_size: int = 1000) -> None:
        """
        Add value to cache with size limit.
        
        Implements LRU-like behavior by clearing cache when full.
        """
        if len(self._cache) >= max_cache_size:
            # Simple strategy: clear entire cache when full
            # More sophisticated: implement proper LRU
            self._cache.clear()
        
        self._cache[key] = value
    
    def get_model_size(self) -> float:
        """Get total model size in MB (for comparison)"""
        return self.metrics.memory_usage_mb
    
    def get_sparsity(self) -> float:
        """Get matrix sparsity percentage (for analysis)"""
        return self._calculate_sparsity()
