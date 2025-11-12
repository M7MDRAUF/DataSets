"""
CineMatch V2.1.0 - Similarity Matrix Builder

Handles computation of movie-movie similarity matrices using cosine similarity.
Supports both pre-computed matrices for small datasets and on-demand computation
for large datasets to optimize memory usage.

Author: CineMatch Development Team
Date: November 12, 2025
Version: 2.1.0
"""

from typing import Optional
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


class SimilarityMatrixBuilder:
    """
    Builds and manages movie-movie similarity matrices.
    
    Supports two modes:
    1. Pre-computed: Full similarity matrix computed upfront (small datasets)
    2. On-demand: Similarities computed as needed (large datasets)
    
    The builder automatically chooses the appropriate mode based on
    dataset size to optimize memory usage.
    """
    
    def __init__(self, min_similarity: float = 0.01):
        """
        Initialize similarity matrix builder.
        
        Args:
            min_similarity: Minimum similarity threshold for sparsification
        """
        self.min_similarity = min_similarity
        self.similarity_matrix = None
        self.normalized_features = None
        self.mode = None  # 'precomputed' or 'on-demand'
    
    def compute_similarity_matrix(
        self,
        feature_matrix: csr_matrix,
        threshold_size: int = 5000
    ) -> Optional[csr_matrix]:
        """
        Compute or prepare for similarity computation.
        
        Automatically chooses between pre-computation and on-demand
        based on dataset size.
        
        Args:
            feature_matrix: Combined feature matrix (movies x features)
            threshold_size: Size threshold for pre-computation decision
            
        Returns:
            Similarity matrix if pre-computed, None if on-demand mode
        """
        n_movies = feature_matrix.shape[0]
        
        print(f"    • Preparing for similarity computation ({n_movies} movies)")
        
        if n_movies <= threshold_size:
            # Small dataset: pre-compute full similarity matrix
            self.mode = 'precomputed'
            return self._precompute_similarity_matrix(feature_matrix)
        else:
            # Large dataset: prepare for on-demand computation
            self.mode = 'on-demand'
            return self._prepare_on_demand_similarity(feature_matrix, n_movies)
    
    def _precompute_similarity_matrix(self, feature_matrix: csr_matrix) -> csr_matrix:
        """
        Pre-compute full similarity matrix for small datasets.
        
        Args:
            feature_matrix: Combined feature matrix
            
        Returns:
            Sparse similarity matrix
        """
        print(f"    • Small dataset detected - pre-computing full similarity matrix...")
        
        # Compute cosine similarity
        self.similarity_matrix = cosine_similarity(
            feature_matrix,
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
        
        return self.similarity_matrix
    
    def _prepare_on_demand_similarity(
        self,
        feature_matrix: csr_matrix,
        n_movies: int
    ) -> None:
        """
        Prepare for on-demand similarity computation.
        
        Normalizes features for efficient on-the-fly cosine similarity.
        
        Args:
            feature_matrix: Combined feature matrix
            n_movies: Number of movies
            
        Returns:
            None (sets similarity_matrix to None to indicate on-demand mode)
        """
        print(f"    • Large dataset detected - using on-demand similarity computation")
        print(f"    • This saves memory: ~{(n_movies * n_movies * 8) / (1024**3):.1f} GB avoided")
        
        # Normalize features for efficient cosine similarity computation
        # Cosine similarity = dot product of normalized vectors
        self.normalized_features = normalize(feature_matrix, norm='l2', axis=1)
        
        # Set similarity_matrix to None to indicate on-demand mode
        self.similarity_matrix = None
        
        memory_mb = (self.normalized_features.data.nbytes + 
                    self.normalized_features.indices.nbytes +
                    self.normalized_features.indptr.nbytes) / (1024 * 1024)
        
        print(f"    ✓ Features normalized for on-demand similarity computation")
        print(f"    ✓ Memory usage: {memory_mb:.1f} MB")
        
        return None
    
    def compute_similarities_for_movie(
        self,
        movie_idx: int,
        feature_matrix: Optional[csr_matrix] = None,
        top_k: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute similarities for a specific movie.
        
        Works in both pre-computed and on-demand modes.
        
        Args:
            movie_idx: Index of the movie
            feature_matrix: Feature matrix (used if different from normalized_features)
            top_k: If specified, return only top k similar movies
            
        Returns:
            Array of similarities with other movies
        """
        if self.mode == 'precomputed' and self.similarity_matrix is not None:
            # Use pre-computed matrix
            similarities = self.similarity_matrix[movie_idx].toarray().flatten()
        else:
            # Compute on-demand
            features = feature_matrix if feature_matrix is not None else self.normalized_features
            if features is None:
                raise ValueError("No features available for similarity computation")
            
            movie_features = features[movie_idx]
            similarities = (features @ movie_features.T).toarray().flatten()
        
        # Apply minimum threshold
        similarities[similarities < self.min_similarity] = 0
        
        # Return top-k if specified
        if top_k is not None:
            top_indices = np.argsort(similarities)[-top_k:]
            result = np.zeros_like(similarities)
            result[top_indices] = similarities[top_indices]
            return result
        
        return similarities
    
    def compute_similarities_for_profile(
        self,
        user_profile: np.ndarray,
        feature_matrix: Optional[csr_matrix] = None
    ) -> np.ndarray:
        """
        Compute similarities between user profile and all movies.
        
        Args:
            user_profile: User profile vector
            feature_matrix: Feature matrix (uses normalized_features if None)
            
        Returns:
            Array of similarities with all movies
        """
        features = feature_matrix if feature_matrix is not None else self.normalized_features
        
        if features is None:
            if self.mode == 'precomputed':
                raise ValueError("No features available - similarity matrix exists but features not stored")
            else:
                raise ValueError("No features available for similarity computation")
        
        # Ensure user_profile is normalized
        user_profile_norm = user_profile / (np.linalg.norm(user_profile) + 1e-10)
        
        # Compute similarities (dot product with normalized features)
        similarities = (features @ user_profile_norm).flatten()
        
        return similarities
    
    def get_memory_usage(self) -> float:
        """
        Get memory usage of similarity structures in MB.
        
        Returns:
            Memory usage in megabytes
        """
        memory_bytes = 0
        
        if self.similarity_matrix is not None:
            memory_bytes += self.similarity_matrix.data.nbytes
            memory_bytes += self.similarity_matrix.indices.nbytes
            memory_bytes += self.similarity_matrix.indptr.nbytes
        
        if self.normalized_features is not None:
            memory_bytes += self.normalized_features.data.nbytes
            memory_bytes += self.normalized_features.indices.nbytes
            memory_bytes += self.normalized_features.indptr.nbytes
        
        return memory_bytes / (1024 * 1024)
    
    def is_precomputed(self) -> bool:
        """Check if similarity matrix is pre-computed."""
        return self.mode == 'precomputed' and self.similarity_matrix is not None
