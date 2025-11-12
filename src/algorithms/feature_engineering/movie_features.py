"""
CineMatch V2.1.0 - Movie Feature Extraction

Handles extraction and combination of movie features (genres, tags, titles)
using TF-IDF vectorization for content-based filtering.

Author: CineMatch Development Team
Date: November 12, 2025
Version: 2.1.0
"""

from typing import Dict, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import re


# Module-level identity function (picklable, unlike lambdas)
def identity_function(x):
    """Identity function for TfidfVectorizer - returns input unchanged."""
    return x


class MovieFeatureExtractor:
    """
    Extracts and combines movie features for content-based filtering.
    
    Handles:
    - Genre feature extraction (multi-hot TF-IDF)
    - Tag feature extraction (TF-IDF on aggregated tags)
    - Title feature extraction (TF-IDF on title words)
    - Feature combination with configurable weights
    - Feature normalization
    
    Features are extracted using TF-IDF vectorization and combined
    with user-specified weights to create a unified feature matrix.
    """
    
    def __init__(
        self,
        genre_weight: float = 0.5,
        tag_weight: float = 0.3,
        title_weight: float = 0.2
    ):
        """
        Initialize feature extractor.
        
        Args:
            genre_weight: Weight for genre features (default: 0.5)
            tag_weight: Weight for tag features (default: 0.3)
            title_weight: Weight for title features (default: 0.2)
        """
        self.genre_weight = genre_weight
        self.tag_weight = tag_weight
        self.title_weight = title_weight
        
        # Feature vectorizers (use identity_function for picklability)
        self.genre_vectorizer = TfidfVectorizer(
            tokenizer=identity_function,
            lowercase=False,
            preprocessor=identity_function
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
        
        # Movie mapping
        self.movie_mapper = {}
        self.movie_inv_mapper = {}
    
    def load_and_preprocess_tags(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """
        Load and preprocess tags data from CSV file.
        
        Args:
            movies_df: DataFrame containing movie information
            
        Returns:
            movies_df with 'tags_text' column added
        """
        try:
            data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'ml-32m'
            tags_path = data_path / 'tags.csv'
            
            if tags_path.exists():
                # Load tags
                tags_df = pd.read_csv(
                    tags_path,
                    usecols=['userId', 'movieId', 'tag']
                )
                
                # Clean tags: lowercase, remove special chars
                tags_df['tag'] = tags_df['tag'].str.lower()
                tags_df['tag'] = tags_df['tag'].str.replace(r'[^a-z0-9\s]', '', regex=True)
                
                # Aggregate tags by movie
                movie_tags = tags_df.groupby('movieId')['tag'].apply(
                    lambda x: ' '.join(x.unique())
                ).reset_index()
                movie_tags.columns = ['movieId', 'tags_text']
                
                # Merge with movies
                movies_df = movies_df.merge(
                    movie_tags,
                    on='movieId',
                    how='left'
                )
                movies_df['tags_text'].fillna('', inplace=True)
                
                print(f"    ✓ Loaded {len(tags_df)} tags for {len(movie_tags)} movies")
            else:
                print(f"    ⚠ Tags file not found, using genres and titles only")
                movies_df['tags_text'] = ''
                
        except Exception as e:
            print(f"    ⚠ Error loading tags: {e}")
            movies_df['tags_text'] = ''
        
        return movies_df
    
    def extract_genre_features(self, movies_df: pd.DataFrame) -> csr_matrix:
        """
        Extract genre features using TF-IDF on genre lists.
        
        Args:
            movies_df: DataFrame with 'genres_list' column
            
        Returns:
            Sparse matrix of genre features
        """
        print("    • Processing genre features...")
        genre_lists = movies_df['genres_list'].apply(
            lambda x: x if isinstance(x, list) else []
        ).tolist()
        
        self.genre_features = self.genre_vectorizer.fit_transform(genre_lists)
        print(f"    ✓ Genre features: {self.genre_features.shape[1]} dimensions")
        
        return self.genre_features
    
    def extract_tag_features(self, movies_df: pd.DataFrame) -> csr_matrix:
        """
        Extract tag features using TF-IDF on aggregated tag text.
        
        Args:
            movies_df: DataFrame with 'tags_text' column
            
        Returns:
            Sparse matrix of tag features
        """
        print("    • Processing tag features...")
        tags_text = movies_df['tags_text'].fillna('').tolist()
        
        # Only fit if we have tags
        if any(tags_text):
            self.tag_features = self.tag_vectorizer.fit_transform(tags_text)
        else:
            # Create empty sparse matrix if no tags
            self.tag_features = csr_matrix((len(movies_df), 1))
        
        print(f"    ✓ Tag features: {self.tag_features.shape[1]} dimensions")
        
        return self.tag_features
    
    def extract_title_features(self, movies_df: pd.DataFrame) -> csr_matrix:
        """
        Extract title features using TF-IDF on title words.
        
        Args:
            movies_df: DataFrame with 'title' column
            
        Returns:
            Sparse matrix of title features
        """
        print("    • Processing title features...")
        titles = movies_df['title'].fillna('').tolist()
        
        # Extract title text (remove year in parentheses)
        title_texts = [re.sub(r'\s*\(\d{4}\)\s*', '', title) for title in titles]
        self.title_features = self.title_vectorizer.fit_transform(title_texts)
        
        print(f"    ✓ Title features: {self.title_features.shape[1]} dimensions")
        
        return self.title_features
    
    def combine_features(
        self,
        genre_features: Optional[csr_matrix] = None,
        tag_features: Optional[csr_matrix] = None,
        title_features: Optional[csr_matrix] = None
    ) -> csr_matrix:
        """
        Combine and weight feature matrices.
        
        Args:
            genre_features: Genre feature matrix (uses self.genre_features if None)
            tag_features: Tag feature matrix (uses self.tag_features if None)
            title_features: Title feature matrix (uses self.title_features if None)
            
        Returns:
            Combined weighted feature matrix
        """
        print("    • Combining features with weights...")
        
        # Use instance features if not provided
        genre_features = genre_features if genre_features is not None else self.genre_features
        tag_features = tag_features if tag_features is not None else self.tag_features
        title_features = title_features if title_features is not None else self.title_features
        
        # Normalize each feature type
        genre_features_norm = normalize(genre_features, norm='l2', axis=1)
        tag_features_norm = normalize(tag_features, norm='l2', axis=1)
        title_features_norm = normalize(title_features, norm='l2', axis=1)
        
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
        
        print(f"    ✓ Combined features: {self.combined_features.shape}")
        
        return self.combined_features
    
    def build_feature_matrix(self, movies_df: pd.DataFrame) -> Tuple[csr_matrix, Dict, Dict]:
        """
        Build complete feature matrix from movie data.
        
        This is the main entry point that orchestrates the entire
        feature extraction process.
        
        Args:
            movies_df: DataFrame containing movie information
            
        Returns:
            Tuple of (combined_features, movie_mapper, movie_inv_mapper)
        """
        # Create movie index mapping
        self.movie_mapper = {
            mid: idx for idx, mid in enumerate(movies_df['movieId'].values)
        }
        self.movie_inv_mapper = {
            idx: mid for mid, idx in self.movie_mapper.items()
        }
        
        # Extract individual feature types
        self.extract_genre_features(movies_df)
        self.extract_tag_features(movies_df)
        self.extract_title_features(movies_df)
        
        # Combine features
        self.combine_features()
        
        return self.combined_features, self.movie_mapper, self.movie_inv_mapper
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """
        Get dimensions of each feature type.
        
        Returns:
            Dictionary with feature dimensions
        """
        return {
            'genre': self.genre_features.shape[1] if self.genre_features is not None else 0,
            'tag': self.tag_features.shape[1] if self.tag_features is not None else 0,
            'title': self.title_features.shape[1] if self.title_features is not None else 0,
            'combined': self.combined_features.shape[1] if self.combined_features is not None else 0
        }
