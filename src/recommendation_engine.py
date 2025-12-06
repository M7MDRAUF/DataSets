"""
CineMatch V2.1.6 - Recommendation Engine

Core recommendation logic using the pre-trained SVD model.
Implements F-02 (Recommendation Generation) and F-04 (Model Pre-computation).

Author: CineMatch Team
Date: October 24, 2025
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any, Union

import pandas as pd
import numpy as np
import joblib

# Import data loading functions
sys.path.append(str(Path(__file__).parent.parent))
from src.data_processing import load_ratings, load_movies


# Model configuration
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "svd_model.pkl"


class RecommendationEngine:
    """
    Movie recommendation engine using pre-trained SVD model.
    
    Features:
    - F-02: Generate personalized recommendations
    - F-04: Load pre-trained model for instant predictions
    - F-07: "Surprise Me" mode for serendipity recommendations
    
    Attributes:
        model_path: Path to pre-trained SVD model file
        model: Loaded SVD model instance (None until load_model() called)
        ratings_df: Ratings DataFrame (None until load_data() called)
        movies_df: Movies DataFrame (None until load_data() called)
        user_rated_movies: Cache for user rating history
    """
    
    def __init__(self, model_path: Union[Path, str] = MODEL_PATH) -> None:
        """
        Initialize recommendation engine.
        
        Args:
            model_path: Path to pre-trained SVD model
        """
        self.model_path: Path = Path(model_path) if isinstance(model_path, str) else model_path
        self.model: Optional[Any] = None
        self.ratings_df: Optional[pd.DataFrame] = None
        self.movies_df: Optional[pd.DataFrame] = None
        self.user_rated_movies: Dict[int, pd.DataFrame] = {}  # Cache for user history
        
    def load_model(self) -> None:
        """
        Load pre-trained SVD model.
        
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                "Please run model training first:\n"
                "  python src/model_training.py"
            )
        
        print(f"Loading model from {self.model_path}...")
        self.model = joblib.load(self.model_path)
        print("  ✓ Model loaded successfully")
        
    def load_data(self, sample_size: Optional[int] = None) -> None:
        """
        Load ratings and movies data.
        
        Args:
            sample_size: If provided, sample this many ratings (for faster loading)
        """
        print("Loading dataset...")
        self.ratings_df = load_ratings(sample_size=sample_size)
        self.movies_df = load_movies()
        print("  ✓ Data loaded successfully")
        
    def get_user_history(self, user_id: int) -> pd.DataFrame:
        """
        Get user's rating history.
        
        Args:
            user_id: User ID to query
        
        Returns:
            DataFrame with user's rated movies and ratings
        """
        # Check cache first
        if user_id in self.user_rated_movies:
            return self.user_rated_movies[user_id]
        
        # Query from database
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id].copy()
        
        # Merge with movie info
        user_ratings = user_ratings.merge(
            self.movies_df[['movieId', 'title', 'genres']],
            on='movieId',
            how='left'
        )
        
        # Sort by rating (descending) and timestamp (recent first)
        user_ratings = user_ratings.sort_values(['rating', 'timestamp'], ascending=[False, False])
        
        # Cache result
        self.user_rated_movies[user_id] = user_ratings
        
        return user_ratings
    
    def get_recommendations(
        self,
        user_id: int,
        n: int = 10,
        exclude_rated: bool = True
    ) -> pd.DataFrame:
        """
        Generate top-N movie recommendations for a user.
        
        Implements F-02: Recommendation Generation
        
        Args:
            user_id: User ID to generate recommendations for
            n: Number of recommendations to return
            exclude_rated: If True, exclude movies the user has already rated
        
        Returns:
            DataFrame with top-N recommended movies and predicted ratings
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if self.ratings_df is None or self.movies_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Validate user exists
        if user_id not in self.ratings_df['userId'].values:
            raise ValueError(f"User ID {user_id} not found in dataset")
        
        # Get user's rated movies
        user_history = self.get_user_history(user_id)
        rated_movie_ids = set(user_history['movieId'].values)
        
        print(f"\nGenerating recommendations for User {user_id}...")
        print(f"  • User has rated {len(rated_movie_ids)} movies")
        
        # Get all movies
        all_movies = self.movies_df['movieId'].values
        
        # Filter out rated movies if requested
        if exclude_rated:
            candidate_movies = [m for m in all_movies if m not in rated_movie_ids]
        else:
            candidate_movies = all_movies
        
        print(f"  • Evaluating {len(candidate_movies)} candidate movies...")
        
        # Generate predictions for all candidate movies
        predictions = []
        for movie_id in candidate_movies:
            pred = self.model.predict(user_id, movie_id)
            predictions.append({
                'movieId': movie_id,
                'predicted_rating': float(pred.est)  # Ensure float type
            })
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        # Sort by predicted rating (descending)
        predictions_df = predictions_df.sort_values('predicted_rating', ascending=False)
        
        # Get top N
        top_predictions = predictions_df.head(n)
        
        # Merge with movie info
        recommendations = top_predictions.merge(
            self.movies_df[['movieId', 'title', 'genres', 'genres_list']],
            on='movieId',
            how='left'
        )
        
        print(f"  ✓ Generated {len(recommendations)} recommendations")
        
        return recommendations
    
    def get_surprise_recommendations(
        self,
        user_id: int,
        n: int = 10
    ) -> pd.DataFrame:
        """
        Get "Surprise Me" recommendations - movies outside user's usual taste.
        
        Implements F-07: "Surprise Me" Button
        
        Strategy: Find highly-rated movies from genres the user rarely watches
        but are similar to movies they liked.
        
        Args:
            user_id: User ID
            n: Number of recommendations
        
        Returns:
            DataFrame with serendipitous recommendations
        """
        # Get user's genre preferences
        user_history = self.get_user_history(user_id)
        
        # Explode genres to count
        user_genres = user_history.explode('genres_list')['genres_list'].value_counts()
        
        # Find underrepresented genres (user has watched few movies from)
        rare_genres = user_genres[user_genres <= user_genres.quantile(0.25)].index.tolist()
        
        if not rare_genres:
            # Fallback: just get regular recommendations
            return self.get_recommendations(user_id, n=n)
        
        # Get movies from rare genres
        rare_genre_movies = self.movies_df[
            self.movies_df['genres_list'].apply(
                lambda x: any(genre in rare_genres for genre in x)
            )
        ]
        
        # Get rated movie IDs to exclude
        rated_movie_ids = set(user_history['movieId'].values)
        rare_genre_movies = rare_genre_movies[~rare_genre_movies['movieId'].isin(rated_movie_ids)]
        
        # Predict ratings for these movies
        predictions = []
        for _, movie in rare_genre_movies.iterrows():
            pred = self.model.predict(user_id, movie['movieId'])
            predictions.append({
                'movieId': movie['movieId'],
                'predicted_rating': float(pred.est),  # Ensure float type
                'title': movie['title'],
                'genres': movie['genres'],
                'genres_list': movie['genres_list']
            })
        
        # Convert to DataFrame and sort
        surprises = pd.DataFrame(predictions)
        surprises = surprises.sort_values('predicted_rating', ascending=False).head(n)
        
        return surprises
    
    def get_similar_movies(
        self,
        movie_id: int,
        n: int = 10
    ) -> pd.DataFrame:
        """
        Find movies similar to a given movie.
        
        Implements F-10: Movie Similarity Explorer
        
        Uses the learned item (movie) latent factors from the SVD model.
        
        Args:
            movie_id: Reference movie ID
            n: Number of similar movies to return
        
        Returns:
            DataFrame with similar movies and similarity scores
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Get the movie's latent factors
        try:
            # In Surprise, movie factors are in model.qi (item factors)
            # We need the internal movie id
            inner_id = self.model.trainset.to_inner_iid(movie_id)
            movie_factors = self.model.qi[inner_id]
        except (ValueError, KeyError, IndexError) as e:
            raise ValueError(f"Movie ID {movie_id} not found in trained model")
        
        # Calculate similarity with all other movies
        similarities = []
        for iid in range(self.model.trainset.n_items):
            other_movie_id = self.model.trainset.to_raw_iid(iid)
            
            if other_movie_id == movie_id:
                continue  # Skip self
            
            other_factors = self.model.qi[iid]
            
            # Cosine similarity
            similarity = np.dot(movie_factors, other_factors) / (
                np.linalg.norm(movie_factors) * np.linalg.norm(other_factors)
            )
            
            similarities.append({
                'movieId': other_movie_id,
                'similarity': similarity
            })
        
        # Convert to DataFrame and sort
        similarities_df = pd.DataFrame(similarities)
        similarities_df = similarities_df.sort_values('similarity', ascending=False).head(n)
        
        # Merge with movie info
        similar_movies = similarities_df.merge(
            self.movies_df[['movieId', 'title', 'genres']],
            on='movieId',
            how='left'
        )
        
        return similar_movies


def validate_user_id(user_id: int, ratings_df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate if user ID exists in dataset.
    
    Args:
        user_id: User ID to validate
        ratings_df: Ratings DataFrame
    
    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    if not isinstance(user_id, (int, np.integer)):
        return False, "User ID must be an integer"
    
    if user_id <= 0:
        return False, "User ID must be positive"
    
    if user_id not in ratings_df['userId'].values:
        max_user_id = ratings_df['userId'].max()
        return False, f"User ID {user_id} not found. Valid range: 1 to {max_user_id}"
    
    return True, ""


if __name__ == "__main__":
    """
    Test the recommendation engine.
    """
    print("CineMatch V2.1.6 - Recommendation Engine Test\n")
    print("=" * 70)
    
    # Initialize engine
    engine = RecommendationEngine()
    
    # Load model and data
    try:
        engine.load_model()
        engine.load_data(sample_size=100000)  # Sample for testing
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Test with a sample user
    test_user_id = 1
    print(f"\n{'='*70}")
    print(f"Testing recommendations for User {test_user_id}")
    print(f"{'='*70}")
    
    try:
        # Get user history
        history = engine.get_user_history(test_user_id)
        print(f"\nUser {test_user_id}'s Top Rated Movies:")
        print(history[['title', 'rating', 'genres']].head(5).to_string(index=False))
        
        # Get recommendations
        recommendations = engine.get_recommendations(test_user_id, n=5)
        print(f"\nTop 5 Recommendations for User {test_user_id}:")
        print(recommendations[['title', 'predicted_rating', 'genres']].to_string(index=False))
        
        print("\n✅ Recommendation engine test successful!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        sys.exit(1)
