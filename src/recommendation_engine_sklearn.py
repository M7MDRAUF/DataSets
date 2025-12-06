"""
CineMatch V2.1.6 - Alternative Recommendation Engine (sklearn-compatible)

Core recommendation logic using sklearn-based SVD model.
Windows-compatible alternative to scikit-surprise.

Author: CineMatch Team
Date: October 24, 2025
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
import joblib

# Import data loading functions and model class
sys.path.append(str(Path(__file__).parent.parent))
from src.data_processing import load_ratings, load_movies
from src.svd_model_sklearn import SimpleSVDRecommender

# Model configuration
MODEL_DIR = Path("models")
MODEL_PATH_SKLEARN = MODEL_DIR / "svd_model_sklearn.pkl"
MODEL_PATH_SURPRISE = MODEL_DIR / "svd_model.pkl"


class RecommendationEngine:
    """
    Movie recommendation engine using sklearn-based SVD model.
    Windows-compatible implementation.
    """
    
    def __init__(self):
        self.model = None
        self.ratings_df = None
        self.movies_df = None
        self.user_rated_movies = {}
        self.model_type = None
        
    def load_model(self):
        """Load pre-trained SVD model (try sklearn first, then surprise)"""
        # Try sklearn model first
        if MODEL_PATH_SKLEARN.exists():
            print(f"Loading sklearn-based model from {MODEL_PATH_SKLEARN}...")
            self.model = joblib.load(MODEL_PATH_SKLEARN)
            self.model_type = 'sklearn'
            print("  ✓ sklearn model loaded successfully")
        elif MODEL_PATH_SURPRISE.exists():
            print(f"Loading surprise-based model from {MODEL_PATH_SURPRISE}...")
            self.model = joblib.load(MODEL_PATH_SURPRISE)
            self.model_type = 'surprise'
            print("  ✓ surprise model loaded successfully")
        else:
            raise FileNotFoundError(
                f"Model file not found. Checked:\n"
                f"  - {MODEL_PATH_SKLEARN}\n"
                f"  - {MODEL_PATH_SURPRISE}\n"
                f"Please run model training first:\n"
                f"  python src/model_training_sklearn.py"
            )
        
    def load_data(self, sample_size: Optional[int] = None):
        """Load ratings and movies data"""
        print("Loading dataset...")
        self.ratings_df = load_ratings(sample_size=sample_size)
        self.movies_df = load_movies()
        print("  ✓ Data loaded successfully")
        
    def get_user_history(self, user_id: int) -> pd.DataFrame:
        """Get user's rating history"""
        if user_id in self.user_rated_movies:
            return self.user_rated_movies[user_id]
        
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id].copy()
        user_ratings = user_ratings.merge(
            self.movies_df[['movieId', 'title', 'genres', 'genres_list']],
            on='movieId',
            how='left'
        )
        user_ratings = user_ratings.sort_values(['rating', 'timestamp'], ascending=[False, False])
        
        self.user_rated_movies[user_id] = user_ratings
        return user_ratings
    
    def get_recommendations(
        self,
        user_id: int,
        n: int = 10,
        exclude_rated: bool = True
    ) -> pd.DataFrame:
        """Generate top-N movie recommendations for a user"""
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
            pred_rating = self.model.predict(user_id, movie_id)
            predictions.append({
                'movieId': movie_id,
                'predicted_rating': pred_rating
            })
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        # Sort by predicted rating
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
        """Get "Surprise Me" recommendations"""
        user_history = self.get_user_history(user_id)
        
        # Get user's genre preferences
        if 'genres_list' in user_history.columns:
            user_genres = user_history.explode('genres_list')['genres_list'].value_counts()
            rare_genres = user_genres[user_genres <= user_genres.quantile(0.25)].index.tolist()
        else:
            rare_genres = []
        
        if not rare_genres:
            return self.get_recommendations(user_id, n=n)
        
        # Get movies from rare genres
        rare_genre_movies = self.movies_df[
            self.movies_df['genres_list'].apply(
                lambda x: any(genre in rare_genres for genre in x) if isinstance(x, list) else False
            )
        ]
        
        rated_movie_ids = set(user_history['movieId'].values)
        rare_genre_movies = rare_genre_movies[~rare_genre_movies['movieId'].isin(rated_movie_ids)]
        
        # Predict ratings
        predictions = []
        for _, movie in rare_genre_movies.iterrows():
            pred = self.model.predict(user_id, movie['movieId'])
            predictions.append({
                'movieId': movie['movieId'],
                'predicted_rating': pred,
                'title': movie['title'],
                'genres': movie['genres'],
                'genres_list': movie['genres_list']
            })
        
        surprises = pd.DataFrame(predictions)
        surprises = surprises.sort_values('predicted_rating', ascending=False).head(n)
        
        return surprises
    
    def get_similar_movies(
        self,
        movie_id: int,
        n: int = 10
    ) -> pd.DataFrame:
        """Find movies similar to a given movie using latent factors"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if self.model_type != 'sklearn':
            raise NotImplementedError("Movie similarity only available with sklearn model")
        
        # Check if movie exists
        if movie_id not in self.model.movie_mapper:
            raise ValueError(f"Movie ID {movie_id} not found in trained model")
        
        movie_idx = self.model.movie_mapper[movie_id]
        movie_factors = self.model.movie_factors[movie_idx]
        
        # Calculate similarity with all movies
        similarities = []
        for idx in range(len(self.model.movie_ids)):
            other_movie_id = self.model.movie_inv_mapper[idx]
            
            if other_movie_id == movie_id:
                continue
            
            other_factors = self.model.movie_factors[idx]
            
            # Cosine similarity
            similarity = np.dot(movie_factors, other_factors) / (
                np.linalg.norm(movie_factors) * np.linalg.norm(other_factors)
            )
            
            similarities.append({
                'movieId': other_movie_id,
                'similarity': similarity
            })
        
        # Sort and get top N
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
    """Validate if user ID exists in dataset"""
    if not isinstance(user_id, (int, np.integer)):
        return False, "User ID must be an integer"
    
    if user_id <= 0:
        return False, "User ID must be positive"
    
    if user_id not in ratings_df['userId'].values:
        max_user_id = ratings_df['userId'].max()
        return False, f"User ID {user_id} not found. Valid range: 1 to {max_user_id}"
    
    return True, ""


if __name__ == "__main__":
    """Test the recommendation engine"""
    print("CineMatch V2.1.6 - Recommendation Engine Test (sklearn)\n")
    print("=" * 70)
    
    engine = RecommendationEngine()
    
    try:
        engine.load_model()
        engine.load_data(sample_size=100000)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    test_user_id = 1
    print(f"\n{'='*70}")
    print(f"Testing recommendations for User {test_user_id}")
    print(f"{'='*70}")
    
    try:
        history = engine.get_user_history(test_user_id)
        print(f"\nUser {test_user_id}'s Top Rated Movies:")
        print(history[['title', 'rating', 'genres']].head(5).to_string(index=False))
        
        recommendations = engine.get_recommendations(test_user_id, n=5)
        print(f"\nTop 5 Recommendations for User {test_user_id}:")
        print(recommendations[['title', 'predicted_rating', 'genres']].to_string(index=False))
        
        print("\n✅ Recommendation engine test successful!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        sys.exit(1)
