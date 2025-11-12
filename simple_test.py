"""
Simple test script to verify individual algorithms work correctly.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.algorithms.svd_recommender import SVDRecommender
from src.algorithms.user_knn_recommender import UserKNNRecommender  
from src.algorithms.item_knn_recommender import ItemKNNRecommender
from src.data_processing import load_ratings, load_movies


def simple_test():
    print("🎬 Simple Algorithm Test")
    print("=" * 40)
    
    # Load small dataset
    print("📊 Loading data...")
    ratings_df = load_ratings(sample_size=50_000)
    movies_df = load_movies()
    print(f"✅ Loaded {len(ratings_df):,} ratings and {len(movies_df):,} movies")
    
    # Get valid user
    valid_user = ratings_df['userId'].iloc[0]
    print(f"📝 Testing with User ID: {valid_user}")
    
    # Test SVD
    print(f"\n🔮 Testing SVD...")
    svd = SVDRecommender(n_components=50)
    svd.fit(ratings_df, movies_df)
    svd_recs = svd.get_recommendations(valid_user, n=5)
    print(f"✅ SVD: {len(svd_recs)} recommendations, RMSE: {svd.metrics.rmse:.4f}")
    
    # Test User KNN
    print(f"\n👥 Testing User KNN...")
    user_knn = UserKNNRecommender(n_neighbors=20)
    user_knn.fit(ratings_df, movies_df) 
    user_knn_recs = user_knn.get_recommendations(valid_user, n=5)
    print(f"✅ User KNN: {len(user_knn_recs)} recommendations, RMSE: {user_knn.metrics.rmse:.4f}")
    
    # Test Item KNN
    print(f"\n🎬 Testing Item KNN...")
    item_knn = ItemKNNRecommender(n_neighbors=15, min_ratings=3)
    item_knn.fit(ratings_df, movies_df)
    item_knn_recs = item_knn.get_recommendations(valid_user, n=5)
    print(f"✅ Item KNN: {len(item_knn_recs)} recommendations, RMSE: {item_knn.metrics.rmse:.4f}")
    
    print(f"\n🎉 All individual algorithms working!")
    return True


if __name__ == "__main__":
    simple_test()