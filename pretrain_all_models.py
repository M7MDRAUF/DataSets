"""
Pre-train all recommendation algorithms and save them to disk.
Run this script ONCE to train all models, so the web app loads them instantly.

Usage: python pretrain_all_models.py
"""

import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing import load_ratings, load_movies
from src.algorithms.svd_recommender import SVDRecommender
from src.algorithms.user_knn_recommender import UserKNNRecommender
from src.algorithms.item_knn_recommender import ItemKNNRecommender
from src.algorithms.content_based_recommender import ContentBasedRecommender
from src.algorithms.hybrid_recommender import HybridRecommender

def main():
    """Pre-train all algorithms and save models"""
    
    print("="*80)
    print("🚀 CineMatch V2.0 - Pre-Training All Algorithms")
    print("="*80)
    print("\nThis will train all 5 algorithms and save them to the models/ directory.")
    print("The web app will then load pre-trained models instantly!\n")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print(f"✓ Models directory: {models_dir.absolute()}\n")
    
    # Load data
    print("📊 Loading MovieLens dataset...")
    print("-" * 80)
    
    try:
        # Use sample_size=None for full dataset, or specify a number for testing
        sample_size = None  # Full dataset
        # sample_size = 100000  # Use this for faster testing
        
        ratings_df = load_ratings(sample_size=sample_size)
        movies_df = load_movies()
        
        print(f"✓ Loaded {len(ratings_df):,} ratings")
        print(f"✓ Loaded {len(movies_df):,} movies")
        print()
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Track total time
    total_start = time.time()
    
    # 1. Train SVD
    print("="*80)
    print("1️⃣  TRAINING SVD MATRIX FACTORIZATION")
    print("="*80)
    try:
        svd_start = time.time()
        svd = SVDRecommender(n_components=100)
        svd.fit(ratings_df, movies_df)
        svd.save_model("models/svd_model.pkl")
        svd_time = time.time() - svd_start
        print(f"✅ SVD trained and saved in {svd_time:.1f}s")
        print(f"   RMSE: {svd.metrics.rmse:.4f}")
        print(f"   Coverage: {svd.metrics.coverage:.1f}%")
        print(f"   Memory: {svd.metrics.memory_usage_mb:.1f} MB\n")
    except Exception as e:
        print(f"❌ SVD training failed: {e}\n")
    
    # 2. Train User KNN
    print("="*80)
    print("2️⃣  TRAINING USER-BASED KNN")
    print("="*80)
    try:
        user_knn_start = time.time()
        user_knn = UserKNNRecommender(n_neighbors=50, similarity_metric='cosine')
        user_knn.fit(ratings_df, movies_df)
        user_knn.save_model("models/user_knn_model.pkl")
        user_knn_time = time.time() - user_knn_start
        print(f"✅ User KNN trained and saved in {user_knn_time:.1f}s")
        print(f"   RMSE: {user_knn.metrics.rmse:.4f}")
        print(f"   Coverage: {user_knn.metrics.coverage:.1f}%")
        print(f"   Memory: {user_knn.metrics.memory_usage_mb:.1f} MB\n")
    except Exception as e:
        print(f"❌ User KNN training failed: {e}\n")
    
    # 3. Train Item KNN
    print("="*80)
    print("3️⃣  TRAINING ITEM-BASED KNN")
    print("="*80)
    try:
        item_knn_start = time.time()
        item_knn = ItemKNNRecommender(n_neighbors=30, min_ratings=5)
        item_knn.fit(ratings_df, movies_df)
        item_knn.save_model("models/item_knn_model.pkl")
        item_knn_time = time.time() - item_knn_start
        print(f"✅ Item KNN trained and saved in {item_knn_time:.1f}s")
        print(f"   RMSE: {item_knn.metrics.rmse:.4f}")
        print(f"   Coverage: {item_knn.metrics.coverage:.1f}%")
        print(f"   Memory: {item_knn.metrics.memory_usage_mb:.1f} MB\n")
    except Exception as e:
        print(f"❌ Item KNN training failed: {e}\n")
    
    # 4. Train Content-Based
    print("="*80)
    print("4️⃣  TRAINING CONTENT-BASED FILTERING")
    print("="*80)
    try:
        cb_start = time.time()
        cb = ContentBasedRecommender()
        cb.fit(ratings_df, movies_df)
        cb.save_model("models/content_based_model.pkl")
        cb_time = time.time() - cb_start
        print(f"✅ Content-Based trained and saved in {cb_time:.1f}s")
        print(f"   RMSE: {cb.metrics.rmse:.4f}")
        print(f"   Coverage: {cb.metrics.coverage:.1f}%")
        print(f"   Memory: {cb.metrics.memory_usage_mb:.1f} MB\n")
    except Exception as e:
        print(f"❌ Content-Based training failed: {e}\n")
    
    # 5. Train Hybrid
    print("="*80)
    print("5️⃣  TRAINING HYBRID ALGORITHM")
    print("="*80)
    try:
        hybrid_start = time.time()
        hybrid = HybridRecommender()
        hybrid.fit(ratings_df, movies_df)
        hybrid.save_model("models/hybrid_model.pkl")
        hybrid_time = time.time() - hybrid_start
        print(f"✅ Hybrid trained and saved in {hybrid_time:.1f}s")
        print(f"   RMSE: {hybrid.metrics.rmse:.4f}")
        print(f"   Coverage: {hybrid.metrics.coverage:.1f}%")
        print(f"   Memory: {hybrid.metrics.memory_usage_mb:.1f} MB\n")
    except Exception as e:
        print(f"❌ Hybrid training failed: {e}\n")
    
    # Summary
    total_time = time.time() - total_start
    print("="*80)
    print("🎉 PRE-TRAINING COMPLETE!")
    print("="*80)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"\nAll models saved to: {models_dir.absolute()}")
    print("\n✅ Your web app will now load these pre-trained models instantly!")
    print("   No more waiting for training when users visit the site.\n")
    
    # Show saved files
    print("📁 Saved model files:")
    for model_file in sorted(models_dir.glob("*.pkl")):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"   • {model_file.name} ({size_mb:.1f} MB)")
    print()

if __name__ == "__main__":
    main()
