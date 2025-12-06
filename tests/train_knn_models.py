#!/usr/bin/env python3
"""
CineMatch - KNN Models Pre-training Script

This script trains both User-based and Item-based KNN models on the full 
MovieLens 32M dataset and saves them as .pkl files for fast loading.

Usage:
    python train_knn_models.py

Author: CineMatch Development Team
Date: November 7, 2025
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_processing import load_ratings, load_movies
from src.algorithms.user_knn_recommender import UserKNNRecommender
from src.algorithms.item_knn_recommender import ItemKNNRecommender


def train_and_save_knn_models():
    """Train both KNN models on full dataset and save them"""
    
    print("🎬 CineMatch KNN Models Training Script")
    print("=" * 50)
    print("This will train both User and Item KNN models on the FULL dataset")
    print("⚠️  Warning: This may take 1-2 hours and use significant RAM!")
    print()
    
    # Confirm before starting
    response = input("Continue? (y/N): ").strip().lower()
    if response != 'y':
        print("Training cancelled.")
        return
    
    print("\n📊 Loading MovieLens 32M dataset...")
    start_time = time.time()
    
    try:
        # Load FULL dataset (no sampling)
        print("Loading ratings (this may take a few minutes)...")
        ratings_df = load_ratings(sample_size=None)  # Full dataset
        print("Loading movies...")
        movies_df = load_movies()
        
        load_time = time.time() - start_time
        print(f"✅ Data loaded in {load_time:.1f}s")
        print(f"   • Ratings: {len(ratings_df):,}")
        print(f"   • Movies: {len(movies_df):,}")
        print(f"   • Users: {ratings_df['userId'].nunique():,}")
        print()
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Make sure the ml-32m dataset is properly extracted in data/ml-32m/")
        return
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Train User KNN
    print("🔥 Training User-based KNN...")
    print("=" * 30)
    user_knn = UserKNNRecommender(
        n_neighbors=50,  # Optimized for full dataset
        similarity_metric='cosine'
    )
    
    try:
        user_knn_start = time.time()
        user_knn.fit(ratings_df, movies_df)
        user_knn_time = time.time() - user_knn_start
        
        # Save model
        user_knn_path = models_dir / "user_knn_model.pkl"
        user_knn.save_model(user_knn_path)
        
        print(f"✅ User KNN trained and saved in {user_knn_time:.1f}s")
        print(f"   • Model saved to: {user_knn_path}")
        print(f"   • RMSE: {user_knn.metrics.rmse:.4f}")
        print()
        
    except Exception as e:
        print(f"❌ Error training User KNN: {e}")
        print()
    
    # Train Item KNN  
    print("🎭 Training Item-based KNN...")
    print("=" * 30)
    item_knn = ItemKNNRecommender(
        n_neighbors=30,  # Optimized for full dataset
        similarity_metric='cosine',
        min_ratings=10  # Higher threshold for full dataset
    )
    
    try:
        item_knn_start = time.time()
        item_knn.fit(ratings_df, movies_df)
        item_knn_time = time.time() - item_knn_start
        
        # Save model
        item_knn_path = models_dir / "item_knn_model.pkl" 
        item_knn.save_model(item_knn_path)
        
        print(f"✅ Item KNN trained and saved in {item_knn_time:.1f}s")
        print(f"   • Model saved to: {item_knn_path}")
        print(f"   • RMSE: {item_knn.metrics.rmse:.4f}")
        print()
        
    except Exception as e:
        print(f"❌ Error training Item KNN: {e}")
        print()
    
    total_time = time.time() - start_time
    print("🎉 Training Complete!")
    print("=" * 30)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print()
    print("📦 Saved models:")
    print(f"   • User KNN: models/user_knn_model.pkl")
    print(f"   • Item KNN: models/item_knn_model.pkl")
    print()
    print("🚀 Now KNN models will load instantly in CineMatch!")
    

def check_models_exist():
    """Check if pre-trained models exist"""
    models_dir = Path("models")
    user_knn_path = models_dir / "user_knn_model.pkl"
    item_knn_path = models_dir / "item_knn_model.pkl"
    
    print("📦 Checking existing models...")
    
    if user_knn_path.exists():
        size_mb = user_knn_path.stat().st_size / (1024 * 1024)
        print(f"✅ User KNN model exists ({size_mb:.1f} MB)")
    else:
        print("❌ User KNN model not found")
    
    if item_knn_path.exists():
        size_mb = item_knn_path.stat().st_size / (1024 * 1024)
        print(f"✅ Item KNN model exists ({size_mb:.1f} MB)")
    else:
        print("❌ Item KNN model not found")
    
    return user_knn_path.exists() and item_knn_path.exists()


if __name__ == "__main__":
    print("🎬 CineMatch KNN Model Trainer")
    print()
    
    # Check if models already exist
    if check_models_exist():
        print()
        response = input("Pre-trained models found. Re-train anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("Using existing models. Run with 'y' to retrain.")
            sys.exit(0)
        print()
    
    # Train models
    train_and_save_knn_models()