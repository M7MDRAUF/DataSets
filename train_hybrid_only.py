"""
Train ONLY the Hybrid model using existing pre-trained sub-algorithms.
This should be very fast since sub-algorithms are already trained!

Usage: python train_hybrid_only.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing import load_ratings, load_movies
from src.algorithms.hybrid_recommender import HybridRecommender

def main():
    print("="*80)
    print("🚀 Training ONLY Hybrid Algorithm")
    print("="*80)
    print("This uses your existing pre-trained models, so it should be fast!\n")
    
    # Load data
    print("📊 Loading dataset...")
    ratings_df = load_ratings(sample_size=None)  # Full dataset
    movies_df = load_movies()
    print(f"✓ Loaded {len(ratings_df):,} ratings and {len(movies_df):,} movies\n")
    
    # Train Hybrid
    print("🚀 Training Hybrid Algorithm...")
    print("(This will use your existing SVD, KNN, and Content-Based models)\n")
    
    try:
        hybrid = HybridRecommender()
        hybrid.fit(ratings_df, movies_df)
        hybrid.save_model("models/hybrid_model.pkl")
        
        print("\n" + "="*80)
        print("✅ SUCCESS!")
        print("="*80)
        print(f"Hybrid model saved to: models/hybrid_model.pkl")
        print(f"RMSE: {hybrid.metrics.rmse:.4f}")
        print(f"Coverage: {hybrid.metrics.coverage:.1f}%")
        print(f"Training Time: {hybrid.metrics.training_time:.1f}s")
        print(f"Memory: {hybrid.metrics.memory_usage_mb:.1f} MB")
        print("\n🎉 Your Hybrid algorithm will now load instantly on the website!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
