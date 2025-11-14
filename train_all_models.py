#!/usr/bin/env python3
"""
CineMatch V2.1.0 - Complete Model Retraining Script

Trains all corrupted models (SVD Surprise, SVD sklearn, User-KNN, Item-KNN)
on the full MovieLens 32M dataset and saves them for production use.

This script will:
1. Train SVD model using surprise library (svd_model.pkl)
2. Train SVD model using sklearn (svd_model_sklearn.pkl) 
3. Train User-based KNN model (user_knn_model.pkl)
4. Train Item-based KNN model (item_knn_model.pkl)

Usage:
    python train_all_models.py

Author: CineMatch Development Team
Date: November 13, 2025
"""

import sys
import time
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_processing import load_ratings, load_movies
from src.algorithms.user_knn_recommender import UserKNNRecommender
from src.algorithms.item_knn_recommender import ItemKNNRecommender
from src.algorithms.svd_recommender import SVDRecommender


def print_header(message: str):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {message}")
    print("=" * 80)


def print_section(message: str):
    """Print formatted section header"""
    print(f"\n{'-' * 80}")
    print(f"  {message}")
    print(f"{'-' * 80}")


def save_model(model, filepath: Path, model_name: str):
    """Save model with error handling and verification"""
    print(f"\nSaving {model_name}...")
    
    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with highest protocol
        start_time = time.time()
        with open(filepath, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        save_time = time.time() - start_time
        
        # Get file size
        size_mb = filepath.stat().st_size / (1024 * 1024)
        
        print(f"[OK] Saved successfully!")
        print(f"   • Path: {filepath}")
        print(f"   • Size: {size_mb:.1f} MB")
        print(f"   • Save time: {save_time:.2f}s")
        
        # Verify by loading
        print(f"Verifying {model_name}...")
        verify_start = time.time()
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)
        verify_time = time.time() - verify_start
        
        print(f"[OK] Verification successful!")
        print(f"   • Load time: {verify_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to save {model_name}: {e}")
        return False


def train_svd_surprise():
    """Train SVD model using surprise library"""
    print_section("[1/4] Training SVD (Surprise Library)")
    
    try:
        from surprise import SVD, Dataset, Reader
        from surprise.model_selection import train_test_split
        
        print("Loading data for Surprise format...")
        ratings_df = load_ratings(sample_size=None)
        
        # Convert to Surprise format
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(
            ratings_df[['userId', 'movieId', 'rating']], 
            reader
        )
        
        print(f"Training on {len(ratings_df):,} ratings...")
        print("[WARNING] This may take 30-60 minutes...")
        
        start_time = time.time()
        
        # Build full training set
        trainset = data.build_full_trainset()
        
        # Train model
        model = SVD(
            n_factors=100,
            n_epochs=20,
            lr_all=0.005,
            reg_all=0.02,
            random_state=42,
            verbose=True
        )
        
        print("\nTraining SVD model...")
        model.fit(trainset)
        
        training_time = time.time() - start_time
        
        print(f"\n[OK] SVD (Surprise) training complete!")
        print(f"   • Training time: {training_time/60:.1f} minutes")
        print(f"   • Factors: 100")
        print(f"   • Epochs: 20")
        
        # Save model
        model_path = Path("models/svd_model.pkl")
        if save_model(model, model_path, "SVD (Surprise)"):
            return True, training_time
        else:
            return False, training_time
            
    except ImportError as e:
        print(f"[ERROR] scikit-surprise not installed: {e}")
        print("   Install with: pip install scikit-surprise")
        return False, 0
    except Exception as e:
        print(f"[ERROR] Error training SVD (Surprise): {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def train_svd_sklearn():
    """Train SVD model using sklearn (SimpleSVDRecommender)"""
    print_section("[2/4] Training SVD (sklearn Implementation)")
    
    try:
        print("Loading data...")
        ratings_df = load_ratings(sample_size=None)
        movies_df = load_movies()
        
        print(f"Training on {len(ratings_df):,} ratings...")
        print("[WARNING] This may take 20-40 minutes...")
        
        start_time = time.time()
        
        # Initialize and train
        recommender = SVDRecommender(n_components=100)
        recommender.fit(ratings_df, movies_df)
        
        training_time = time.time() - start_time
        
        print(f"\n[OK] SVD (sklearn) training complete!")
        print(f"   • Training time: {training_time/60:.1f} minutes")
        print(f"   • RMSE: {recommender.metrics.rmse:.4f}")
        print(f"   • Coverage: {recommender.metrics.coverage:.1f}%")
        
        # Save model
        model_path = Path("models/svd_model_sklearn.pkl")
        if save_model(recommender, model_path, "SVD (sklearn)"):
            return True, training_time
        else:
            return False, training_time
            
    except Exception as e:
        print(f"[ERROR] Error training SVD (sklearn): {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def train_user_knn():
    """Train User-based KNN model"""
    print_section("[3/4] Training User-based KNN")
    
    try:
        print("Loading data...")
        ratings_df = load_ratings(sample_size=None)
        movies_df = load_movies()
        
        print(f"Training on {len(ratings_df):,} ratings...")
        print("  This may take 20-30 minutes...")
        
        start_time = time.time()
        
        # Initialize and train
        recommender = UserKNNRecommender(k=40, min_support=5)
        recommender.fit(ratings_df, movies_df)
        
        training_time = time.time() - start_time
        
        print(f"\n[OK] User-KNN training complete!")
        print(f"   • Training time: {training_time/60:.1f} minutes")
        print(f"   • RMSE: {recommender.metrics.rmse:.4f}")
        print(f"   • Coverage: {recommender.metrics.coverage:.1f}%")
        print(f"   • k neighbors: 40")
        
        # Save model
        model_path = Path("models/user_knn_model.pkl")
        if save_model(recommender, model_path, "User-KNN"):
            return True, training_time
        else:
            return False, training_time
            
    except Exception as e:
        print(f"[ERROR] Error training User-KNN: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def train_item_knn():
    """Train Item-based KNN model"""
    print_section("[4/4] Training Item-based KNN")
    
    try:
        print("Loading data...")
        ratings_df = load_ratings(sample_size=None)
        movies_df = load_movies()
        
        print(f"Training on {len(ratings_df):,} ratings...")
        print("  This may take 20-30 minutes...")
        
        start_time = time.time()
        
        # Initialize and train
        recommender = ItemKNNRecommender(k=40, min_support=5)
        recommender.fit(ratings_df, movies_df)
        
        training_time = time.time() - start_time
        
        print(f"\n[OK] Item-KNN training complete!")
        print(f"   • Training time: {training_time/60:.1f} minutes")
        print(f"   • RMSE: {recommender.metrics.rmse:.4f}")
        print(f"   • Coverage: {recommender.metrics.coverage:.1f}%")
        print(f"   • k neighbors: 40")
        
        # Save model
        model_path = Path("models/item_knn_model.pkl")
        if save_model(recommender, model_path, "Item-KNN"):
            return True, training_time
        else:
            return False, training_time
            
    except Exception as e:
        print(f"[ERROR] Error training Item-KNN: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def main():
    """Main training orchestration"""
    # Set UTF-8 encoding for Windows console
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    print_header("CineMatch V2.1.0 - Complete Model Retraining")
    
    print("\nThis script will retrain all 4 corrupted models:")
    print("  1. SVD (Surprise) - svd_model.pkl")
    print("  2. SVD (sklearn) - svd_model_sklearn.pkl")
    print("  3. User-KNN - user_knn_model.pkl")
    print("  4. Item-KNN - item_knn_model.pkl")
    print("\n  Total estimated time: 1.5 - 2.5 hours")
    print("  Will use significant RAM (recommend 16GB+ available)")
    print("\nNote: Content-Based model is already working and won't be retrained.")
    
    # Confirm
    print("\n" + "-" * 80)
    response = input("Continue with training? (y/N): ").strip().lower()
    if response != 'y':
        print("\n Training cancelled.")
        return
    
    # Track results
    results = {}
    total_start = time.time()
    
    # Train each model
    models_to_train = [
        ("SVD (Surprise)", train_svd_surprise),
        ("SVD (sklearn)", train_svd_sklearn),
        ("User-KNN", train_user_knn),
        ("Item-KNN", train_item_knn)
    ]
    
    for model_name, train_func in models_to_train:
        success, train_time = train_func()
        results[model_name] = {
            'success': success,
            'time': train_time
        }
        
        if not success:
            print(f"\n  {model_name} training failed. Continue with remaining models? (y/N): ", end='')
            cont = input().strip().lower()
            if cont != 'y':
                print("\n Training stopped.")
                break
    
    # Final summary
    total_time = time.time() - total_start
    
    print_header(" TRAINING SUMMARY")
    
    print("\nResults:")
    for model_name, result in results.items():
        status = " SUCCESS" if result['success'] else " FAILED"
        time_str = f"{result['time']/60:.1f} min" if result['time'] > 0 else "N/A"
        print(f"  {status} - {model_name}: {time_str}")
    
    print(f"\n⏱  Total execution time: {total_time/60:.1f} minutes")
    
    # Check if all succeeded
    all_success = all(r['success'] for r in results.values())
    
    if all_success:
        print("\n" + "=" * 80)
        print(" ALL MODELS TRAINED SUCCESSFULLY!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Verify models load correctly with validation script")
        print("  2. Test CineMatch application with all 5 algorithms")
        print("  3. Fix Git LFS configuration to prevent future corruption")
        print("  4. Deploy to production")
    else:
        print("\n  Some models failed to train. Please check errors above.")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
