#!/usr/bin/env python3
"""
CineMatch V2.1.0 - Automated Model Retraining (No Prompts)

Trains all corrupted models without user interaction.

Author: CineMatch Development Team
Date: November 13, 2025
"""

import sys
import time
import pickle
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_processing import load_ratings, load_movies
from src.algorithms.user_knn_recommender import UserKNNRecommender
from src.algorithms.item_knn_recommender import ItemKNNRecommender
from src.algorithms.svd_recommender import SVDRecommender


def log(msg):
    """Print with timestamp"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {msg}")


def save_model(model, filepath, model_name):
    """Save model with verification"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    log(f"Saving {model_name} to {filepath}...")
    start = time.time()
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    size_mb = filepath.stat().st_size / (1024**2)
    save_time = time.time() - start
    
    log(f"Saved {model_name}: {size_mb:.1f} MB in {save_time:.2f}s")
    
    # Verify
    log(f"Verifying {model_name}...")
    verify_start = time.time()
    with open(filepath, 'rb') as f:
        loaded = pickle.load(f)
    verify_time = time.time() - verify_start
    
    log(f"Verified {model_name}: loaded in {verify_time:.2f}s")
    return True


def train_svd_surprise():
    """Train SVD (Surprise)"""
    log("="*80)
    log("[1/4] TRAINING SVD (SURPRISE)")
    log("="*80)
    
    try:
        from surprise import SVD, Dataset, Reader
        
        log("Loading ratings data...")
        ratings_df = load_ratings(sample_size=None)
        log(f"Loaded {len(ratings_df):,} ratings")
        
        log("Converting to Surprise format...")
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
        trainset = data.build_full_trainset()
        
        log("Training SVD model (this will take 30-60 minutes)...")
        start = time.time()
        
        model = SVD(
            n_factors=100,
            n_epochs=20,
            lr_all=0.005,
            reg_all=0.02,
            random_state=42,
            verbose=True
        )
        model.fit(trainset)
        
        train_time = time.time() - start
        log(f"SVD (Surprise) training complete: {train_time/60:.1f} minutes")
        
        save_model(model, "models/svd_model.pkl", "SVD (Surprise)")
        return True, train_time
        
    except Exception as e:
        log(f"ERROR training SVD (Surprise): {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def train_svd_sklearn():
    """Train SVD (sklearn)"""
    log("="*80)
    log("[2/4] TRAINING SVD (SKLEARN)")
    log("="*80)
    
    try:
        log("Loading data...")
        ratings_df = load_ratings(sample_size=None)
        movies_df = load_movies()
        log(f"Loaded {len(ratings_df):,} ratings, {len(movies_df):,} movies")
        
        log("Training SVD (sklearn) model (this will take 20-40 minutes)...")
        start = time.time()
        
        recommender = SVDRecommender(n_components=100)
        recommender.fit(ratings_df, movies_df)
        
        train_time = time.time() - start
        log(f"SVD (sklearn) training complete: {train_time/60:.1f} minutes")
        log(f"RMSE: {recommender.metrics.rmse:.4f}, Coverage: {recommender.metrics.coverage:.1f}%")
        
        save_model(recommender, "models/svd_model_sklearn.pkl", "SVD (sklearn)")
        return True, train_time
        
    except Exception as e:
        log(f"ERROR training SVD (sklearn): {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def train_user_knn():
    """Train User-KNN"""
    log("="*80)
    log("[3/4] TRAINING USER-KNN")
    log("="*80)
    
    try:
        log("Loading data...")
        ratings_df = load_ratings(sample_size=None)
        movies_df = load_movies()
        log(f"Loaded {len(ratings_df):,} ratings, {len(movies_df):,} movies")
        
        log("Training User-KNN model (this will take 20-30 minutes)...")
        start = time.time()
        
        recommender = UserKNNRecommender(k=40, min_support=5)
        recommender.fit(ratings_df, movies_df)
        
        train_time = time.time() - start
        log(f"User-KNN training complete: {train_time/60:.1f} minutes")
        log(f"RMSE: {recommender.metrics.rmse:.4f}, Coverage: {recommender.metrics.coverage:.1f}%")
        
        save_model(recommender, "models/user_knn_model.pkl", "User-KNN")
        return True, train_time
        
    except Exception as e:
        log(f"ERROR training User-KNN: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def train_item_knn():
    """Train Item-KNN"""
    log("="*80)
    log("[4/4] TRAINING ITEM-KNN")
    log("="*80)
    
    try:
        log("Loading data...")
        ratings_df = load_ratings(sample_size=None)
        movies_df = load_movies()
        log(f"Loaded {len(ratings_df):,} ratings, {len(movies_df):,} movies")
        
        log("Training Item-KNN model (this will take 20-30 minutes)...")
        start = time.time()
        
        recommender = ItemKNNRecommender(k=40, min_support=5)
        recommender.fit(ratings_df, movies_df)
        
        train_time = time.time() - start
        log(f"Item-KNN training complete: {train_time/60:.1f} minutes")
        log(f"RMSE: {recommender.metrics.rmse:.4f}, Coverage: {recommender.metrics.coverage:.1f}%")
        
        save_model(recommender, "models/item_knn_model.pkl", "Item-KNN")
        return True, train_time
        
    except Exception as e:
        log(f"ERROR training Item-KNN: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def main():
    log("="*80)
    log("CINEMATCH V2.1.0 - AUTOMATED MODEL RETRAINING")
    log("="*80)
    log("Starting automated retraining of 4 corrupted models...")
    log("Estimated time: 1.5 - 2.5 hours")
    log("="*80)
    
    total_start = time.time()
    results = {}
    
    models = [
        ("SVD (Surprise)", train_svd_surprise),
        ("SVD (sklearn)", train_svd_sklearn),
        ("User-KNN", train_user_knn),
        ("Item-KNN", train_item_knn)
    ]
    
    for model_name, train_func in models:
        success, train_time = train_func()
        results[model_name] = {'success': success, 'time': train_time}
        
        if not success:
            log(f"WARNING: {model_name} failed, continuing with remaining models...")
    
    total_time = time.time() - total_start
    
    log("="*80)
    log("TRAINING SUMMARY")
    log("="*80)
    
    for model_name, result in results.items():
        status = "[OK]" if result['success'] else "[FAIL]"
        time_str = f"{result['time']/60:.1f} min" if result['time'] > 0 else "N/A"
        log(f"{status} {model_name}: {time_str}")
    
    log(f"Total time: {total_time/60:.1f} minutes")
    
    all_success = all(r['success'] for r in results.values())
    if all_success:
        log("="*80)
        log("SUCCESS: ALL MODELS TRAINED SUCCESSFULLY!")
        log("="*80)
    else:
        log("WARNING: Some models failed to train")
    
    log(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
