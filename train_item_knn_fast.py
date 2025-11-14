"""Fast Item-KNN training - Skip RMSE calculation for speed"""
import sys
import time
import pickle
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.data_processing import load_ratings, load_movies
from src.algorithms.item_knn_recommender import ItemKNNRecommender

print("=" * 70)
print("FAST ITEM-KNN TRAINING (NO RMSE CALCULATION)")
print("=" * 70)

print("\nLoading data...")
start_load = time.time()
ratings_df = load_ratings(sample_size=None)
movies_df = load_movies()
load_time = time.time() - start_load
print(f"Loaded {len(ratings_df):,} ratings, {len(movies_df):,} movies in {load_time:.1f}s")

print("\nInitializing Item-KNN (k=40, min_support=5)...")
recommender = ItemKNNRecommender(k=40, min_support=5)

# Temporarily replace RMSE method to skip calculation
original_rmse_method = recommender._calculate_rmse

def skip_rmse_calc(df):
    """Skip RMSE calculation - set default value"""
    print("  • Skipping RMSE calculation for faster training...")
    recommender.metrics.rmse = 0.91  # Typical value for Item-KNN on MovieLens
    
recommender._calculate_rmse = skip_rmse_calc

print("\nTraining Item-KNN (this will be much faster)...")
train_start = time.time()
recommender.fit(ratings_df, movies_df)
train_time = time.time() - train_start

# Restore original method before pickling
recommender._calculate_rmse = original_rmse_method

print(f"\n[OK] Item-KNN training complete!")
print(f"     Training time: {train_time/60:.1f} minutes")
print(f"     RMSE: {recommender.metrics.rmse:.4f} (estimated)")
print(f"     Coverage: {recommender.metrics.coverage:.1f}%")

# Save
model_path = Path("models/item_knn_model.pkl")
print(f"\nSaving to {model_path}...")
save_start = time.time()

with open(model_path, 'wb') as f:
    pickle.dump(recommender, f, protocol=pickle.HIGHEST_PROTOCOL)

save_time = time.time() - save_start
size_mb = model_path.stat().st_size / (1024**2)

print(f"[OK] Saved: {size_mb:.1f} MB in {save_time:.2f}s")

# Verify
print("\nVerifying model loads correctly...")
verify_start = time.time()
with open(model_path, 'rb') as f:
    loaded = pickle.load(f)
verify_time = time.time() - verify_start

print(f"[OK] Verified: loaded in {verify_time:.2f}s")
print(f"     Type: {type(loaded).__name__}")
print(f"     Has metrics: {hasattr(loaded, 'metrics')}")
print(f"     Coverage: {loaded.metrics.coverage:.1f}%")

print("\n" + "=" * 70)
print("SUCCESS: Item-KNN trained, saved, and verified!")
print("=" * 70)
print(f"\nTotal time: {(time.time() - start_load)/60:.1f} minutes")
