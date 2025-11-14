"""Quick script to train only Item-KNN model"""
import sys
import time
import pickle
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.data_processing import load_ratings, load_movies
from src.algorithms.item_knn_recommender import ItemKNNRecommender

print("=" * 70)
print("TRAINING ITEM-KNN MODEL")
print("=" * 70)

print("\nLoading data...")
ratings_df = load_ratings(sample_size=None)
movies_df = load_movies()
print(f"Loaded {len(ratings_df):,} ratings, {len(movies_df):,} movies")

print("\nTraining Item-KNN (k=40, min_support=5)...")
start = time.time()

recommender = ItemKNNRecommender(k=40, min_support=5)
recommender.fit(ratings_df, movies_df)

train_time = time.time() - start

print(f"\n[OK] Item-KNN training complete!")
print(f"     Training time: {train_time/60:.1f} minutes")
print(f"     RMSE: {recommender.metrics.rmse:.4f}")
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
print("\nVerifying...")
verify_start = time.time()
with open(model_path, 'rb') as f:
    loaded = pickle.load(f)
verify_time = time.time() - verify_start

print(f"[OK] Verified: loaded in {verify_time:.2f}s")
print("\n" + "=" * 70)
print("SUCCESS: Item-KNN model trained and saved!")
print("=" * 70)
