#!/usr/bin/env python3
"""
Quick test script to verify pre-trained KNN models load fast
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.algorithms.user_knn_recommender import UserKNNRecommender
from src.algorithms.item_knn_recommender import ItemKNNRecommender

def test_model_loading():
    """Test loading speed of pre-trained models"""
    
    print("🧪 Testing Pre-trained KNN Model Loading Speed")
    print("=" * 50)
    
    models_dir = Path("models")
    
    # Test User KNN
    user_knn_path = models_dir / "user_knn_model.pkl"
    if user_knn_path.exists():
        print("👥 Testing User KNN loading...")
        user_knn = UserKNNRecommender()
        
        start_time = time.time()
        user_knn.load_model(user_knn_path)
        load_time = time.time() - start_time
        
        print(f"✅ User KNN loaded in {load_time:.3f}s")
        print(f"   • Trained: {user_knn.is_trained}")
        print(f"   • RMSE: {user_knn.metrics.rmse:.4f}")
        print()
    else:
        print("❌ User KNN model not found")
    
    # Test Item KNN  
    item_knn_path = models_dir / "item_knn_model.pkl"
    if item_knn_path.exists():
        print("🎬 Testing Item KNN loading...")
        item_knn = ItemKNNRecommender()
        
        start_time = time.time()
        item_knn.load_model(item_knn_path)
        load_time = time.time() - start_time
        
        print(f"✅ Item KNN loaded in {load_time:.3f}s")
        print(f"   • Trained: {item_knn.is_trained}")
        print(f"   • RMSE: {item_knn.metrics.rmse:.4f}")
        print()
    else:
        print("❌ Item KNN model not found")
    
    print("🎉 Speed test complete!")
    print("🚀 Models should now load in seconds instead of minutes!")

if __name__ == "__main__":
    test_model_loading()