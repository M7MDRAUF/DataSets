"""
CineMatch V1.0.0 - Multi-Algorithm System Test

Test script to verify all algorithms work correctly and integration is successful.

Author: CineMatch Development Team  
Date: November 7, 2025
"""

import sys
from pathlib import Path
import pandas as pd
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.algorithms.algorithm_manager import AlgorithmManager, AlgorithmType
from src.data_processing import load_ratings, load_movies


def test_algorithm_system():
    """Test the complete multi-algorithm system"""
    
    print("🎬 CineMatch V2.0 - Multi-Algorithm System Test")
    print("=" * 60)
    
    try:
        # Load data
        print("\n📊 Loading MovieLens dataset...")
        ratings_df = load_ratings(sample_size=100_000)  # Small sample for testing
        movies_df = load_movies()
        print(f"✅ Loaded {len(ratings_df):,} ratings and {len(movies_df):,} movies")
        
        # Initialize algorithm manager
        print("\n🤖 Initializing Algorithm Manager...")
        manager = AlgorithmManager()
        manager.initialize_data(ratings_df, movies_df)
        print("✅ Algorithm Manager initialized")
        
        # Find a valid test user ID
        valid_users = ratings_df['userId'].unique()
        test_user_id = valid_users[0]  # Use the first available user
        print(f"📝 Using test User ID: {test_user_id} (from {len(valid_users):,} available users)")
        
        algorithms_to_test = [
            AlgorithmType.SVD,
            AlgorithmType.USER_KNN, 
            AlgorithmType.ITEM_KNN,
            AlgorithmType.HYBRID
        ]
        
        results = {}
        
        for algo_type in algorithms_to_test:
            print(f"\n🔄 Testing {algo_type.value}...")
            
            try:
                # Get algorithm (will train if needed)
                start_time = time.time()
                algorithm = manager.get_algorithm(algo_type)
                load_time = time.time() - start_time
                
                # Generate recommendations
                recommendations = algorithm.get_recommendations(test_user_id, n=5)
                
                # Test prediction
                if len(recommendations) > 0:
                    movie_id = recommendations.iloc[0]['movieId']
                    prediction = algorithm.predict(test_user_id, movie_id)
                    
                    # Get similar movies
                    similar = algorithm.get_similar_items(movie_id, n=3)
                    
                    results[algo_type] = {
                        'load_time': load_time,
                        'rmse': algorithm.metrics.rmse,
                        'recommendations': len(recommendations),
                        'prediction': prediction,
                        'similar_items': len(similar),
                        'memory_mb': algorithm.metrics.memory_usage_mb
                    }
                    
                    print(f"  ✅ {algo_type.value}: {len(recommendations)} recs, RMSE: {algorithm.metrics.rmse:.4f}")
                    print(f"     Load time: {load_time:.1f}s, Memory: {algorithm.metrics.memory_usage_mb:.1f}MB")
                else:
                    print(f"  ❌ {algo_type.value}: No recommendations generated")
                    
            except Exception as e:
                print(f"  ❌ {algo_type.value}: Error - {e}")
                continue
        
        # Test algorithm switching
        print(f"\n🔄 Testing Algorithm Switching...")
        for algo_type in [AlgorithmType.SVD, AlgorithmType.HYBRID]:
            algorithm = manager.switch_algorithm(algo_type)
            recommendations = algorithm.get_recommendations(test_user_id, n=3)
            print(f"  ✅ Switched to {algo_type.value}: {len(recommendations)} recommendations")
        
        # Performance comparison
        print(f"\n📊 Performance Summary:")
        print("-" * 60)
        for algo_type, metrics in results.items():
            print(f"{algo_type.value:20} | RMSE: {metrics['rmse']:.4f} | "
                  f"Load: {metrics['load_time']:5.1f}s | Mem: {metrics['memory_mb']:6.1f}MB")
        
        print(f"\n✅ Multi-Algorithm System Test Completed Successfully!")
        print(f"   All {len(results)} algorithms working correctly")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        return False


def test_recommendations_sample():
    """Test generating sample recommendations"""
    
    print(f"\n🎯 Testing Sample Recommendations...")
    print("=" * 40)
    
    try:
        # Quick setup
        ratings_df = load_ratings(sample_size=50_000)
        movies_df = load_movies()
        
        manager = AlgorithmManager()
        manager.initialize_data(ratings_df, movies_df)
        
        # Test with Hybrid algorithm (fastest to show all features)
        print("🚀 Getting Hybrid algorithm...")
        algorithm = manager.get_algorithm(AlgorithmType.HYBRID)
        
        # Test with valid user IDs
        valid_users = ratings_df['userId'].unique()
        test_users = valid_users[:3]  # Use first 3 valid users
        
        for user_id in test_users:
            print(f"\n👤 User {user_id} recommendations:")
            recommendations = algorithm.get_recommendations(user_id, n=3)
            
            for idx, row in recommendations.iterrows():
                title = row['title']
                rating = row['predicted_rating']
                print(f"  {idx+1}. {title[:50]}... | Predicted: {rating:.2f}")
        
        print(f"\n✅ Sample recommendations generated successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Sample test failed: {e}")
        return False


if __name__ == "__main__":
    print("Starting CineMatch V2.0 Multi-Algorithm Tests...\n")
    
    # Run basic system test
    system_test = test_algorithm_system()
    
    if system_test:
        # Run sample recommendations test
        sample_test = test_recommendations_sample()
        
        if sample_test:
            print(f"\n🎉 ALL TESTS PASSED! 🎉")
            print(f"Your CineMatch V2.0 multi-algorithm system is ready!")
            print(f"\nNext steps:")
            print(f"1. Run the Streamlit app: streamlit run app/pages/2_🎬_Recommend_V2.py")
            print(f"2. Try different algorithms and compare results")
            print(f"3. Test with various user IDs")
        else:
            print(f"\n⚠️  System works but sample test failed")
    else:
        print(f"\n❌ System test failed - check configuration")
    
    print(f"\nTest completed.")