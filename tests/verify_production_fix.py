"""
Production verification script for CBF dict error fix.
This script simulates the exact production flow to verify the fix works.

Tests:
1. Initialize AlgorithmManager with full data
2. Load Content-Based algorithm (simulates user selecting it)
3. Check cache types after loading
4. Generate recommendations for multiple users
5. Verify all operations succeed without 'dict' errors

Expected: All checks pass, no AttributeError
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.algorithms.algorithm_manager import AlgorithmManager, AlgorithmType
from src.utils.lru_cache import LRUCache

def verify_production_fix():
    """Verify the CBF fix works in production scenario."""
    
    print("=" * 80)
    print("PRODUCTION VERIFICATION TEST - CBF DICT ERROR FIX")
    print("=" * 80)
    
    try:
        # Step 1: Initialize AlgorithmManager (simulates app startup)
        print("\n1. Initializing AlgorithmManager...")
        import pandas as pd
        
        # Load data using the same pattern as the Streamlit app
        ratings_path = Path('data/ml-32m/ratings.csv')
        movies_path = Path('data/ml-32m/movies.csv')
        
        print(f"   Loading ratings from {ratings_path}...")
        ratings_df = pd.read_csv(ratings_path)
        print(f"   Loading movies from {movies_path}...")
        movies_df = pd.read_csv(movies_path)
        
        manager = AlgorithmManager()
        manager.initialize_data(ratings_df, movies_df)
        print("   ✅ AlgorithmManager initialized")
        print(f"   • Ratings: {len(ratings_df):,} rows")
        print(f"   • Movies: {len(movies_df):,} rows")
        
        # Step 2: Load Content-Based algorithm (simulates user selecting CBF)
        print("\n2. Loading Content-Based Filter algorithm...")
        cbf = manager.get_algorithm(AlgorithmType.CONTENT_BASED, fast_load=True, use_joblib=True)
        print(f"   ✅ Algorithm loaded: {cbf.name}")
        print(f"   • is_trained: {cbf.is_trained}")
        
        # Step 3: Verify cache types (CRITICAL CHECK)
        print("\n3. Verifying cache types:")
        print("-" * 80)
        
        up_type = type(cbf.user_profiles)
        up_is_lru = isinstance(cbf.user_profiles, LRUCache)
        
        msc_type = type(cbf.movie_similarity_cache)
        msc_is_lru = isinstance(cbf.movie_similarity_cache, LRUCache)
        
        print(f"   user_profiles: {up_type}")
        print(f"      Is LRUCache: {'✅' if up_is_lru else '❌'} {up_is_lru}")
        print(f"   movie_similarity_cache: {msc_type}")
        print(f"      Is LRUCache: {'✅' if msc_is_lru else '❌'} {msc_is_lru}")
        
        if not (up_is_lru and msc_is_lru):
            print("\n   ❌ CRITICAL: Caches are NOT LRUCache!")
            print("   → This means the cached algorithm instance is from BEFORE the fix")
            print("   → Solution: Clear AlgorithmManager._algorithms cache or restart")
            return False
        
        # Step 4: Test recommendations for multiple users
        print("\n4. Testing recommendations for multiple users:")
        print("-" * 80)
        
        test_users = [1, 10, 100, 500]
        all_passed = True
        
        for user_id in test_users:
            try:
                print(f"   Testing user {user_id}...", end=" ")
                recommendations = cbf.get_recommendations(user_id=user_id, n=5)
                print(f"✅ Got {len(recommendations)} recommendations")
                
                # Verify recommendations structure
                if len(recommendations) > 0:
                    first_rec = recommendations.iloc[0]
                    assert 'movieId' in first_rec.index, "Missing movieId"
                    assert 'predicted_rating' in first_rec.index, "Missing predicted_rating"
                    
            except AttributeError as e:
                if "'dict' object has no attribute" in str(e):
                    print(f"❌ DICT ERROR: {e}")
                    all_passed = False
                else:
                    raise
            except Exception as e:
                print(f"❌ ERROR: {e}")
                all_passed = False
        
        if not all_passed:
            return False
        
        # Step 5: Verify poster URLs work
        print("\n5. Checking poster URLs:")
        print("-" * 80)
        
        try:
            # Get recommendations with posters
            recs_with_posters = cbf.get_recommendations(user_id=1, n=3)
            has_posters = 'poster_path' in recs_with_posters.columns
            print(f"   Has poster_path column: {'✅' if has_posters else '⚠️'} {has_posters}")
            
            if has_posters:
                poster_count = recs_with_posters['poster_path'].notna().sum()
                print(f"   Recommendations with posters: {poster_count}/{len(recs_with_posters)}")
        except Exception as e:
            print(f"   ⚠️ Poster check failed: {e}")
        
        print("\n" + "=" * 80)
        print("✅ ALL PRODUCTION CHECKS PASSED!")
        print("=" * 80)
        print("\nConclusion:")
        print("• Fallback reinitialization is working correctly")
        print("• Caches are properly converted to LRUCache instances")
        print("• get_recommendations() works without 'dict' errors")
        print("• The fix is PRODUCTION READY")
        print("\nIf Streamlit still shows errors:")
        print("1. Restart Streamlit completely (kill all processes)")
        print("2. Clear browser cache (Ctrl+Shift+R)")
        print("3. Check Streamlit session_state is not caching old algorithm")
        
        return True
        
    except Exception as e:
        print(f"\n❌ PRODUCTION VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_production_fix()
    sys.exit(0 if success else 1)
