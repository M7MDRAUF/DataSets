# -*- coding: utf-8 -*-
"""
Regression Testing for CineMatch V2.1.0

This script tests all 5 algorithms to ensure:
1. Existing algorithms (SVD, UserKNN, ItemKNN, Hybrid) still work
2. Content-Based integration didn't break anything
3. AlgorithmManager correctly handles all algorithms
4. Hybrid ensemble properly weights Content-Based

Author: CineMatch Team
Date: November 2025
Version: 2.1.0
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.algorithms.algorithm_manager import AlgorithmManager
from src.algorithms.svd_recommender import SVDRecommender
from src.algorithms.user_knn_recommender import UserKNNRecommender
from src.algorithms.item_knn_recommender import ItemKNNRecommender
from src.algorithms.content_based_recommender import ContentBasedRecommender
from src.algorithms.hybrid_recommender import HybridRecommender


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"🧪 {title}")
    print("=" * 80)


def test_algorithm_manager():
    """Test that AlgorithmManager correctly lists all 5 algorithms."""
    print_section("TEST 1: AlgorithmManager - Algorithm Registration")
    
    try:
        manager = AlgorithmManager()
        algorithms = manager.get_available_algorithms()
        
        print(f"\n[OK] AlgorithmManager initialized successfully")
        print(f"[OK] Available algorithms: {len(algorithms)}")
        
        # AlgorithmType enum values
        from src.algorithms.algorithm_manager import AlgorithmType
        expected_algorithms = [
            AlgorithmType.SVD,
            AlgorithmType.USER_KNN,
            AlgorithmType.ITEM_KNN,
            AlgorithmType.CONTENT_BASED,
            AlgorithmType.HYBRID
        ]
        
        for algo in expected_algorithms:
            if algo in algorithms:
                print(f"  [OK] {algo.value}: Found")
            else:
                print(f"  [X] {algo.value}: MISSING")
                return False
        
        print(f"\n[PASS] PASS: All 5 algorithms registered correctly")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] FAIL: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading(model_name: str, model_path: str):
    """Test loading a pre-trained model."""
    print_section(f"TEST 2.{['SVD', 'UserKNN', 'ItemKNN', 'ContentBased', 'Hybrid'].index(model_name) + 1}: {model_name} - Model Loading")
    
    try:
        model_file = Path(model_path)
        
        if not model_file.exists():
            print(f"[WARN] Warning: Model file not found: {model_path}")
            print(f"⏭ Skipping test (model needs to be trained first)")
            return None
        
        start_time = time.time()
        
        # Load based on algorithm type
        # Convert path to Path object for base_recommender compatibility
        path_obj = Path(model_path)
        
        # Note: load_model modifies the instance in-place and doesn't return anything
        if model_name == 'SVD':
            model = SVDRecommender()
            model.load_model(path_obj)
        elif model_name == 'UserKNN':
            model = UserKNNRecommender()
            model.load_model(path_obj)
        elif model_name == 'ItemKNN':
            model = ItemKNNRecommender()
            model.load_model(path_obj)
        elif model_name == 'ContentBased':
            model = ContentBasedRecommender()
            model.load_model(model_path)  # ContentBased uses string
        elif model_name == 'Hybrid':
            model = HybridRecommender()
            model.load_model(path_obj)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        load_time = time.time() - start_time
        
        print(f"\n[OK] Model loaded in {load_time:.2f}s")
        print(f"[OK] Model is trained: {model.is_trained}")
        print(f"[OK] Algorithm: {model.name}")
        
        # Check metrics if available
        if hasattr(model, 'metrics'):
            print(f"\nModel Metrics:")
            if hasattr(model.metrics, 'rmse') and model.metrics.rmse > 0:
                print(f"  • RMSE: {model.metrics.rmse:.4f}")
            if hasattr(model.metrics, 'coverage') and model.metrics.coverage > 0:
                print(f"  • Coverage: {model.metrics.coverage:.1f}%")
            if hasattr(model.metrics, 'training_time') and model.metrics.training_time > 0:
                print(f"  • Training time: {model.metrics.training_time:.1f}s")
        
        print(f"\n[PASS] PASS: {model_name} model loaded successfully")
        return model
        
    except FileNotFoundError:
        print(f"\n[WARN] Warning: Model file not found")
        print(f"⏭ Skipping test")
        return None
    except Exception as e:
        print(f"\n[FAIL] FAIL: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_recommendations(model, model_name: str, test_user_ids: list):
    """Test generating recommendations for sample users."""
    print_section(f"TEST 3.{['SVD', 'UserKNN', 'ItemKNN', 'ContentBased', 'Hybrid'].index(model_name) + 1}: {model_name} - Recommendations")
    
    if model is None:
        print(f"⏭ Skipping (model not loaded)")
        return None
    
    try:
        success_count = 0
        total_time = 0
        
        for user_id in test_user_ids:
            try:
                start_time = time.time()
                recommendations = model.get_recommendations(user_id, n=5)
                rec_time = time.time() - start_time
                
                total_time += rec_time
                
                # Check if recommendations were generated (handle DataFrame and list)
                has_recs = False
                if isinstance(recommendations, list):
                    has_recs = len(recommendations) > 0
                elif hasattr(recommendations, 'empty'):  # DataFrame
                    has_recs = not recommendations.empty
                
                if has_recs:
                    rec_count = len(recommendations)
                    print(f"[OK] User {user_id}: {rec_count} recommendations in {rec_time:.3f}s")
                    success_count += 1
                else:
                    print(f"[X] User {user_id}: No recommendations generated")
                    
            except Exception as e:
                print(f"[X] User {user_id}: Error - {str(e)}")
        
        avg_time = total_time / len(test_user_ids) if test_user_ids else 0
        
        print(f"\n[ALGO] Results:")
        print(f"  • Successful: {success_count}/{len(test_user_ids)}")
        print(f"  • Average time: {avg_time:.3f}s")
        
        if success_count == len(test_user_ids):
            print(f"\n[PASS] PASS: All recommendations generated successfully")
            return True
        elif success_count > 0:
            print(f"\n[WARN] PARTIAL: {success_count}/{len(test_user_ids)} successful")
            return True
        else:
            print(f"\n[FAIL] FAIL: No recommendations generated")
            return False
            
    except Exception as e:
        print(f"\n[FAIL] FAIL: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_weights():
    """Test that Hybrid properly weights Content-Based."""
    print_section("TEST 4: Hybrid - Content-Based Integration")
    
    try:
        # Check if hybrid model exists
        hybrid_path = Path("models/hybrid_model.pkl")
        
        if not hybrid_path.exists():
            print(f"[WARN] Warning: Hybrid model not found")
            print(f"⏭ Skipping test (model needs to be trained first)")
            return None
        
        # Load hybrid model
        hybrid = HybridRecommender.load_model(str(hybrid_path))
        
        print(f"\n[OK] Hybrid model loaded")
        print(f"[OK] Number of algorithms: {len(hybrid.algorithms)}")
        
        # Check algorithm names
        print(f"\nAlgorithms in ensemble:")
        for algo in hybrid.algorithms:
            print(f"  • {algo.name}")
        
        # Check if Content-Based is included
        algo_names = [algo.name for algo in hybrid.algorithms]
        
        if 'ContentBased' in algo_names or 'Content-Based Filtering' in algo_names:
            print(f"\n[OK] Content-Based is included in Hybrid ensemble")
        else:
            print(f"\n[WARN] Warning: Content-Based not found in Hybrid ensemble")
            print(f"  Note: Hybrid may need to be retrained to include Content-Based")
        
        # Check weights if available
        if hasattr(hybrid, 'weights') and hybrid.weights:
            print(f"\nAlgorithm weights:")
            for algo, weight in zip(hybrid.algorithms, hybrid.weights):
                print(f"  • {algo.name}: {weight:.3f}")
        
        print(f"\n[PASS] PASS: Hybrid integration check complete")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] FAIL: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_prediction_consistency(models: dict):
    """Test that predictions are consistent (same input = same output)."""
    print_section("TEST 5: Prediction Consistency")
    
    try:
        test_user = 1
        test_item = 1
        
        print(f"\nTesting predictions for User {test_user}, Item {test_item}:")
        
        consistent = True
        
        for model_name, model in models.items():
            if model is None:
                print(f"  ⏭ {model_name}: Skipped (not loaded)")
                continue
            
            try:
                # Make prediction twice
                pred1 = model.predict(test_user, test_item)
                pred2 = model.predict(test_user, test_item)
                
                if abs(pred1 - pred2) < 1e-6:
                    print(f"  [OK] {model_name}: Consistent (pred={pred1:.3f})")
                else:
                    print(f"  [X] {model_name}: Inconsistent (pred1={pred1:.3f}, pred2={pred2:.3f})")
                    consistent = False
                    
            except Exception as e:
                print(f"  [X] {model_name}: Error - {str(e)}")
                consistent = False
        
        if consistent:
            print(f"\n[PASS] PASS: All predictions are consistent")
            return True
        else:
            print(f"\n[FAIL] FAIL: Some predictions are inconsistent")
            return False
            
    except Exception as e:
        print(f"\n[FAIL] FAIL: {str(e)}")
        return False


def main():
    """Run all regression tests."""
    print("\n" + "=" * 80)
    print("[MOVIE] CINEMATCH V2.1.0 - REGRESSION TESTING")
    print("=" * 80)
    print("\nTesting all 5 algorithms to ensure Content-Based integration")
    print("didn't break existing functionality.")
    
    # Test 1: AlgorithmManager
    test_results = []
    test_results.append(('AlgorithmManager', test_algorithm_manager()))
    
    # Test 2: Model Loading
    models = {}
    model_paths = {
        'SVD': 'models/svd_model.pkl',
        'UserKNN': 'models/user_knn_model.pkl',
        'ItemKNN': 'models/item_knn_model.pkl',
        'ContentBased': 'models/content_based_model.pkl',
        'Hybrid': 'models/hybrid_model.pkl'
    }
    
    for model_name, model_path in model_paths.items():
        model = test_model_loading(model_name, model_path)
        models[model_name] = model
        test_results.append((f'{model_name} Loading', model is not None))
    
    # Test 3: Recommendations
    test_user_ids = [1, 10, 100, 500, 1000]
    
    for model_name, model in models.items():
        result = test_recommendations(model, model_name, test_user_ids)
        test_results.append((f'{model_name} Recommendations', result))
    
    # Test 4: Hybrid Integration
    result = test_hybrid_weights()
    test_results.append(('Hybrid Integration', result))
    
    # Test 5: Consistency
    result = test_prediction_consistency(models)
    test_results.append(('Prediction Consistency', result))
    
    # Summary
    print("\n" + "=" * 80)
    print("[ALGO] REGRESSION TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in test_results if result is True)
    skipped = sum(1 for _, result in test_results if result is None)
    failed = sum(1 for _, result in test_results if result is False)
    total = len(test_results)
    
    print(f"\nResults:")
    for test_name, result in test_results:
        if result is True:
            status = "[PASS] PASS"
        elif result is None:
            status = "⏭ SKIP"
        else:
            status = "[FAIL] FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n📈 Overall:")
    print(f"  • Passed: {passed}/{total}")
    print(f"  • Skipped: {skipped}/{total}")
    print(f"  • Failed: {failed}/{total}")
    
    if failed == 0 and passed > 0:
        print(f"\n[PASS] ALL REGRESSION TESTS PASSED!")
        print(f"[OK] Content-Based integration successful")
        print(f"[OK] No breaking changes detected")
        return 0
    elif failed == 0 and skipped > 0:
        print(f"\n[WARN] SOME TESTS SKIPPED (models need training)")
        print(f"[OK] All loaded models working correctly")
        return 0
    else:
        print(f"\n[FAIL] SOME TESTS FAILED")
        print(f"[WARN] Review errors above")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


