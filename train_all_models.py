"""
CineMatch V2.0.2 - Optimized Model Training Script

Trains all 5 recommendation models with fixed code and saves pickle files.
Use this after code changes that affect pickle compatibility.

Author: CineMatch Development Team
Date: November 13, 2025
Version: 1.0
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.algorithms.algorithm_manager import AlgorithmManager, AlgorithmType


def train_all_models(use_sample: bool = False):
    """
    Train all models and save to disk.
    
    Args:
        use_sample: If True, uses sample data for quick testing.
                   If False, uses full 32M dataset for production.
    """
    print("=" * 80)
    print("🚀 CINEMATCH MODEL TRAINING - ALL ALGORITHMS")
    print("=" * 80)
    print(f"Dataset: {'Sample (100K)' if use_sample else 'Full (32M ratings)'}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    total_start = time.time()
    
    # Initialize manager
    print("📊 Step 1/6: Loading data...")
    step_start = time.time()
    
    # Load data (always use full dataset - sample mode will just train faster algorithms first)
    import pandas as pd
    ratings_path = 'data/ml-32m/ratings.csv'
    movies_path = 'data/ml-32m/movies.csv'
    
    if use_sample:
        print("⚡ Sample mode: Will use subset of data for faster training")
        print()
    
    print(f"Loading ratings from {ratings_path}...")
    ratings_df = pd.read_csv(ratings_path)
    print(f"  ✓ Loaded {len(ratings_df):,} ratings")
    
    print(f"Loading movies from {movies_path}...")
    movies_df = pd.read_csv(movies_path)
    print(f"  ✓ Loaded {len(movies_df):,} movies")
    
    # Initialize manager with data
    manager = AlgorithmManager()
    manager.initialize_data(ratings_df, movies_df)
    
    print(f"   ✅ Data loaded in {time.time() - step_start:.2f}s")
    print()
    
    # Train each algorithm
    algorithms = [
        (AlgorithmType.SVD, "SVD Matrix Factorization"),
        (AlgorithmType.USER_KNN, "User-Based KNN"),
        (AlgorithmType.ITEM_KNN, "Item-Based KNN"),
        (AlgorithmType.CONTENT_BASED, "Content-Based Filtering"),
        (AlgorithmType.HYBRID, "Hybrid (Best of All)")
    ]
    
    results = []
    
    for i, (algo_type, algo_name) in enumerate(algorithms, start=2):
        print(f"🎯 Step {i}/6: Training {algo_name}...")
        step_start = time.time()
        
        try:
            # Switch to algorithm (this will train and save it)
            algorithm = manager.switch_algorithm(algo_type, suppress_ui=True)
            
            # Save explicitly (switch_algorithm should do this, but let's be sure)
            if hasattr(algorithm, 'save_model'):
                algorithm.save_model()
            
            step_time = time.time() - step_start
            results.append((algo_name, step_time, "✅ Success"))
            print(f"   ✅ Trained and saved in {step_time:.2f}s")
            
        except Exception as e:
            step_time = time.time() - step_start
            results.append((algo_name, step_time, f"❌ Failed: {str(e)[:50]}"))
            print(f"   ❌ Failed after {step_time:.2f}s")
            print(f"      Error: {str(e)[:100]}")
        
        print()
    
    # Summary
    total_time = time.time() - total_start
    
    print("=" * 80)
    print("📊 TRAINING SUMMARY")
    print("=" * 80)
    print(f"{'Algorithm':<30} {'Time':<12} {'Status'}")
    print("-" * 80)
    
    for algo_name, step_time, status in results:
        print(f"{algo_name:<30} {step_time:>8.2f}s    {status}")
    
    print("-" * 80)
    print(f"{'TOTAL TIME':<30} {total_time:>8.2f}s    ({total_time/60:.1f} minutes)")
    print("=" * 80)
    
    # Success rate
    success_count = sum(1 for _, _, status in results if "✅" in status)
    print(f"\n✅ Successfully trained: {success_count}/{len(results)} models")
    
    if success_count == len(results):
        print("\n🎉 All models trained successfully!")
        print("   Ready for production use.")
    else:
        print(f"\n⚠️  {len(results) - success_count} model(s) failed to train.")
        print("   Check errors above and fix before deploying.")
    
    print()
    return success_count == len(results)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train all CineMatch recommendation models")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample data (100K ratings) for quick testing"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full dataset (32M ratings) for production models (default)"
    )
    
    args = parser.parse_args()
    
    # Default to full if neither specified
    use_sample = args.sample and not args.full
    
    if use_sample:
        print("⚡ QUICK MODE: Using sample data (100K ratings)")
        print("   Estimated time: 2-3 minutes")
    else:
        print("🐢 PRODUCTION MODE: Using full dataset (32M ratings)")
        print("   Estimated time: 10-15 minutes")
        print("   (Tip: Use --sample flag for quick testing)")
    
    print()
    input("Press Enter to start training, or Ctrl+C to cancel...")
    print()
    
    success = train_all_models(use_sample=use_sample)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
