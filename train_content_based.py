"""
CineMatch V2.1.0 - Content-Based Model Training Script

Train Content-Based Filtering model on MovieLens 32M dataset and save for production use.

Usage:
    python train_content_based.py [--sample-size SIZE]

Options:
    --sample-size: Number of ratings to sample (default: uses full dataset)

Author: CineMatch Development Team
Date: November 11, 2025
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import time
import pickle
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.algorithms.content_based_recommender import ContentBasedRecommender


def load_data(data_path: Path, sample_size: int = None) -> tuple:
    """
    Load MovieLens dataset.
    
    Args:
        data_path: Path to data directory
        sample_size: Optional sample size for testing
        
    Returns:
        Tuple of (ratings_df, movies_df)
    """
    print("=" * 80)
    print("📊 LOADING MOVIELENS 32M DATASET")
    print("=" * 80)
    
    # Load movies
    print("\n1️⃣ Loading movies.csv...")
    movies_df = pd.read_csv(data_path / 'movies.csv')
    print(f"   ✓ Loaded {len(movies_df):,} movies")
    
    # Load ratings
    print("\n2️⃣ Loading ratings.csv...")
    if sample_size:
        print(f"   • Sampling {sample_size:,} ratings for testing...")
        ratings_df = pd.read_csv(
            data_path / 'ratings.csv',
            nrows=sample_size
        )
    else:
        print("   • Loading full dataset (this may take a few minutes)...")
        ratings_df = pd.read_csv(data_path / 'ratings.csv')
    
    print(f"   ✓ Loaded {len(ratings_df):,} ratings")
    print(f"   ✓ {ratings_df['userId'].nunique():,} unique users")
    print(f"   ✓ {ratings_df['movieId'].nunique():,} unique movies")
    print(f"   ✓ Rating range: {ratings_df['rating'].min():.1f} - {ratings_df['rating'].max():.1f}")
    print(f"   ✓ Mean rating: {ratings_df['rating'].mean():.2f}")
    
    return ratings_df, movies_df


def train_model(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    **params
) -> ContentBasedRecommender:
    """
    Train Content-Based model.
    
    Args:
        ratings_df: Ratings DataFrame
        movies_df: Movies DataFrame
        **params: Model hyperparameters
        
    Returns:
        Trained ContentBasedRecommender
    """
    print("\n" + "=" * 80)
    print("🧠 TRAINING CONTENT-BASED FILTERING MODEL")
    print("=" * 80)
    
    # Initialize model
    model = ContentBasedRecommender(**params)
    
    # Train
    start_time = time.time()
    model.fit(ratings_df, movies_df)
    training_time = time.time() - start_time
    
    print(f"\n{'=' * 80}")
    print(f"✓ Training completed in {training_time:.1f}s ({training_time/60:.1f} minutes)")
    print(f"{'=' * 80}")
    
    return model


def validate_model(
    model: ContentBasedRecommender,
    ratings_df: pd.DataFrame,
    n_samples: int = 10
) -> Dict[str, Any]:
    """
    Validate model with sample predictions.
    
    Args:
        model: Trained model
        ratings_df: Ratings DataFrame
        n_samples: Number of samples to validate
        
    Returns:
        Validation metrics
    """
    print("\n" + "=" * 80)
    print("✅ MODEL VALIDATION")
    print("=" * 80)
    
    # Sample random users
    sample_users = ratings_df['userId'].sample(n=n_samples, random_state=42).unique()
    
    print(f"\n📋 Testing recommendations for {len(sample_users)} sample users...")
    
    successful = 0
    failed = 0
    recommendation_times = []
    
    for user_id in sample_users:
        try:
            start_time = time.time()
            recommendations = model.get_recommendations(user_id, n=10)
            rec_time = time.time() - start_time
            
            recommendation_times.append(rec_time)
            successful += 1
            
            print(f"\n   User {user_id}:")
            print(f"   • Generated {len(recommendations)} recommendations in {rec_time:.3f}s")
            if len(recommendations) > 0:
                print(f"   • Top recommendation: {recommendations.iloc[0]['title']}")
                print(f"   • Predicted rating: {recommendations.iloc[0]['predicted_rating']:.2f}")
        
        except Exception as e:
            print(f"\n   ❌ User {user_id} failed: {e}")
            failed += 1
    
    # Calculate metrics
    avg_rec_time = np.mean(recommendation_times) if recommendation_times else 0
    
    print(f"\n{'=' * 80}")
    print("📊 VALIDATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"✓ Successful: {successful}/{n_samples}")
    print(f"✗ Failed: {failed}/{n_samples}")
    print(f"⚡ Avg recommendation time: {avg_rec_time:.3f}s")
    print(f"📈 RMSE: {model.metrics.rmse:.4f}")
    print(f"📊 Coverage: {model.metrics.coverage:.1f}%")
    print(f"💾 Memory usage: {model.metrics.memory_usage_mb:.1f} MB")
    
    return {
        'successful': successful,
        'failed': failed,
        'avg_recommendation_time': avg_rec_time,
        'rmse': model.metrics.rmse,
        'coverage': model.metrics.coverage,
        'memory_mb': model.metrics.memory_usage_mb
    }


def save_model(
    model: ContentBasedRecommender,
    output_path: Path,
    validation_metrics: Dict[str, Any]
) -> None:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        output_path: Path to save model
        validation_metrics: Validation metrics to save with model
    """
    print("\n" + "=" * 80)
    print("💾 SAVING MODEL")
    print("=" * 80)
    
    # Create models directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare model data
    model_data = {
        'model': model,
        'metrics': {
            'rmse': model.metrics.rmse,
            'training_time': model.metrics.training_time,
            'coverage': model.metrics.coverage,
            'memory_mb': model.metrics.memory_usage_mb,
            **validation_metrics
        },
        'metadata': {
            'trained_on': time.strftime("%Y-%m-%d %H:%M:%S"),
            'version': '2.1.0',
            'n_movies': len(model.movie_mapper),
            'feature_dimensions': model.combined_features.shape,
            'similarity_matrix_shape': model.similarity_matrix.shape if model.similarity_matrix is not None else None,
            'on_demand_computation': model.similarity_matrix is None,
            'params': model.params
        }
    }
    
    # Save with pickle
    print(f"\n💾 Saving model to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Get file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"   ✓ Model saved successfully!")
    print(f"   ✓ File size: {file_size_mb:.1f} MB")
    
    # Print model info
    print(f"\n{'=' * 80}")
    print("📦 MODEL INFO")
    print(f"{'=' * 80}")
    print(f"Version: {model_data['metadata']['version']}")
    print(f"Trained: {model_data['metadata']['trained_on']}")
    print(f"Movies: {model_data['metadata']['n_movies']:,}")
    print(f"Feature dimensions: {model_data['metadata']['feature_dimensions']}")
    print(f"Similarity matrix: {model_data['metadata']['similarity_matrix_shape']}")
    print(f"RMSE: {model_data['metrics']['rmse']:.4f}")
    print(f"Coverage: {model_data['metrics']['coverage']:.1f}%")
    print(f"Training time: {model_data['metrics']['training_time']:.1f}s")
    print(f"Memory usage: {model_data['metrics']['memory_mb']:.1f} MB")
    print(f"File size: {file_size_mb:.1f} MB")


def test_model_loading(model_path: Path) -> bool:
    """
    Test loading the saved model.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        True if successful
    """
    print("\n" + "=" * 80)
    print("🧪 TESTING MODEL LOADING")
    print("=" * 80)
    
    try:
        print(f"\n📂 Loading model from: {model_path}")
        start_time = time.time()
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        load_time = time.time() - start_time
        
        model = model_data['model']
        
        print(f"   ✓ Model loaded successfully in {load_time:.2f}s")
        print(f"   ✓ Model is trained: {model.is_trained}")
        print(f"   ✓ Feature matrix shape: {model.combined_features.shape}")
        if model.similarity_matrix is not None:
            print(f"   ✓ Similarity matrix shape: {model.similarity_matrix.shape}")
        else:
            print(f"   ✓ Similarity computation: On-demand (memory-optimized)")
        
        # Test a quick prediction
        print("\n🔮 Testing prediction...")
        test_pred = model.predict(1, 1)
        print(f"   ✓ Test prediction successful: {test_pred:.2f}")
        
        print("\n✅ Model loading test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Model loading test FAILED: {e}")
        return False


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(
        description='Train Content-Based Filtering model for CineMatch'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Number of ratings to sample (default: use full dataset)'
    )
    parser.add_argument(
        '--genre-weight',
        type=float,
        default=0.5,
        help='Weight for genre features (default: 0.5)'
    )
    parser.add_argument(
        '--tag-weight',
        type=float,
        default=0.3,
        help='Weight for tag features (default: 0.3)'
    )
    parser.add_argument(
        '--title-weight',
        type=float,
        default=0.2,
        help='Weight for title features (default: 0.2)'
    )
    parser.add_argument(
        '--min-similarity',
        type=float,
        default=0.01,
        help='Minimum similarity threshold (default: 0.01)'
    )
    
    args = parser.parse_args()
    
    # Paths
    project_root = Path(__file__).parent
    data_path = project_root / 'data' / 'ml-32m'
    output_path = project_root / 'models' / 'content_based_model.pkl'
    
    print("=" * 80)
    print("🎬 CINEMATCH V2.1.0 - CONTENT-BASED FILTERING TRAINING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  • Data path: {data_path}")
    print(f"  • Output path: {output_path}")
    print(f"  • Sample size: {args.sample_size or 'Full dataset'}")
    print(f"  • Genre weight: {args.genre_weight}")
    print(f"  • Tag weight: {args.tag_weight}")
    print(f"  • Title weight: {args.title_weight}")
    print(f"  • Min similarity: {args.min_similarity}")
    
    try:
        # 1. Load data
        ratings_df, movies_df = load_data(data_path, args.sample_size)
        
        # 2. Train model
        model = train_model(
            ratings_df,
            movies_df,
            genre_weight=args.genre_weight,
            tag_weight=args.tag_weight,
            title_weight=args.title_weight,
            min_similarity=args.min_similarity
        )
        
        # 3. Validate model
        validation_metrics = validate_model(model, ratings_df, n_samples=10)
        
        # 4. Save model
        save_model(model, output_path, validation_metrics)
        
        # 5. Test loading
        test_model_loading(output_path)
        
        print("\n" + "=" * 80)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\n✓ Model saved to: {output_path}")
        print("✓ Ready for production use")
        print("\nNext steps:")
        print("  1. Test the model with: python test_content_based.py")
        print("  2. Start the app with: streamlit run app/main.py")
        print("  3. Select 'Content-Based Filtering' from the algorithm dropdown")
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TRAINING FAILED")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
