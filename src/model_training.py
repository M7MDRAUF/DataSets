"""
CineMatch V1.0.0 - Model Training Module

This module handles SVD model training on the MovieLens dataset.
Implements collaborative filtering using matrix factorization.

Author: CineMatch Team
Date: October 24, 2025
"""

import sys
import time
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy
import joblib

# Import data processing functions
sys.path.append(str(Path(__file__).parent.parent))
from src.data_processing import (
    check_data_integrity,
    load_ratings,
    load_movies,
    preprocess_data
)


# Model configuration
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "svd_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# Hyperparameters (optimized for MovieLens 32M)
HYPERPARAMETERS = {
    'n_factors': 100,       # Number of latent factors
    'n_epochs': 20,         # Training iterations
    'lr_all': 0.005,        # Learning rate (SGD)
    'reg_all': 0.02,        # Regularization factor
    'random_state': 42,     # For reproducibility
    'verbose': True         # Print progress
}

# Target RMSE from PRD
TARGET_RMSE = 0.87


def prepare_surprise_dataset(ratings_df: pd.DataFrame) -> Dataset:
    """
    Convert pandas DataFrame to Surprise Dataset format.
    
    Args:
        ratings_df: DataFrame with columns [userId, movieId, rating]
    
    Returns:
        Surprise Dataset object ready for training
    """
    print("\nPreparing dataset for Surprise library...")
    
    # Define rating scale (MovieLens uses 0.5 to 5.0)
    reader = Reader(rating_scale=(0.5, 5.0))
    
    # Create Dataset from DataFrame
    data = Dataset.load_from_df(
        ratings_df[['userId', 'movieId', 'rating']],
        reader
    )
    
    print(f"  ✓ Dataset prepared with {len(ratings_df):,} ratings")
    
    return data


def train_svd_model(
    data: Dataset,
    hyperparameters: Dict[str, Any] = HYPERPARAMETERS
) -> Tuple[SVD, float, float]:
    """
    Train SVD model using collaborative filtering.
    
    Args:
        data: Surprise Dataset object
        hyperparameters: Model hyperparameters
    
    Returns:
        Tuple of (trained_model, train_rmse, test_rmse)
    """
    print("\n" + "=" * 70)
    print("TRAINING SVD MODEL")
    print("=" * 70)
    
    # Split data: 80% training, 20% testing
    print("\nSplitting data (80% train, 20% test)...")
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    print(f"  • Training set: {trainset.n_ratings:,} ratings")
    print(f"  • Test set: {len(testset):,} ratings")
    
    # Initialize SVD model
    print(f"\nInitializing SVD model with hyperparameters:")
    for key, value in hyperparameters.items():
        if key != 'verbose':
            print(f"  • {key}: {value}")
    
    model = SVD(**hyperparameters)
    
    # Train model
    print(f"\n🚀 Starting training ({hyperparameters['n_epochs']} epochs)...")
    print("This may take 15-30 minutes on the full dataset...")
    
    start_time = time.time()
    model.fit(trainset)
    training_time = time.time() - start_time
    
    print(f"\n✓ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_predictions = model.test(trainset.build_testset())
    train_rmse = accuracy.rmse(train_predictions, verbose=False)
    print(f"  • Training RMSE: {train_rmse:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_predictions = model.test(testset)
    test_rmse = accuracy.rmse(test_predictions, verbose=False)
    print(f"  • Test RMSE: {test_rmse:.4f}")
    
    # Check if target RMSE is met
    if test_rmse < TARGET_RMSE:
        print(f"\n🎉 SUCCESS! Test RMSE ({test_rmse:.4f}) < Target ({TARGET_RMSE})")
    else:
        print(f"\n⚠️  WARNING: Test RMSE ({test_rmse:.4f}) >= Target ({TARGET_RMSE})")
        print("   Consider hyperparameter tuning or using more data")
    
    return model, train_rmse, test_rmse


def cross_validate_model(data: Dataset, n_folds: int = 5):
    """
    Perform cross-validation to assess model robustness.
    
    Args:
        data: Surprise Dataset object
        n_folds: Number of folds for cross-validation
    """
    print("\n" + "=" * 70)
    print(f"CROSS-VALIDATION ({n_folds}-Fold)")
    print("=" * 70)
    
    model = SVD(**HYPERPARAMETERS)
    
    print("\nPerforming cross-validation...")
    print("This may take some time...")
    
    results = cross_validate(
        model,
        data,
        measures=['RMSE', 'MAE'],
        cv=n_folds,
        verbose=True
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 70)
    print(f"\nRMSE: {results['test_rmse'].mean():.4f} (+/- {results['test_rmse'].std():.4f})")
    print(f"MAE:  {results['test_mae'].mean():.4f} (+/- {results['test_mae'].std():.4f})")
    
    return results


def save_model(model: SVD, train_rmse: float, test_rmse: float, training_time: float = 0):
    """
    Save trained model and metadata to disk.
    
    Args:
        model: Trained SVD model
        train_rmse: Training set RMSE
        test_rmse: Test set RMSE
        training_time: Time taken to train (seconds)
    """
    # Create models directory if it doesn't exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving model to {MODEL_PATH}...")
    
    # Save model using joblib (more efficient than pickle for large numpy arrays)
    joblib.dump(model, MODEL_PATH)
    
    # Save metadata (convert all numpy types to Python native types)
    metadata = {
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'training_time_seconds': float(training_time),
        'hyperparameters': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                           for k, v in HYPERPARAMETERS.items()},
        'target_rmse': float(TARGET_RMSE),
        'success': bool(test_rmse < TARGET_RMSE),  # Convert numpy bool_ to Python bool
        'model_type': 'SVD (Surprise)',
        'trained_on': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    import json
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Model saved: {MODEL_PATH}")
    print(f"  ✓ Metadata saved: {METADATA_PATH}")
    
    # Print file sizes
    model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"\n📊 Model file size: {model_size_mb:.2f} MB")


def load_model() -> SVD:
    """
    Load trained model from disk.
    
    Returns:
        Loaded SVD model
    
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Please run model training first: python src/model_training.py"
        )
    
    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    print("  ✓ Model loaded successfully")
    
    # Load and display metadata if available
    if METADATA_PATH.exists():
        import json
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        print(f"\n📊 Model Metadata:")
        print(f"  • Test RMSE: {metadata['test_rmse']:.4f}")
        print(f"  • Trained on: {metadata['trained_on']}")
        print(f"  • Target met: {'✓ Yes' if metadata['success'] else '✗ No'}")
    
    return model


def main():
    """
    Main training pipeline.
    """
    print("\n" + "=" * 70)
    print("CineMatch V1.0.0 - Model Training Pipeline")
    print("=" * 70)
    
    # Step 1: Check data integrity
    print("\n[1/5] Checking data integrity...")
    success, missing, error = check_data_integrity()
    if not success:
        print(error)
        sys.exit(1)
    print("  ✓ All required files found")
    
    # Step 2: Load data
    print("\n[2/5] Loading dataset...")
    # For full training, use: ratings_df = load_ratings()
    # For quick testing, use: ratings_df = load_ratings(sample_size=1000000)
    
    # Decide based on user input or default to sample for safety
    use_full_dataset = input("\nTrain on FULL dataset (32M ratings, ~30 min) or SAMPLE (1M ratings, ~2 min)? [full/sample]: ").strip().lower()
    
    if use_full_dataset == 'full':
        print("\n⚠️  Training on full dataset - this will take 15-30 minutes")
        ratings_df = load_ratings()
    else:
        print("\n📊 Training on 1M sample for faster results")
        ratings_df = load_ratings(sample_size=1_000_000)
    
    movies_df = load_movies()
    
    # Step 3: Preprocess
    print("\n[3/5] Preprocessing data...")
    ratings_df, movies_df = preprocess_data(ratings_df, movies_df)
    
    # Step 4: Train model
    print("\n[4/5] Training model...")
    data = prepare_surprise_dataset(ratings_df)
    
    start_time = time.time()
    model, train_rmse, test_rmse = train_svd_model(data, HYPERPARAMETERS)
    training_time = time.time() - start_time
    
    # Step 5: Save model
    print("\n[5/5] Saving model...")
    save_model(model, train_rmse, test_rmse, training_time)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nFinal Results:")
    print(f"  • Training RMSE: {train_rmse:.4f}")
    print(f"  • Test RMSE: {test_rmse:.4f}")
    print(f"  • Training Time: {training_time/60:.2f} minutes")
    print(f"  • Model saved: {MODEL_PATH}")
    print(f"\nYou can now run the Streamlit app: streamlit run app/main.py")


if __name__ == "__main__":
    main()
