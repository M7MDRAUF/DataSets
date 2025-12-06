#!/usr/bin/env python3
"""
CineMatch V2.1.6 - Model Training with Joblib Format

Train and save models using joblib format for faster loading.
Implements Task 31-32 of the model loading performance optimization plan.

Features:
- Train all models and save in joblib format
- Compression support for smaller file sizes
- Memory-efficient training pipeline
- Progress tracking and reporting

Usage:
    python scripts/train_models_joblib.py
    python scripts/train_models_joblib.py --algorithm svd
    python scripts/train_models_joblib.py --compress 3
    python scripts/train_models_joblib.py --all

Author: CineMatch Development Team
Date: December 5, 2025
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configurations
MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

ALGORITHMS = {
    'svd': {
        'filename': 'svd_model_sklearn.joblib',
        'description': 'SVD Matrix Factorization',
        'trainer': 'train_svd_model'
    },
    'user_knn': {
        'filename': 'user_knn_model.joblib',
        'description': 'User-based KNN',
        'trainer': 'train_user_knn_model'
    },
    'item_knn': {
        'filename': 'item_knn_model.joblib',
        'description': 'Item-based KNN',
        'trainer': 'train_item_knn_model'
    },
    'content_based': {
        'filename': 'content_based_model.joblib',
        'description': 'Content-Based Filtering',
        'trainer': 'train_content_based_model'
    },
    'hybrid': {
        'filename': 'hybrid_model.joblib',
        'description': 'Hybrid Ensemble',
        'trainer': 'train_hybrid_model'
    }
}


def get_file_size_mb(path: Path) -> float:
    """Get file size in megabytes."""
    if path.exists():
        return path.stat().st_size / (1024 * 1024)
    return 0.0


def save_model_joblib(
    model: Any,
    output_path: Path,
    compress: int = 3
) -> Dict[str, Any]:
    """
    Save model using joblib with compression.
    
    Args:
        model: Model object to save
        output_path: Path to save to
        compress: Compression level (0-9, 0=none, 3=recommended)
        
    Returns:
        Dict with save stats (size, time, compression ratio)
    """
    start_time = time.time()
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if pickle version exists for comparison
    pickle_path = output_path.with_suffix('.pkl')
    original_size = get_file_size_mb(pickle_path) if pickle_path.exists() else 0
    
    # Save with joblib
    joblib.dump(model, output_path, compress=compress)
    
    save_time = time.time() - start_time
    new_size = get_file_size_mb(output_path)
    
    compression_ratio = original_size / new_size if new_size > 0 and original_size > 0 else 0
    
    return {
        'path': str(output_path),
        'size_mb': new_size,
        'original_size_mb': original_size,
        'compression_ratio': compression_ratio,
        'save_time_seconds': save_time,
        'compress_level': compress
    }


def train_svd_model(compress: int = 3) -> Dict[str, Any]:
    """Train and save SVD model in joblib format."""
    logger.info("Training SVD model...")
    
    try:
        from src.svd_model_sklearn import SVDRecommender
        from src.data_processing import load_processed_data
        
        # Load data
        logger.info("Loading data...")
        ratings_df, movies_df = load_processed_data()
        
        # Train model
        logger.info("Training SVD...")
        model = SVDRecommender()
        model.fit(ratings_df)
        
        # Save
        output_path = MODELS_DIR / 'svd_model_sklearn.joblib'
        stats = save_model_joblib(model, output_path, compress)
        
        logger.info(f"SVD model saved: {stats['size_mb']:.1f}MB")
        return stats
        
    except Exception as e:
        logger.error(f"Failed to train SVD: {e}")
        raise


def train_user_knn_model(compress: int = 3) -> Dict[str, Any]:
    """Train and save User KNN model in joblib format."""
    logger.info("Training User KNN model...")
    
    try:
        from src.algorithms.user_knn_recommender import UserKNNRecommender
        from src.data_processing import load_processed_data
        
        # Load data
        ratings_df, movies_df = load_processed_data()
        
        # Train model
        model = UserKNNRecommender()
        model.fit(ratings_df)
        
        # Save
        output_path = MODELS_DIR / 'user_knn_model.joblib'
        stats = save_model_joblib(model, output_path, compress)
        
        logger.info(f"User KNN model saved: {stats['size_mb']:.1f}MB")
        return stats
        
    except Exception as e:
        logger.error(f"Failed to train User KNN: {e}")
        raise


def train_item_knn_model(compress: int = 3) -> Dict[str, Any]:
    """Train and save Item KNN model in joblib format."""
    logger.info("Training Item KNN model...")
    
    try:
        from src.algorithms.item_knn_recommender import ItemKNNRecommender
        from src.data_processing import load_processed_data
        
        # Load data
        ratings_df, movies_df = load_processed_data()
        
        # Train model
        model = ItemKNNRecommender()
        model.fit(ratings_df)
        
        # Save
        output_path = MODELS_DIR / 'item_knn_model.joblib'
        stats = save_model_joblib(model, output_path, compress)
        
        logger.info(f"Item KNN model saved: {stats['size_mb']:.1f}MB")
        return stats
        
    except Exception as e:
        logger.error(f"Failed to train Item KNN: {e}")
        raise


def train_content_based_model(compress: int = 3) -> Dict[str, Any]:
    """Train and save Content-Based model in joblib format."""
    logger.info("Training Content-Based model...")
    
    try:
        from src.algorithms.content_based_recommender import ContentBasedRecommender
        from src.data_processing import load_processed_data
        
        # Load data
        ratings_df, movies_df = load_processed_data()
        
        # Train model
        model = ContentBasedRecommender()
        model.fit(movies_df)
        
        # Save
        output_path = MODELS_DIR / 'content_based_model.joblib'
        stats = save_model_joblib(model, output_path, compress)
        
        logger.info(f"Content-Based model saved: {stats['size_mb']:.1f}MB")
        return stats
        
    except Exception as e:
        logger.error(f"Failed to train Content-Based: {e}")
        raise


def train_hybrid_model(compress: int = 3) -> Dict[str, Any]:
    """Train and save Hybrid model in joblib format."""
    logger.info("Training Hybrid model...")
    
    try:
        from src.algorithms.hybrid_recommender import HybridRecommender
        
        # Hybrid model requires sub-models to be trained first
        # It coordinates multiple algorithms
        model = HybridRecommender()
        # Note: Hybrid model typically doesn't need separate training
        # as it combines already-trained models
        
        # Save
        output_path = MODELS_DIR / 'hybrid_model.joblib'
        stats = save_model_joblib(model, output_path, compress)
        
        logger.info(f"Hybrid model saved: {stats['size_mb']:.1f}MB")
        return stats
        
    except Exception as e:
        logger.error(f"Failed to train Hybrid: {e}")
        raise


def train_all_models(compress: int = 3) -> Dict[str, Dict[str, Any]]:
    """Train and save all models in joblib format."""
    logger.info("=" * 60)
    logger.info("Training all models with joblib format")
    logger.info(f"Compression level: {compress}")
    logger.info("=" * 60)
    
    results = {}
    total_start = time.time()
    
    trainers = {
        'svd': train_svd_model,
        'user_knn': train_user_knn_model,
        'item_knn': train_item_knn_model,
        'content_based': train_content_based_model,
        'hybrid': train_hybrid_model
    }
    
    for name, trainer in trainers.items():
        logger.info(f"\n--- {name.upper()} ---")
        try:
            results[name] = trainer(compress)
            results[name]['status'] = 'success'
        except Exception as e:
            results[name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    total_time = time.time() - total_start
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    
    total_size = 0
    total_original = 0
    
    for name, stats in results.items():
        if stats.get('status') == 'success':
            logger.info(f"{name}: {stats['size_mb']:.1f}MB (saved in {stats['save_time_seconds']:.1f}s)")
            total_size += stats['size_mb']
            total_original += stats.get('original_size_mb', 0)
        else:
            logger.error(f"{name}: FAILED - {stats.get('error', 'Unknown error')}")
    
    logger.info(f"\nTotal new size: {total_size:.1f}MB")
    if total_original > 0:
        logger.info(f"Total original size: {total_original:.1f}MB")
        logger.info(f"Space saved: {total_original - total_size:.1f}MB ({((total_original - total_size) / total_original) * 100:.1f}%)")
    logger.info(f"Total time: {total_time:.1f}s")
    
    return results


def convert_existing_model(
    pickle_path: Path,
    compress: int = 3,
    keep_original: bool = True
) -> Dict[str, Any]:
    """
    Convert an existing pickle model to joblib format.
    
    Args:
        pickle_path: Path to existing pickle model
        compress: Compression level
        keep_original: Whether to keep the original pickle file
        
    Returns:
        Conversion stats
    """
    import pickle
    
    if not pickle_path.exists():
        raise FileNotFoundError(f"Model not found: {pickle_path}")
    
    logger.info(f"Converting {pickle_path.name}...")
    
    # Load pickle model
    start_load = time.time()
    with open(pickle_path, 'rb') as f:
        model = pickle.load(f)
    load_time = time.time() - start_load
    
    # Save as joblib
    joblib_path = pickle_path.with_suffix('.joblib')
    stats = save_model_joblib(model, joblib_path, compress)
    
    # Update stats
    stats['load_time_seconds'] = load_time
    stats['original_path'] = str(pickle_path)
    stats['original_size_mb'] = get_file_size_mb(pickle_path)
    
    # Optionally remove original
    if not keep_original:
        pickle_path.unlink()
        logger.info(f"Removed original: {pickle_path}")
    
    compression_pct = (1 - stats['size_mb'] / stats['original_size_mb']) * 100 if stats['original_size_mb'] > 0 else 0
    logger.info(f"Converted: {stats['original_size_mb']:.1f}MB -> {stats['size_mb']:.1f}MB ({compression_pct:.1f}% reduction)")
    
    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train CineMatch models with joblib format'
    )
    parser.add_argument(
        '--algorithm', '-a',
        choices=['svd', 'user_knn', 'item_knn', 'content_based', 'hybrid'],
        help='Train specific algorithm (default: all)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Train all algorithms'
    )
    parser.add_argument(
        '--compress', '-c',
        type=int,
        default=3,
        choices=range(0, 10),
        help='Compression level (0-9, default: 3)'
    )
    parser.add_argument(
        '--convert',
        action='store_true',
        help='Convert existing pickle models instead of retraining'
    )
    parser.add_argument(
        '--keep-original',
        action='store_true',
        default=True,
        help='Keep original pickle files when converting'
    )
    
    args = parser.parse_args()
    
    if args.convert:
        # Convert existing models
        logger.info("Converting existing pickle models to joblib...")
        for name, config in ALGORITHMS.items():
            pickle_path = MODELS_DIR / config['filename'].replace('.joblib', '.pkl')
            if pickle_path.exists():
                try:
                    convert_existing_model(
                        pickle_path,
                        compress=args.compress,
                        keep_original=args.keep_original
                    )
                except Exception as e:
                    logger.error(f"Failed to convert {name}: {e}")
    elif args.algorithm:
        # Train specific algorithm
        trainers = {
            'svd': train_svd_model,
            'user_knn': train_user_knn_model,
            'item_knn': train_item_knn_model,
            'content_based': train_content_based_model,
            'hybrid': train_hybrid_model
        }
        trainer = trainers.get(args.algorithm)
        if trainer:
            trainer(args.compress)
        else:
            logger.error(f"Unknown algorithm: {args.algorithm}")
            sys.exit(1)
    else:
        # Train all
        train_all_models(args.compress)


if __name__ == '__main__':
    main()
