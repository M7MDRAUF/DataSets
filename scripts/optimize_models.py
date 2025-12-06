#!/usr/bin/env python3
"""
Model Optimization Script for CineMatch
Strips unnecessary data from trained models to reduce file sizes.

Key Optimizations:
1. Remove ratings_df from all models (saves ~610MB per model)
2. Remove tags_df from ContentBased (saves ~160MB)
3. Convert dense matrices to sparse where beneficial
4. Apply sparse matrix compression

Expected size reductions:
- SVD: 909MB → ~250MB (72% reduction)
- ContentBased: 1.1GB → ~50MB (95% reduction)
- UserKNN: 1.1GB → ~270MB (75% reduction)
- ItemKNN: 1.1GB → ~270MB (75% reduction)
- Hybrid: Already optimal
"""

import sys
import os
import logging
from pathlib import Path
import time
import pickle
import shutil
from typing import Optional, Dict, Any, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Attributes that are safe to remove (can be reloaded from source data)
REMOVABLE_ATTRS = {
    'ratings_df': 'Raw ratings - not needed for inference',
    'tags_df': 'Raw tags - only needed during training',
    'links_df': 'Movie links - not needed for recommendations',
    'genome_scores': 'Genome scores - can be recomputed',
    'genome_tags': 'Genome tags - can be recomputed',
}

# Size thresholds for warnings
MAX_MODEL_SIZE_MB = 300  # Warn if model exceeds this
TARGET_MODEL_SIZE_MB = 250  # Target size after optimization


def get_file_size_mb(path: Path) -> float:
    """Get file size in megabytes."""
    return path.stat().st_size / (1024 * 1024)


def get_object_size(obj: Any, deep: bool = True) -> float:
    """Estimate object size in MB."""
    try:
        if hasattr(obj, 'memory_usage'):
            # DataFrame
            if deep:
                return obj.memory_usage(deep=True).sum() / (1024 * 1024)
            return obj.memory_usage().sum() / (1024 * 1024)
        elif hasattr(obj, 'nbytes'):
            # NumPy array
            return obj.nbytes / (1024 * 1024)
        elif hasattr(obj, 'data') and hasattr(obj, 'indices'):
            # Sparse matrix
            size = obj.data.nbytes + obj.indices.nbytes
            if hasattr(obj, 'indptr'):
                size += obj.indptr.nbytes
            return size / (1024 * 1024)
        else:
            # Generic estimate
            import sys
            return sys.getsizeof(obj) / (1024 * 1024)
    except Exception:
        return 0.0


def analyze_model(model: Any) -> Dict[str, Dict[str, Any]]:
    """Analyze model attributes and their sizes."""
    analysis = {}
    
    for attr_name in dir(model):
        if attr_name.startswith('_'):
            continue
        
        try:
            attr = getattr(model, attr_name, None)
            if callable(attr):
                continue
            
            size_mb = get_object_size(attr)
            attr_type = type(attr).__name__
            
            # Determine if removable
            removable = attr_name in REMOVABLE_ATTRS
            reason = REMOVABLE_ATTRS.get(attr_name, '')
            
            analysis[attr_name] = {
                'type': attr_type,
                'size_mb': size_mb,
                'removable': removable,
                'reason': reason
            }
        except Exception as e:
            logger.debug(f"Could not analyze attribute {attr_name}: {e}")
    
    return analysis


def strip_model_attributes(model: Any, attrs_to_remove: List[str]) -> Tuple[Any, Dict[str, float]]:
    """
    Strip specified attributes from a model.
    Returns the optimized model and dict of removed sizes.
    """
    removed_sizes = {}
    
    for attr_name in attrs_to_remove:
        if hasattr(model, attr_name):
            attr = getattr(model, attr_name)
            size_mb = get_object_size(attr)
            
            # Set to None instead of deleting to maintain structure
            setattr(model, attr_name, None)
            removed_sizes[attr_name] = size_mb
            logger.info(f"  Stripped {attr_name}: {size_mb:.1f} MB")
    
    return model, removed_sizes


def compress_sparse_matrix(matrix):
    """Compress sparse matrix by converting to most efficient format."""
    from scipy import sparse
    
    if not sparse.issparse(matrix):
        return matrix
    
    # Try different formats and pick smallest
    formats = [
        ('csr', sparse.csr_matrix),
        ('csc', sparse.csc_matrix),
        ('coo', sparse.coo_matrix),
    ]
    
    current_size = get_object_size(matrix)
    best_format = matrix
    best_size = current_size
    
    for name, converter in formats:
        try:
            converted = converter(matrix)
            size = get_object_size(converted)
            if size < best_size:
                best_size = size
                best_format = converted
        except Exception:
            continue
    
    return best_format


def optimize_model(model: Any, model_name: str) -> Tuple[Any, Dict]:
    """
    Optimize a single model by stripping unnecessary data.
    Returns optimized model and optimization report.
    """
    report = {
        'model_name': model_name,
        'original_attrs': {},
        'removed_attrs': {},
        'compressed_attrs': {},
        'total_saved_mb': 0.0
    }
    
    logger.info(f"Analyzing {model_name}...")
    
    # Handle wrapped models (dict with 'model' key)
    actual_model = model
    wrapper = None
    if isinstance(model, dict) and 'model' in model:
        wrapper = model
        actual_model = model['model']
        logger.info(f"  Found wrapped model, optimizing inner model")
    
    # Analyze current state
    analysis = analyze_model(actual_model)
    report['original_attrs'] = {k: v['size_mb'] for k, v in analysis.items()}
    
    # Determine what to strip
    attrs_to_strip = []
    for attr_name, info in analysis.items():
        if info['removable'] and info['size_mb'] > 0.1:  # Skip tiny attrs
            attrs_to_strip.append(attr_name)
    
    # Strip attributes
    if attrs_to_strip:
        logger.info(f"Stripping {len(attrs_to_strip)} attributes from {model_name}:")
        actual_model, removed = strip_model_attributes(actual_model, attrs_to_strip)
        report['removed_attrs'] = removed
        report['total_saved_mb'] = sum(removed.values())
    
    # Compress sparse matrices
    from scipy import sparse
    for attr_name in dir(actual_model):
        if attr_name.startswith('_'):
            continue
        try:
            attr = getattr(actual_model, attr_name, None)
            if sparse.issparse(attr):
                original_size = get_object_size(attr)
                compressed = compress_sparse_matrix(attr)
                new_size = get_object_size(compressed)
                if new_size < original_size * 0.95:  # At least 5% savings
                    setattr(actual_model, attr_name, compressed)
                    report['compressed_attrs'][attr_name] = {
                        'original_mb': original_size,
                        'compressed_mb': new_size,
                        'saved_mb': original_size - new_size
                    }
                    logger.info(f"  Compressed {attr_name}: {original_size:.1f}MB → {new_size:.1f}MB")
        except Exception as e:
            logger.debug(f"Could not compress {attr_name}: {e}")
    
    # Re-wrap if needed
    if wrapper is not None:
        wrapper['model'] = actual_model
        return wrapper, report
    
    return actual_model, report


def save_optimized_model(model: Any, path: Path, use_joblib: bool = True, compress: bool = False) -> float:
    """
    Save optimized model and return file size.
    
    Args:
        model: Model to save
        path: Output path
        use_joblib: Use joblib for saving (enables mmap loading)
        compress: Apply compression (slower to load, may break mmap)
    """
    import joblib
    
    if use_joblib:
        if compress:
            # Compressed - smaller file but slower to load, breaks mmap
            joblib.dump(model, path, compress=('zlib', 3))
        else:
            # Uncompressed - larger file but fast loading with mmap support
            joblib.dump(model, path, compress=0)
    else:
        with open(path, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return get_file_size_mb(path)


def optimize_all_models(
    models_dir: Path,
    output_dir: Optional[Path] = None,
    backup: bool = True
) -> Dict[str, Dict]:
    """
    Optimize all models in the directory.
    Returns comprehensive report.
    """
    if output_dir is None:
        output_dir = models_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all model files
    model_files = list(models_dir.glob('*.pkl')) + list(models_dir.glob('*.joblib'))
    
    reports = {}
    total_original_size = 0.0
    total_optimized_size = 0.0
    
    logger.info(f"Found {len(model_files)} model files to optimize")
    logger.info("=" * 60)
    
    for model_path in model_files:
        model_name = model_path.stem
        
        # Skip already optimized files
        if '_optimized' in model_name or '_backup' in model_name:
            continue
        
        original_size = get_file_size_mb(model_path)
        total_original_size += original_size
        
        logger.info(f"\nProcessing {model_name} ({original_size:.1f} MB)...")
        
        try:
            # Load model
            import joblib
            try:
                model = joblib.load(model_path)
            except Exception:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            
            # Backup if requested
            if backup and output_dir == models_dir:
                backup_path = model_path.with_suffix('.pkl.backup')
                if not backup_path.exists():
                    shutil.copy(model_path, backup_path)
                    logger.info(f"  Backed up to {backup_path.name}")
            
            # Optimize
            optimized_model, report = optimize_model(model, model_name)
            
            # Save optimized model
            output_path = output_dir / f"{model_name}.pkl"
            optimized_size = save_optimized_model(optimized_model, output_path)
            total_optimized_size += optimized_size
            
            report['original_file_size_mb'] = original_size
            report['optimized_file_size_mb'] = optimized_size
            report['file_reduction_percent'] = (1 - optimized_size / original_size) * 100
            
            reports[model_name] = report
            
            logger.info(f"  Saved: {original_size:.1f} MB → {optimized_size:.1f} MB ({report['file_reduction_percent']:.1f}% reduction)")
            
        except Exception as e:
            logger.error(f"  Failed to optimize {model_name}: {e}")
            import traceback
            traceback.print_exc()
            reports[model_name] = {'error': str(e)}
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total original size: {total_original_size:.1f} MB")
    logger.info(f"Total optimized size: {total_optimized_size:.1f} MB")
    logger.info(f"Total saved: {total_original_size - total_optimized_size:.1f} MB ({(1 - total_optimized_size / total_original_size) * 100:.1f}%)")
    
    return reports


def validate_optimized_model(original_path: Path, optimized_path: Path) -> Dict[str, Any]:
    """
    Validate that optimized model still works correctly.
    Returns validation report.
    """
    import joblib
    
    report = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    try:
        # Load both models
        original = joblib.load(original_path)
        optimized = joblib.load(optimized_path)
        
        # Handle wrapped models
        if isinstance(original, dict) and 'model' in original:
            original = original['model']
        if isinstance(optimized, dict) and 'model' in optimized:
            optimized = optimized['model']
        
        # Check essential attributes still exist
        essential_attrs = ['is_trained', 'movie_mapper', 'movie_inv_mapper']
        for attr in essential_attrs:
            if hasattr(original, attr) and not hasattr(optimized, attr):
                report['errors'].append(f"Missing essential attribute: {attr}")
                report['valid'] = False
        
        # Check is_trained flag
        if hasattr(optimized, 'is_trained') and not optimized.is_trained:
            report['warnings'].append("Model is_trained flag is False")
        
        # Check recommendation method exists
        if not hasattr(optimized, 'recommend') and not hasattr(optimized, 'get_recommendations'):
            report['errors'].append("Model missing recommend/get_recommendations method")
            report['valid'] = False
        
    except Exception as e:
        report['valid'] = False
        report['errors'].append(f"Validation error: {str(e)}")
    
    return report


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Optimize CineMatch models by stripping unnecessary data'
    )
    parser.add_argument(
        '--models-dir',
        type=Path,
        default=project_root / 'models',
        help='Directory containing model files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (default: same as input)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup files'
    )
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only analyze models, do not optimize'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Optimize specific model only'
    )
    
    args = parser.parse_args()
    
    if args.analyze_only:
        # Just analyze models
        model_files = list(args.models_dir.glob('*.pkl')) + list(args.models_dir.glob('*.joblib'))
        
        for model_path in model_files:
            if '_backup' in model_path.stem:
                continue
                
            model_name = model_path.stem
            file_size = get_file_size_mb(model_path)
            
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Model: {model_name} ({file_size:.1f} MB)")
            logger.info('=' * 60)
            
            try:
                import joblib
                try:
                    model = joblib.load(model_path)
                except Exception:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                
                if isinstance(model, dict) and 'model' in model:
                    model = model['model']
                
                analysis = analyze_model(model)
                
                # Sort by size
                sorted_attrs = sorted(
                    analysis.items(),
                    key=lambda x: x[1]['size_mb'],
                    reverse=True
                )
                
                removable_size = 0.0
                for attr_name, info in sorted_attrs:
                    if info['size_mb'] < 0.1:
                        continue
                    
                    status = "🔴 REMOVABLE" if info['removable'] else "✓"
                    if info['removable']:
                        removable_size += info['size_mb']
                    
                    logger.info(
                        f"  {attr_name}: {info['size_mb']:.1f} MB "
                        f"({info['type']}) {status}"
                    )
                
                if removable_size > 0:
                    logger.info(f"\n  Potential savings: {removable_size:.1f} MB "
                              f"({removable_size / file_size * 100:.1f}%)")
                
            except Exception as e:
                logger.error(f"  Failed to analyze: {e}")
    else:
        # Optimize models
        if args.model:
            model_path = args.models_dir / f"{args.model}.pkl"
            if not model_path.exists():
                model_path = args.models_dir / f"{args.model}.joblib"
            if not model_path.exists():
                logger.error(f"Model not found: {args.model}")
                sys.exit(1)
            
            # Optimize single model
            import joblib
            model = joblib.load(model_path) if model_path.suffix == '.joblib' else pickle.load(open(model_path, 'rb'))
            
            original_size = get_file_size_mb(model_path)
            optimized_model, report = optimize_model(model, args.model)
            
            output_dir = args.output_dir or args.models_dir
            output_path = output_dir / f"{args.model}.pkl"
            
            if not args.no_backup:
                backup_path = model_path.with_suffix('.pkl.backup')
                if not backup_path.exists():
                    shutil.copy(model_path, backup_path)
            
            optimized_size = save_optimized_model(optimized_model, output_path)
            
            logger.info(f"\nOptimized {args.model}:")
            logger.info(f"  Original: {original_size:.1f} MB")
            logger.info(f"  Optimized: {optimized_size:.1f} MB")
            logger.info(f"  Saved: {original_size - optimized_size:.1f} MB ({(1 - optimized_size / original_size) * 100:.1f}%)")
        else:
            # Optimize all
            reports = optimize_all_models(
                args.models_dir,
                args.output_dir,
                backup=not args.no_backup
            )


if __name__ == '__main__':
    main()
