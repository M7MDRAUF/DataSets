#!/usr/bin/env python3
"""
Model Validation Utilities for CineMatch
Provides size validation, integrity checks, and warnings for trained models.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# Size thresholds (in MB)
MODEL_SIZE_LIMITS = {
    'svd_model_sklearn.pkl': {
        'max': 400,      # Maximum acceptable size
        'warn': 350,     # Warning threshold
        'expected': 300, # Expected size after optimization
    },
    'content_based_model.pkl': {
        'max': 100,
        'warn': 50,
        'expected': 30,
    },
    'user_knn_model.pkl': {
        'max': 800,
        'warn': 700,
        'expected': 650,
    },
    'item_knn_model.pkl': {
        'max': 800,
        'warn': 700,
        'expected': 650,
    },
    'hybrid_model.pkl': {
        'max': 50,
        'warn': 30,
        'expected': 10,
    },
    # Default for unknown models
    'default': {
        'max': 500,
        'warn': 300,
        'expected': 200,
    }
}

# Attributes that should NOT be in production models
BLACKLIST_ATTRS = {
    'ratings_df': 'Raw ratings data - not needed for inference',
    'tags_df': 'Raw tags data - only needed during training',
    'links_df': 'Movie links - not needed for recommendations',
    '_training_data': 'Training data should be removed after training',
    '_validation_data': 'Validation data should be removed',
    '_test_data': 'Test data should be removed',
}

# Essential attributes that MUST be present
ESSENTIAL_ATTRS = {
    'SVDRecommender': ['is_trained', 'model', 'movie_ids', 'user_ids'],
    'SimpleSVDRecommender': ['is_trained', 'user_factors', 'movie_factors', 'movie_mapper'],
    'ContentBasedRecommender': ['is_trained', 'combined_features', 'movie_mapper'],
    'UserKNNRecommender': ['is_trained', 'knn_model', 'movie_mapper', 'user_movie_matrix'],
    'ItemKNNRecommender': ['is_trained', 'item_user_matrix', 'movie_mapper', 'knn_model'],
    'HybridRecommender': ['is_trained', 'models', 'weights'],
}


class ModelValidationResult:
    """Result of model validation."""
    
    def __init__(self):
        self.valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: Dict[str, Any] = {}
    
    def add_error(self, message: str):
        self.errors.append(message)
        self.valid = False
    
    def add_warning(self, message: str):
        self.warnings.append(message)
    
    def __str__(self) -> str:
        lines = []
        if self.valid:
            lines.append("✅ Model validation PASSED")
        else:
            lines.append("❌ Model validation FAILED")
        
        if self.errors:
            lines.append("\nErrors:")
            for e in self.errors:
                lines.append(f"  ❌ {e}")
        
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ⚠️ {w}")
        
        if self.info:
            lines.append("\nInfo:")
            for k, v in self.info.items():
                lines.append(f"  • {k}: {v}")
        
        return "\n".join(lines)


def get_file_size_mb(path: Path) -> float:
    """Get file size in MB."""
    return path.stat().st_size / (1024 * 1024)


def get_object_memory_mb(obj: Any) -> float:
    """Estimate object memory usage in MB."""
    from scipy import sparse
    
    if obj is None:
        return 0.0
    
    try:
        if hasattr(obj, 'memory_usage'):
            return obj.memory_usage(deep=True).sum() / (1024 * 1024)
        elif hasattr(obj, 'nbytes'):
            return obj.nbytes / (1024 * 1024)
        elif sparse.issparse(obj):
            size = obj.data.nbytes + obj.indices.nbytes
            if hasattr(obj, 'indptr'):
                size += obj.indptr.nbytes
            return size / (1024 * 1024)
        else:
            import sys
            return sys.getsizeof(obj) / (1024 * 1024)
    except Exception:
        return 0.0


def validate_model_size(model_path: Path) -> ModelValidationResult:
    """
    Validate model file size against thresholds.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        ModelValidationResult with size validation results
    """
    result = ModelValidationResult()
    
    if not model_path.exists():
        result.add_error(f"Model file not found: {model_path}")
        return result
    
    size_mb = get_file_size_mb(model_path)
    result.info['file_size_mb'] = f"{size_mb:.1f}"
    
    # Get limits for this model
    model_name = model_path.name
    limits = MODEL_SIZE_LIMITS.get(model_name, MODEL_SIZE_LIMITS['default'])
    
    if size_mb > limits['max']:
        result.add_error(
            f"Model size ({size_mb:.1f}MB) exceeds maximum ({limits['max']}MB). "
            f"Run optimize_models.py to strip unnecessary data."
        )
    elif size_mb > limits['warn']:
        result.add_warning(
            f"Model size ({size_mb:.1f}MB) exceeds warning threshold ({limits['warn']}MB). "
            f"Consider optimizing with optimize_models.py."
        )
    else:
        result.info['size_status'] = f"OK (expected ~{limits['expected']}MB)"
    
    return result


def validate_model_attributes(model: Any, model_name: Optional[str] = None) -> ModelValidationResult:
    """
    Validate model has required attributes and doesn't have blacklisted ones.
    
    Args:
        model: The model to validate
        model_name: Optional name for better error messages
        
    Returns:
        ModelValidationResult with attribute validation results
    """
    result = ModelValidationResult()
    
    # Handle wrapped models
    if isinstance(model, dict) and 'model' in model:
        model = model['model']
    
    model_type = type(model).__name__
    result.info['model_type'] = model_type
    
    # Check for blacklisted attributes
    for attr_name, reason in BLACKLIST_ATTRS.items():
        if hasattr(model, attr_name):
            attr_val = getattr(model, attr_name, None)
            if attr_val is not None:
                size_mb = get_object_memory_mb(attr_val)
                if size_mb > 1.0:  # Only warn if significant size
                    result.add_warning(
                        f"Contains unnecessary attribute '{attr_name}' ({size_mb:.1f}MB). {reason}"
                    )
    
    # Check for essential attributes
    essential = ESSENTIAL_ATTRS.get(model_type, [])
    for attr_name in essential:
        if not hasattr(model, attr_name):
            result.add_error(f"Missing essential attribute: {attr_name}")
        elif getattr(model, attr_name, None) is None:
            if attr_name == 'is_trained':
                result.add_error(f"Model is_trained flag is None/False")
            else:
                result.add_warning(f"Essential attribute '{attr_name}' is None")
    
    # Check is_trained
    if hasattr(model, 'is_trained'):
        if not model.is_trained:
            result.add_error("Model is not marked as trained")
        else:
            result.info['trained'] = "Yes"
    
    return result


def validate_model_integrity(model_path: Path) -> ModelValidationResult:
    """
    Perform full integrity validation on a model file.
    
    Checks:
    1. File exists and is readable
    2. File size is within limits
    3. Model can be loaded
    4. Model has required attributes
    5. Model doesn't have blacklisted attributes
    
    Args:
        model_path: Path to the model file
        
    Returns:
        ModelValidationResult with complete validation results
    """
    result = ModelValidationResult()
    
    if not model_path.exists():
        result.add_error(f"Model file not found: {model_path}")
        return result
    
    result.info['path'] = str(model_path)
    
    # Size validation
    size_result = validate_model_size(model_path)
    result.errors.extend(size_result.errors)
    result.warnings.extend(size_result.warnings)
    result.info.update(size_result.info)
    if not size_result.valid:
        result.valid = False
    
    # Try to load the model
    try:
        import joblib
        # Add project root to path for model class imports
        import sys
        project_root_str = str(Path(__file__).parent.parent.parent)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
        
        model = joblib.load(model_path)
        result.info['load_status'] = "Success"
    except Exception as e:
        result.add_error(f"Failed to load model: {e}")
        return result
    
    # Attribute validation
    attr_result = validate_model_attributes(model, model_path.name)
    result.errors.extend(attr_result.errors)
    result.warnings.extend(attr_result.warnings)
    result.info.update(attr_result.info)
    if not attr_result.valid:
        result.valid = False
    
    return result


def validate_all_models(models_dir: Path) -> Dict[str, ModelValidationResult]:
    """
    Validate all models in a directory.
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        Dict mapping model names to their validation results
    """
    results = {}
    
    model_files = list(models_dir.glob('*.pkl')) + list(models_dir.glob('*.joblib'))
    
    for model_path in model_files:
        if '_backup' in model_path.name:
            continue
        
        results[model_path.name] = validate_model_integrity(model_path)
    
    return results


def assert_model_valid(model_path: Path, strict: bool = False) -> None:
    """
    Assert that a model passes validation. Raises exception on failure.
    
    Args:
        model_path: Path to the model file
        strict: If True, treat warnings as errors
        
    Raises:
        ValueError: If model fails validation
        UserWarning: If model has warnings (non-strict mode)
    """
    result = validate_model_integrity(model_path)
    
    if not result.valid:
        raise ValueError(f"Model validation failed for {model_path.name}:\n{result}")
    
    if result.warnings:
        warning_msg = f"Model {model_path.name} has warnings:\n" + "\n".join(f"  • {w}" for w in result.warnings)
        if strict:
            raise ValueError(warning_msg)
        else:
            warnings.warn(warning_msg, UserWarning)


def print_validation_report(models_dir: Path) -> bool:
    """
    Print a validation report for all models.
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        True if all models pass, False otherwise
    """
    results = validate_all_models(models_dir)
    
    print("\n" + "=" * 60)
    print("MODEL VALIDATION REPORT")
    print("=" * 60)
    
    all_valid = True
    total_size = 0.0
    
    for model_name, result in sorted(results.items()):
        print(f"\n{model_name}:")
        print("-" * 40)
        
        size_mb = float(result.info.get('file_size_mb', 0))
        total_size += size_mb
        
        if result.valid and not result.warnings:
            print(f"  ✅ VALID ({size_mb:.1f}MB)")
        elif result.valid:
            print(f"  ⚠️ VALID with warnings ({size_mb:.1f}MB)")
            for w in result.warnings:
                print(f"     {w}")
        else:
            print(f"  ❌ INVALID")
            all_valid = False
            for e in result.errors:
                print(f"     ERROR: {e}")
            for w in result.warnings:
                print(f"     WARN: {w}")
    
    print("\n" + "=" * 60)
    print(f"Total model size: {total_size:.1f}MB")
    print(f"Overall status: {'✅ ALL VALID' if all_valid else '❌ SOME INVALID'}")
    print("=" * 60)
    
    return all_valid


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate CineMatch models')
    parser.add_argument(
        '--models-dir',
        type=Path,
        default=project_root / 'models',
        help='Directory containing model files'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Validate specific model only'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Treat warnings as errors'
    )
    
    args = parser.parse_args()
    
    if args.model:
        model_path = args.models_dir / args.model
        if not model_path.suffix:
            model_path = model_path.with_suffix('.pkl')
        
        result = validate_model_integrity(model_path)
        print(result)
        sys.exit(0 if result.valid and (not args.strict or not result.warnings) else 1)
    else:
        valid = print_validation_report(args.models_dir)
        sys.exit(0 if valid else 1)
