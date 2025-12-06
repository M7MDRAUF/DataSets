#!/usr/bin/env python3
"""
Model Slimming Utilities for CineMatch
Provides functions to create slim versions of models for faster loading.

These utilities help reduce model file sizes by:
1. Removing unnecessary training data (ratings_df, tags_df)
2. Keeping only inference-essential attributes
3. Converting to efficient storage formats
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Essential attributes for each model type (must keep for inference)
ESSENTIAL_ATTRS = {
    'SVDRecommender': {
        'user_factors',
        'movie_factors',  
        'global_mean',
        'user_biases',
        'movie_biases',
        'movie_mapper',
        'movie_inv_mapper',
        'user_mapper',
        'user_inv_mapper',
        'movie_ids',
        'user_ids',
        'movies_df',  # Needed for movie info display
        'is_trained',
        'name',
        'params',
        'n_factors',
        'metrics',
    },
    'ContentBasedRecommender': {
        'combined_features',
        'genre_features',
        'title_features',
        'tag_features',
        'genre_vectorizer',
        'title_vectorizer',
        'tag_vectorizer',
        'genre_weight',
        'title_weight',
        'tag_weight',
        'movie_mapper',
        'movie_inv_mapper',
        'movie_ids',
        'movies_df',
        'user_profiles',
        'min_similarity',
        'is_trained',
        'name',
        'params',
        'metrics',
    },
    'UserKNNRecommender': {
        'user_movie_matrix',
        'knn_model',
        'movie_mapper',
        'movie_inv_mapper',
        'user_mapper',
        'user_inv_mapper',
        'movie_ids',
        'user_ids',
        'movies_df',
        'user_means',
        'global_mean',
        'n_neighbors',
        'similarity_metric',
        'is_trained',
        'name',
        'params',
        'metrics',
    },
    'ItemKNNRecommender': {
        'item_user_matrix',
        'knn_model',
        'movie_mapper',
        'movie_inv_mapper',
        'movie_ids',
        'movies_df',
        'item_means',
        'global_mean',
        'n_neighbors',
        'min_ratings',
        'is_trained',
        'name',
        'params',
        'metrics',
    },
    'HybridRecommender': {
        'models',
        'weights',
        'model_names',
        'movie_ids',
        'movies_df',
        'is_trained',
        'name',
        'params',
        'metrics',
    },
}

# Attributes that should NEVER be in production models
BLACKLIST_ATTRS = {
    'ratings_df',
    'tags_df',
    'links_df',
    'genome_scores',
    'genome_tags',
    '_training_data',
    '_validation_data',
    '_test_data',
}


def get_model_type(model: Any) -> str:
    """Get the model type name, handling wrapped models."""
    if isinstance(model, dict) and 'model' in model:
        return type(model['model']).__name__
    return type(model).__name__


def get_inner_model(model: Any) -> Any:
    """Extract inner model from wrapper if present."""
    if isinstance(model, dict) and 'model' in model:
        return model['model']
    return model


def wrap_model(inner_model: Any, original_wrapper: Optional[Dict] = None) -> Any:
    """Re-wrap model if it was originally wrapped."""
    if original_wrapper is not None and isinstance(original_wrapper, dict):
        result = original_wrapper.copy()
        result['model'] = inner_model
        return result
    return inner_model


def create_slim_model(model: Any) -> Any:
    """
    Create a slim version of the model with only essential attributes.
    
    Args:
        model: The model to slim (can be wrapped in dict)
        
    Returns:
        Slimmed model (same wrapper structure if applicable)
    """
    # Handle wrapped models
    wrapper = model if isinstance(model, dict) and 'model' in model else None
    inner_model = get_inner_model(model)
    model_type = type(inner_model).__name__
    
    # Get essential attributes for this model type
    essential = ESSENTIAL_ATTRS.get(model_type, set())
    
    # Create list of attributes to remove
    attrs_to_remove = []
    for attr_name in dir(inner_model):
        if attr_name.startswith('_'):
            continue
        if callable(getattr(inner_model, attr_name, None)):
            continue
        
        # Remove if blacklisted or not essential
        if attr_name in BLACKLIST_ATTRS:
            attrs_to_remove.append(attr_name)
        elif essential and attr_name not in essential:
            # Only remove if we have a whitelist and attr not in it
            attrs_to_remove.append(attr_name)
    
    # Remove attributes
    for attr_name in attrs_to_remove:
        if hasattr(inner_model, attr_name):
            try:
                setattr(inner_model, attr_name, None)
            except AttributeError:
                # Some attributes may be read-only
                pass
    
    return wrap_model(inner_model, wrapper)


def validate_slim_model(model: Any) -> Dict[str, Any]:
    """
    Validate that a slim model has all essential attributes.
    
    Returns:
        Dict with 'valid' bool, 'missing' list, 'warnings' list
    """
    inner_model = get_inner_model(model)
    model_type = type(inner_model).__name__
    
    result = {
        'valid': True,
        'missing': [],
        'warnings': [],
        'model_type': model_type
    }
    
    essential = ESSENTIAL_ATTRS.get(model_type, set())
    
    # Check essential attributes
    for attr_name in essential:
        if not hasattr(inner_model, attr_name):
            result['missing'].append(attr_name)
            result['valid'] = False
        elif getattr(inner_model, attr_name, None) is None:
            # Attribute exists but is None
            if attr_name in {'is_trained', 'movie_mapper', 'movie_inv_mapper'}:
                result['missing'].append(f"{attr_name} (is None)")
                result['valid'] = False
            else:
                result['warnings'].append(f"{attr_name} is None")
    
    # Check for blacklisted attributes that shouldn't be there
    for attr_name in BLACKLIST_ATTRS:
        if hasattr(inner_model, attr_name):
            attr_val = getattr(inner_model, attr_name, None)
            if attr_val is not None:
                result['warnings'].append(f"Contains blacklisted attr: {attr_name}")
    
    return result


def estimate_slim_size(model: Any) -> Dict[str, float]:
    """
    Estimate the size of a model before and after slimming.
    
    Returns:
        Dict with 'original_mb', 'slim_mb', 'savings_mb', 'savings_percent'
    """
    import sys
    from scipy import sparse
    
    def get_attr_size(obj: Any) -> float:
        """Get size of an attribute in MB."""
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
                return sys.getsizeof(obj) / (1024 * 1024)
        except Exception:
            return 0.0
    
    inner_model = get_inner_model(model)
    model_type = type(inner_model).__name__
    essential = ESSENTIAL_ATTRS.get(model_type, set())
    
    original_size = 0.0
    slim_size = 0.0
    
    for attr_name in dir(inner_model):
        if attr_name.startswith('_'):
            continue
        if callable(getattr(inner_model, attr_name, None)):
            continue
        
        attr_val = getattr(inner_model, attr_name, None)
        attr_size = get_attr_size(attr_val)
        original_size += attr_size
        
        # Check if it would be kept
        if attr_name not in BLACKLIST_ATTRS:
            if not essential or attr_name in essential:
                slim_size += attr_size
    
    return {
        'original_mb': original_size,
        'slim_mb': slim_size,
        'savings_mb': original_size - slim_size,
        'savings_percent': ((original_size - slim_size) / original_size * 100) if original_size > 0 else 0
    }


def get_removable_attrs(model: Any) -> List[Dict[str, Any]]:
    """
    Get list of removable attributes with their sizes.
    
    Returns:
        List of dicts with 'name', 'size_mb', 'reason'
    """
    from scipy import sparse
    
    def get_attr_size(obj: Any) -> float:
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
        except Exception:
            pass
        return 0.0
    
    inner_model = get_inner_model(model)
    model_type = type(inner_model).__name__
    essential = ESSENTIAL_ATTRS.get(model_type, set())
    
    removable = []
    
    for attr_name in dir(inner_model):
        if attr_name.startswith('_'):
            continue
        if callable(getattr(inner_model, attr_name, None)):
            continue
        
        attr_val = getattr(inner_model, attr_name, None)
        if attr_val is None:
            continue
        
        size_mb = get_attr_size(attr_val)
        if size_mb < 0.1:
            continue
        
        # Determine if removable
        if attr_name in BLACKLIST_ATTRS:
            removable.append({
                'name': attr_name,
                'size_mb': size_mb,
                'reason': 'Blacklisted - not needed for inference'
            })
        elif essential and attr_name not in essential:
            removable.append({
                'name': attr_name,
                'size_mb': size_mb,
                'reason': 'Not in essential attributes list'
            })
    
    return sorted(removable, key=lambda x: x['size_mb'], reverse=True)


# Convenience functions
def slim_svd_model(model: Any) -> Any:
    """Slim an SVD model."""
    return create_slim_model(model)


def slim_content_based_model(model: Any) -> Any:
    """Slim a ContentBased model."""
    return create_slim_model(model)


def slim_user_knn_model(model: Any) -> Any:
    """Slim a UserKNN model."""
    return create_slim_model(model)


def slim_item_knn_model(model: Any) -> Any:
    """Slim an ItemKNN model."""
    return create_slim_model(model)


def slim_hybrid_model(model: Any) -> Any:
    """Slim a Hybrid model."""
    return create_slim_model(model)


if __name__ == '__main__':
    # Test the utilities
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Model Slimming Utilities")
    print("=" * 60)
    print("\nEssential attributes per model type:")
    for model_type, attrs in ESSENTIAL_ATTRS.items():
        print(f"\n{model_type}:")
        for attr in sorted(attrs):
            print(f"  - {attr}")
    
    print("\n\nBlacklisted attributes (always removed):")
    for attr in sorted(BLACKLIST_ATTRS):
        print(f"  - {attr}")
