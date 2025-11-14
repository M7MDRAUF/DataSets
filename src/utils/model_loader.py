"""
CineMatch V2.1.1 - Model Loading Utilities

Helper functions to handle different model serialization formats.
Provides backward compatibility for dict-wrapped models.

Author: CineMatch Development Team
Date: November 14, 2025
"""

import pickle
from pathlib import Path
from typing import Any, Union


def load_model_safe(model_path: Union[str, Path]) -> Any:
    """
    Safely load a model, handling both direct instance and dict wrapper formats.
    
    Handles two serialization formats:
    1. Direct model instance: pickle.dump(model, f)
    2. Dict wrapper: pickle.dump({'model': model, 'metrics': {...}}, f)
    
    Args:
        model_path: Path to pickled model file
        
    Returns:
        Model instance (unwrapped if necessary)
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If loading fails
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        loaded_data = pickle.load(f)
    
    # Handle dict wrapper format (Content-Based model)
    if isinstance(loaded_data, dict) and 'model' in loaded_data:
        return loaded_data['model']
    
    # Direct instance format (all other models)
    return loaded_data


def get_model_metadata(model_path: Union[str, Path]) -> dict:
    """
    Extract metadata from model file if available.
    
    Args:
        model_path: Path to pickled model file
        
    Returns:
        Dict with metadata, or empty dict if not available
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        return {}
    
    try:
        with open(model_path, 'rb') as f:
            loaded_data = pickle.load(f)
        
        if isinstance(loaded_data, dict):
            return {
                'metrics': loaded_data.get('metrics', {}),
                'metadata': loaded_data.get('metadata', {}),
                'format': 'dict_wrapper'
            }
        else:
            return {
                'format': 'direct_instance',
                'type': type(loaded_data).__name__
            }
    
    except Exception as e:
        return {'error': str(e)}


def save_model_standard(model: Any, model_path: Union[str, Path]) -> None:
    """
    Save model using standard format (direct instance).
    
    Args:
        model: Trained model instance
        model_path: Path to save model
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
