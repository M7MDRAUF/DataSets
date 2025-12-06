"""
CineMatch Utilities Package

Helper functions and utilities for model loading and data processing.
"""

# Import from utils/ submodules (new memory optimization modules)
from .model_loader import (
    load_model_safe, 
    get_model_metadata, 
    save_model_standard,
    load_model_fast,
    save_model_fast,
    detect_model_format,
    warm_up_model,
    load_and_warm_up,
    JOBLIB_AVAILABLE
)
from .memory_manager import (
    load_model_with_gc,
    load_models_sequential,
    aggressive_gc,
    get_memory_usage_mb,
    estimate_model_memory_requirement,
    print_memory_report
)

# Import from src.utils module (legacy utility functions for Streamlit)
# Using absolute import to avoid circular reference
import sys
from pathlib import Path

# Get the src directory
_src_dir = Path(__file__).parent.parent
_utils_module_path = _src_dir / 'utils.py'

# Import directly from utils.py file
if _utils_module_path.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("src_utils_legacy", _utils_module_path)
    _utils_legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_utils_legacy)
    
    # Extract functions we need
    format_genres = _utils_legacy.format_genres
    format_rating = _utils_legacy.format_rating
    calculate_diversity_score = _utils_legacy.calculate_diversity_score
    get_recommendation_stats = _utils_legacy.get_recommendation_stats
    extract_year_from_title = _utils_legacy.extract_year_from_title
    format_movie_title = _utils_legacy.format_movie_title
    create_genre_color_map = _utils_legacy.create_genre_color_map
    get_genre_emoji = _utils_legacy.get_genre_emoji
    create_rating_stars = _utils_legacy.create_rating_stars
    
    # TMDB Image utilities (added for poster support)
    get_tmdb_poster_url = _utils_legacy.get_tmdb_poster_url
    get_tmdb_backdrop_url = _utils_legacy.get_tmdb_backdrop_url
    create_movie_card_html = _utils_legacy.create_movie_card_html
    create_compact_movie_card_html = _utils_legacy.create_compact_movie_card_html
    TMDB_IMAGE_BASE_URL = _utils_legacy.TMDB_IMAGE_BASE_URL
    TMDB_BACKDROP_BASE_URL = _utils_legacy.TMDB_BACKDROP_BASE_URL
    PLACEHOLDER_POSTER = _utils_legacy.PLACEHOLDER_POSTER
    PLACEHOLDER_BACKDROP = _utils_legacy.PLACEHOLDER_BACKDROP

__all__ = [
    # Model loading utilities (new)
    'load_model_safe', 
    'get_model_metadata', 
    'save_model_standard',
    'load_model_fast',
    'save_model_fast',
    'detect_model_format',
    'warm_up_model',
    'load_and_warm_up',
    'JOBLIB_AVAILABLE',
    'load_model_with_gc',
    'load_models_sequential',
    'aggressive_gc',
    'get_memory_usage_mb',
    'estimate_model_memory_requirement',
    'print_memory_report',
    # Streamlit display utilities (legacy from utils.py)
    'format_genres',
    'format_rating',
    'calculate_diversity_score',
    'get_recommendation_stats',
    'extract_year_from_title',
    'format_movie_title',
    'create_genre_color_map',
    'get_genre_emoji',
    'create_rating_stars',
    # TMDB Image utilities
    'get_tmdb_poster_url',
    'get_tmdb_backdrop_url',
    'create_movie_card_html',
    'create_compact_movie_card_html',
    'TMDB_IMAGE_BASE_URL',
    'TMDB_BACKDROP_BASE_URL',
    'PLACEHOLDER_POSTER',
    'PLACEHOLDER_BACKDROP'
]
