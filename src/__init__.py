"""
CineMatch V1.0.0 - Package Initialization

Exports key functions for easy importing.
"""

__version__ = "1.0.0"
__author__ = "CineMatch Team"

from .data_processing import (
    check_data_integrity,
    load_ratings,
    load_movies,
    load_links,
    load_tags
)

__all__ = [
    'check_data_integrity',
    'load_ratings',
    'load_movies',
    'load_links',
    'load_tags'
]
