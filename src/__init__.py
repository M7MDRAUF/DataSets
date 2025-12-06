"""
CineMatch V2.1.6 - Package Initialization

Exports key functions for easy importing.
Single source of truth for version is VERSION file in project root.
"""

from pathlib import Path

# Read version from VERSION file (single source of truth)
_version_file = Path(__file__).parent.parent / "VERSION"
if _version_file.exists():
    __version__ = _version_file.read_text().strip()
else:
    __version__ = "2.1.6"  # Fallback

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
