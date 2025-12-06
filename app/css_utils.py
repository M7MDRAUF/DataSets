"""
CineMatch V2.1.6 - CSS Utilities

Utilities for loading and managing CSS styles in Streamlit.
Task 5.9: CSS Modularization

Author: CineMatch Development Team
Date: December 2025
"""

import streamlit as st
from pathlib import Path
from typing import Optional


# =============================================================================
# CSS FILE PATHS
# =============================================================================

STATIC_DIR = Path(__file__).parent / "static"
MAIN_CSS_PATH = STATIC_DIR / "main.css"


# =============================================================================
# CSS LOADING FUNCTIONS
# =============================================================================

def load_css_file(file_path: Path) -> str:
    """
    Load CSS content from a file.
    
    Args:
        file_path: Path to the CSS file
        
    Returns:
        CSS content as string
    """
    if not file_path.exists():
        return ""
    
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def inject_css(css_content: str) -> None:
    """
    Inject CSS into the Streamlit page.
    
    Args:
        css_content: CSS rules to inject
    """
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def load_main_css() -> None:
    """
    Load and inject the main CineMatch stylesheet.
    
    This function should be called at the top of each page to ensure
    consistent styling across the application.
    
    Example:
        import streamlit as st
        from app.css_utils import load_main_css
        
        load_main_css()
        st.title("My Page")
    """
    css = load_css_file(MAIN_CSS_PATH)
    if css:
        inject_css(css)


def get_css_variable(var_name: str) -> Optional[str]:
    """
    Get a CSS custom property value from the main stylesheet.
    
    Note: This parses the CSS file, not the computed value.
    
    Args:
        var_name: CSS variable name (e.g., 'color-primary')
        
    Returns:
        Variable value if found, None otherwise
    """
    css = load_css_file(MAIN_CSS_PATH)
    
    # Parse CSS variable
    import re
    pattern = rf"--{var_name}:\s*([^;]+);"
    match = re.search(pattern, css)
    
    return match.group(1).strip() if match else None


# =============================================================================
# CSS CLASSES REFERENCE
# =============================================================================

# Document available CSS classes for reference
CSS_CLASSES = {
    # Accessibility
    "skip-link": "Skip navigation link for keyboard users",
    "sr-only": "Screen reader only (visually hidden)",
    
    # Typography
    "main-header": "Page main header with gradient text",
    "sub-header": "Page subtitle",
    "section-title": "Section heading",
    
    # Cards
    "movie-card": "Movie recommendation card",
    "info-card": "Information card with blue border",
    "success-card": "Success message card with green border",
    "warning-card": "Warning message card with orange border",
    "error-card": "Error message card with red border",
    
    # Badges
    "badge": "Base badge class",
    "badge-primary": "Primary color badge",
    "badge-success": "Success color badge",
    "badge-warning": "Warning color badge",
    "badge-error": "Error color badge",
    "genre-tag": "Genre tag styling",
    
    # Progress
    "loading-spinner": "Animated loading spinner",
    "progress-bar": "Progress bar container",
    "progress-bar-fill": "Progress bar fill",
    
    # Tables
    "data-table": "Data table styling",
    
    # Ratings
    "rating-stars": "Star rating display",
    "rating-value": "Numeric rating value",
    
    # Metrics
    "metric-container": "Metric card container",
    "metric-value": "Large metric value",
    "metric-label": "Metric label text",
}


def get_css_classes_info() -> dict:
    """
    Get information about available CSS classes.
    
    Returns:
        Dictionary mapping class names to descriptions
    """
    return CSS_CLASSES.copy()


# =============================================================================
# THEME-AWARE CSS
# =============================================================================

def get_theme_css() -> str:
    """
    Get CSS that adapts to the current Streamlit theme.
    
    Returns:
        Theme-aware CSS string
    """
    # Try to detect dark mode from Streamlit config
    try:
        theme = st.get_option("theme.base")
        is_dark = theme == "dark"
    except Exception:
        is_dark = False
    
    if is_dark:
        return """
        :root {
            --color-text-dark: #ffffff;
            --color-text-muted: #aaaaaa;
            --color-bg-light: #1a1a1a;
            --color-bg-card: #2d2d2d;
        }
        """
    
    return ""


def apply_theme_css() -> None:
    """Apply theme-aware CSS overrides."""
    theme_css = get_theme_css()
    if theme_css:
        inject_css(theme_css)
