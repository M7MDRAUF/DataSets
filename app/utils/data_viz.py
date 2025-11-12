"""
CineMatch V2.1 - Data Visualization Utilities

Pure Python/CSS data-driven visualizations using local MovieLens dataset.
NO external API calls - everything generated from movies.csv and ratings.csv.

Author: CineMatch Team
Date: November 11, 2025
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import get_genre_emoji, create_genre_color_map


def get_movie_color_scheme(genres: List[str]) -> Dict[str, str]:
    """
    Get a color scheme for a movie based on its genres.
    
    Args:
        genres: List of genre strings
        
    Returns:
        Dictionary with primary_color, secondary_color, gradient
    """
    genre_colors = create_genre_color_map()
    
    if not genres or len(genres) == 0:
        return {
            'primary_color': '#666666',
            'secondary_color': '#444444',
            'gradient': 'linear-gradient(135deg, #666666 0%, #444444 100%)'
        }
    
    # Get primary color from first genre
    primary = genre_colors.get(genres[0], '#666666')
    
    # Get secondary color from second genre if available
    if len(genres) > 1:
        secondary = genre_colors.get(genres[1], primary)
    else:
        # Darken primary color for secondary
        secondary = primary
    
    gradient = f"linear-gradient(135deg, {primary} 0%, {secondary} 100%)"
    
    return {
        'primary_color': primary,
        'secondary_color': secondary,
        'gradient': gradient
    }


def create_genre_badge_html(genre: str, size: str = 'normal') -> str:
    """
    Create HTML for a genre badge with emoji and color.
    
    Args:
        genre: Genre name
        size: 'small', 'normal', or 'large'
        
    Returns:
        HTML string for the badge
    """
    emoji = get_genre_emoji(genre)
    colors = create_genre_color_map()
    color = colors.get(genre, '#666666')
    
    size_styles = {
        'small': 'font-size: 0.75rem; padding: 0.15rem 0.4rem;',
        'normal': 'font-size: 0.85rem; padding: 0.25rem 0.6rem;',
        'large': 'font-size: 1rem; padding: 0.35rem 0.8rem;'
    }
    
    style = size_styles.get(size, size_styles['normal'])
    
    return f"""
    <span class="genre-badge" style="
        background-color: {color};
        color: white;
        border-radius: 12px;
        {style}
        margin: 0.2rem;
        display: inline-block;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    ">
        {emoji} {genre}
    </span>
    """


def create_rating_visual(rating: float, style: str = 'stars') -> str:
    """
    Create visual representation of rating.
    
    Args:
        rating: Rating value (0.5 to 5.0)
        style: 'stars', 'meter', or 'number'
        
    Returns:
        HTML string for rating visualization
    """
    if style == 'stars':
        full_stars = int(rating)
        half_star = 1 if (rating - full_stars) >= 0.5 else 0
        empty_stars = 5 - full_stars - half_star
        
        stars_html = '⭐' * full_stars
        if half_star:
            stars_html += '✨'
        stars_html += '☆' * empty_stars
        
        return f"""
        <span class="rating-stars" style="
            font-size: 1.2rem;
            color: #FFD700;
            letter-spacing: 2px;
        ">
            {stars_html}
        </span>
        <span style="color: #888; font-size: 0.9rem; margin-left: 0.5rem;">
            {rating:.1f}/5.0
        </span>
        """
    
    elif style == 'meter':
        percentage = (rating / 5.0) * 100
        color = '#4CAF50' if rating >= 4.0 else '#FFC107' if rating >= 3.0 else '#F44336'
        
        return f"""
        <div class="rating-meter" style="
            width: 100%;
            height: 20px;
            background-color: #333;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        ">
            <div style="
                width: {percentage}%;
                height: 100%;
                background: linear-gradient(90deg, {color} 0%, {color}AA 100%);
                transition: width 0.3s ease;
            "></div>
            <span style="
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                color: white;
                font-size: 0.8rem;
                font-weight: bold;
            ">{rating:.1f}</span>
        </div>
        """
    
    else:  # number
        return f"""
        <span class="rating-number" style="
            font-size: 1.5rem;
            font-weight: bold;
            color: #E50914;
        ">
            {rating:.1f}
        </span>
        <span style="color: #888; font-size: 1rem;">/5.0</span>
        """


def generate_movie_card_gradient(genres: List[str], title: str = "") -> str:
    """
    Generate CSS gradient background for movie card based on genres.
    
    Args:
        genres: List of genre names
        title: Movie title (for additional styling variation)
        
    Returns:
        CSS background string
    """
    colors = get_movie_color_scheme(genres)
    
    # Add subtle overlay for better text readability
    overlay = "linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.6))"
    
    return f"{overlay}, {colors['gradient']}"


def create_score_meter(score: float, label: str = "Match") -> str:
    """
    Create a visual score meter (predicted rating, match percentage, etc.)
    
    Args:
        score: Score value (typically 0-5 or 0-100)
        label: Label for the score
        
    Returns:
        HTML string for score meter
    """
    # Normalize to percentage
    if score <= 5:
        percentage = (score / 5.0) * 100
        display_score = f"{score:.1f}"
    else:
        percentage = min(score, 100)
        display_score = f"{score:.0f}%"
    
    # Color based on score
    if percentage >= 80:
        color = '#4CAF50'  # Green
    elif percentage >= 60:
        color = '#FFC107'  # Yellow
    else:
        color = '#F44336'  # Red
    
    return f"""
    <div class="score-meter-container" style="margin: 0.5rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
            <span style="color: #CCC; font-size: 0.85rem;">{label}</span>
            <span style="color: {color}; font-weight: bold; font-size: 0.9rem;">{display_score}</span>
        </div>
        <div style="
            width: 100%;
            height: 8px;
            background-color: #333;
            border-radius: 4px;
            overflow: hidden;
        ">
            <div style="
                width: {percentage}%;
                height: 100%;
                background: {color};
                transition: width 0.5s ease;
                box-shadow: 0 0 10px {color}AA;
            "></div>
        </div>
    </div>
    """


def extract_year_from_title(title: str) -> Optional[int]:
    """
    Extract year from movie title.
    
    Args:
        title: Movie title (e.g., "Toy Story (1995)")
        
    Returns:
        Year as integer, or None if not found
    """
    import re
    match = re.search(r'\((\d{4})\)$', title.strip())
    return int(match.group(1)) if match else None


def format_movie_title_clean(title: str) -> str:
    """
    Remove year from movie title for cleaner display.
    
    Args:
        title: Full movie title with year
        
    Returns:
        Title without year
    """
    import re
    return re.sub(r'\s*\(\d{4}\)$', '', title).strip()


def create_popularity_indicator(num_ratings: int) -> str:
    """
    Create visual indicator of movie popularity based on number of ratings.
    
    Args:
        num_ratings: Number of ratings the movie has
        
    Returns:
        HTML string for popularity indicator
    """
    # Determine popularity level
    if num_ratings >= 10000:
        level = "🔥 Very Popular"
        color = "#FF6B6B"
    elif num_ratings >= 5000:
        level = "⭐ Popular"
        color = "#FFC107"
    elif num_ratings >= 1000:
        level = "👍 Well-Known"
        color = "#4CAF50"
    else:
        level = "💎 Hidden Gem"
        color = "#9C27B0"
    
    return f"""
    <div style="
        display: inline-block;
        background: {color}22;
        border-left: 3px solid {color};
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin: 0.3rem 0;
    ">
        <span style="color: {color}; font-weight: 600;">{level}</span>
        <span style="color: #999; margin-left: 0.5rem;">({num_ratings:,} ratings)</span>
    </div>
    """
