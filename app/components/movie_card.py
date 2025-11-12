"""
CineMatch V2.1 - Enhanced Movie Card Component

Beautiful, data-driven movie cards with genre-based gradients.
NO external APIs - everything from local dataset.
"""

import streamlit as st
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.utils.data_viz import (
    get_movie_color_scheme,
    create_genre_badge_html,
    create_rating_visual,
    create_score_meter,
    format_movie_title_clean,
    extract_year_from_title,
    create_popularity_indicator
)


def render_movie_card_enhanced(
    title: str,
    genres: List[str],
    avg_rating: Optional[float] = None,
    predicted_rating: Optional[float] = None,
    num_ratings: Optional[int] = None,
    match_score: Optional[float] = None,
    rank: Optional[int] = None,
    explanation: Optional[str] = None,
    compact: bool = False
) -> None:
    """
    Render an enhanced movie card with genre-based styling.
    
    Args:
        title: Movie title (with year)
        genres: List of genre names
        avg_rating: Average rating from dataset
        predicted_rating: Predicted rating for user
        num_ratings: Number of ratings in dataset
        match_score: Match percentage (0-100)
        rank: Recommendation rank (1, 2, 3...)
        explanation: Why this movie was recommended
        compact: Use compact layout for grid views
    """
    # Parse title and year
    clean_title = format_movie_title_clean(title)
    year = extract_year_from_title(title)
    
    # Get color scheme from genres
    colors = get_movie_color_scheme(genres)
    
    # Build card HTML
    if compact:
        _render_compact_card(
            clean_title, year, genres, colors,
            avg_rating, predicted_rating, match_score, rank
        )
    else:
        _render_full_card(
            clean_title, year, genres, colors,
            avg_rating, predicted_rating, num_ratings,
            match_score, rank, explanation
        )


def _render_full_card(
    title: str,
    year: Optional[int],
    genres: List[str],
    colors: Dict[str, str],
    avg_rating: Optional[float],
    predicted_rating: Optional[float],
    num_ratings: Optional[int],
    match_score: Optional[float],
    rank: Optional[int],
    explanation: Optional[str]
) -> None:
    """Render full-sized movie card with all details."""
    
    # Card container with gradient background
    card_html = f"""
    <div class="movie-card-enhanced" style="
        background: {colors['gradient']};
        position: relative;
    ">
    """
    
    # Rank badge (top-right corner)
    if rank is not None:
        rank_color = "#FFD700" if rank <= 3 else "#E50914"
        card_html += f"""
        <div style="
            position: absolute;
            top: 1rem;
            right: 1rem;
            background: {rank_color};
            color: black;
            font-size: 1.5rem;
            font-weight: bold;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        ">
            #{rank}
        </div>
        """
    
    # Title and year
    year_html = f'<span class="movie-year">({year})</span>' if year else ''
    card_html += f"""
    <div class="movie-title-large">
        {title} {year_html}
    </div>
    """
    
    # Genre badges
    card_html += '<div class="genre-container">'
    for genre in genres:
        card_html += create_genre_badge_html(genre, size='normal')
    card_html += '</div>'
    
    # Ratings section
    if avg_rating or predicted_rating:
        card_html += '<div style="margin: 1.5rem 0;">'
        
        if predicted_rating:
            card_html += f'<div style="margin-bottom: 1rem;">'
            card_html += f'<strong style="color: #FFD700;">Predicted for You:</strong>'
            card_html += create_rating_visual(predicted_rating, style='stars')
            card_html += '</div>'
        
        if avg_rating:
            card_html += f'<div>'
            card_html += f'<strong style="color: #CCC;">Community Rating:</strong>'
            card_html += create_rating_visual(avg_rating, style='stars')
            card_html += '</div>'
        
        card_html += '</div>'
    
    # Match score meter
    if match_score is not None:
        card_html += create_score_meter(match_score, label="Match")
    
    # Popularity indicator
    if num_ratings is not None:
        card_html += create_popularity_indicator(num_ratings)
    
    # Explanation (if provided)
    if explanation:
        card_html += f"""
        <div style="
            margin-top: 1.5rem;
            padding: 1rem;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            border-left: 3px solid {colors['primary_color']};
        ">
            <div style="color: #CCC; font-size: 0.85rem; margin-bottom: 0.5rem;">
                💡 Why this recommendation?
            </div>
            <div style="color: #EEE; font-size: 0.95rem; line-height: 1.5;">
                {explanation}
            </div>
        </div>
        """
    
    card_html += "</div>"
    
    # Render
    st.markdown(card_html, unsafe_allow_html=True)


def _render_compact_card(
    title: str,
    year: Optional[int],
    genres: List[str],
    colors: Dict[str, str],
    avg_rating: Optional[float],
    predicted_rating: Optional[float],
    match_score: Optional[float],
    rank: Optional[int]
) -> None:
    """Render compact movie card for grid layouts."""
    
    # Compact card
    year_html = f'<span class="movie-year">({year})</span>' if year else ''
    
    card_html = f"""
    <div class="movie-card-enhanced" style="
        background: {colors['gradient']};
        padding: 1rem;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    ">
    """
    
    # Rank badge (smaller)
    if rank is not None:
        card_html += f"""
        <div style="
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            background: #E50914;
            color: white;
            font-size: 0.9rem;
            font-weight: bold;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
        ">
            #{rank}
        </div>
        """
    
    # Title (smaller)
    card_html += f"""
    <div style="
        font-size: 1.3rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    ">
        {title} {year_html}
    </div>
    """
    
    # Genres (badges smaller)
    card_html += '<div style="margin: 0.5rem 0;">'
    for genre in genres[:3]:  # Max 3 genres in compact mode
        card_html += create_genre_badge_html(genre, size='small')
    card_html += '</div>'
    
    # Rating (compact)
    rating_to_show = predicted_rating if predicted_rating else avg_rating
    if rating_to_show:
        card_html += f"""
        <div style="margin-top: auto;">
            <div style="font-size: 1.5rem; color: #FFD700;">
                {'⭐' * int(rating_to_show)}
            </div>
            <div style="color: #CCC; font-size: 0.85rem;">
                {rating_to_show:.1f}/5.0
            </div>
        </div>
        """
    
    # Match score (compact)
    if match_score is not None:
        percentage = min(match_score, 100)
        color = '#4CAF50' if percentage >= 80 else '#FFC107'
        card_html += f"""
        <div style="
            margin-top: 0.5rem;
            font-size: 0.9rem;
            color: {color};
            font-weight: bold;
        ">
            {percentage:.0f}% Match
        </div>
        """
    
    card_html += "</div>"
    
    st.markdown(card_html, unsafe_allow_html=True)


def render_movie_grid(movies_data: List[Dict], columns: int = 3) -> None:
    """
    Render a grid of compact movie cards.
    
    Args:
        movies_data: List of movie dictionaries with keys:
            - title, genres, avg_rating, predicted_rating, match_score, rank
        columns: Number of columns (default 3)
    """
    # Create columns
    cols = st.columns(columns)
    
    for idx, movie in enumerate(movies_data):
        with cols[idx % columns]:
            render_movie_card_enhanced(
                title=movie['title'],
                genres=movie.get('genres', []),
                avg_rating=movie.get('avg_rating'),
                predicted_rating=movie.get('predicted_rating'),
                match_score=movie.get('match_score'),
                rank=movie.get('rank'),
                compact=True
            )
