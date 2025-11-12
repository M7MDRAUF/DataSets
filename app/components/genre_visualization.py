"""
CineMatch V2.1 - Genre Visualization Component

Charts and visualizations for genre analysis from local dataset.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import get_genre_emoji, create_genre_color_map


def render_genre_distribution(
    genres_list: List[str],
    title: str = "Genre Distribution",
    show_count: bool = True
) -> None:
    """
    Render genre distribution visualization using HTML/CSS (no external charts).
    
    Args:
        genres_list: List of all genres (can have duplicates)
        title: Chart title
        show_count: Show count numbers
    """
    # Count genres
    genre_counts = Counter(genres_list)
    total = sum(genre_counts.values())
    
    # Sort by count
    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Get colors
    colors = create_genre_color_map()
    
    # Build visualization
    st.markdown(f"### {title}")
    
    for genre, count in sorted_genres:
        percentage = (count / total) * 100
        color = colors.get(genre, '#666666')
        emoji = get_genre_emoji(genre)
        
        # Bar with gradient
        st.markdown(
            f"""
            <div style="margin: 0.75rem 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                    <span style="color: #CCC; font-weight: 500;">
                        {emoji} {genre}
                    </span>
                    <span style="color: {color}; font-weight: bold;">
                        {count if show_count else ''} ({percentage:.1f}%)
                    </span>
                </div>
                <div style="
                    width: 100%;
                    height: 24px;
                    background: #333;
                    border-radius: 12px;
                    overflow: hidden;
                    position: relative;
                ">
                    <div style="
                        width: {percentage}%;
                        height: 100%;
                        background: linear-gradient(90deg, {color} 0%, {color}AA 100%);
                        transition: width 0.5s ease;
                        display: flex;
                        align-items: center;
                        padding-left: 0.5rem;
                    ">
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_genre_badges_cloud(genres: List[str]) -> None:
    """
    Render a tag cloud of genre badges.
    
    Args:
        genres: List of unique genres
    """
    colors = create_genre_color_map()
    
    # Create flexible grid of badges
    badges_html = '<div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0;">'
    
    for genre in genres:
        emoji = get_genre_emoji(genre)
        color = colors.get(genre, '#666666')
        
        badges_html += f"""
        <span style="
            background: {color};
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 1rem;
            font-weight: 500;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            transition: transform 0.2s ease;
            cursor: pointer;
        " onmouseover="this.style.transform='scale(1.1)'" onmouseout="this.style.transform='scale(1)'">
            {emoji} {genre}
        </span>
        """
    
    badges_html += '</div>'
    
    st.markdown(badges_html, unsafe_allow_html=True)


def render_top_genres_summary(
    genres_list: List[str],
    top_n: int = 5
) -> None:
    """
    Render a summary card of top N genres.
    
    Args:
        genres_list: List of all genres
        top_n: Number of top genres to show
    """
    # Count and sort
    genre_counts = Counter(genres_list)
    top_genres = genre_counts.most_common(top_n)
    
    colors = create_genre_color_map()
    
    # Build summary card
    card_html = """
    <div style="
        background: linear-gradient(135deg, #222222 0%, #333333 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    ">
        <h3 style="color: white; margin-top: 0;">🏆 Top Genres</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
    """
    
    for idx, (genre, count) in enumerate(top_genres, 1):
        emoji = get_genre_emoji(genre)
        color = colors.get(genre, '#666666')
        
        # Medal emoji for top 3
        medal = '🥇' if idx == 1 else '🥈' if idx == 2 else '🥉' if idx == 3 else f'#{idx}'
        
        card_html += f"""
        <div style="
            background: {color}22;
            border-left: 4px solid {color};
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        ">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">
                {medal}
            </div>
            <div style="font-size: 1.2rem; color: white; font-weight: 600; margin-bottom: 0.25rem;">
                {emoji} {genre}
            </div>
            <div style="font-size: 1.5rem; color: {color}; font-weight: bold;">
                {count}
            </div>
        </div>
        """
    
    card_html += """
        </div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)


def analyze_genre_diversity(genres_lists: List[List[str]]) -> Dict:
    """
    Analyze genre diversity across multiple items (e.g., recommendations).
    
    Args:
        genres_lists: List of genre lists (e.g., [[Action, Comedy], [Drama], ...])
        
    Returns:
        Dictionary with diversity metrics
    """
    # Flatten all genres
    all_genres = [genre for genres in genres_lists for genre in genres]
    unique_genres = set(all_genres)
    
    # Calculate metrics
    total_items = len(genres_lists)
    total_genre_mentions = len(all_genres)
    unique_count = len(unique_genres)
    
    # Diversity score (0-1, higher is more diverse)
    diversity_score = unique_count / total_genre_mentions if total_genre_mentions > 0 else 0
    
    # Average genres per item
    avg_genres_per_item = total_genre_mentions / total_items if total_items > 0 else 0
    
    return {
        'unique_genres': unique_count,
        'total_genre_mentions': total_genre_mentions,
        'diversity_score': diversity_score,
        'avg_genres_per_item': avg_genres_per_item,
        'genre_list': list(unique_genres)
    }


def render_genre_diversity_metrics(diversity_data: Dict) -> None:
    """
    Render genre diversity metrics.
    
    Args:
        diversity_data: Output from analyze_genre_diversity()
    """
    # Diversity score visualization
    score = diversity_data['diversity_score'] * 100
    color = '#4CAF50' if score >= 70 else '#FFC107' if score >= 40 else '#F44336'
    
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #222222 0%, #333333 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
        ">
            <h4 style="color: white; margin-top: 0;">🎭 Genre Diversity Analysis</h4>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; color: {color}; font-weight: bold;">
                        {score:.0f}%
                    </div>
                    <div style="color: #999; font-size: 0.9rem;">Diversity Score</div>
                </div>
                
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; color: #E50914; font-weight: bold;">
                        {diversity_data['unique_genres']}
                    </div>
                    <div style="color: #999; font-size: 0.9rem;">Unique Genres</div>
                </div>
                
                <div style="text-align: center;">
                    <div style="font-size: 2.5rem; color: #2196F3; font-weight: bold;">
                        {diversity_data['avg_genres_per_item']:.1f}
                    </div>
                    <div style="color: #999; font-size: 0.9rem;">Avg. Genres/Item</div>
                </div>
            </div>
            
            <div style="margin-top: 1.5rem;">
                <div style="color: #CCC; margin-bottom: 0.5rem;">Diversity Level:</div>
                <div style="
                    width: 100%;
                    height: 12px;
                    background: #333;
                    border-radius: 6px;
                    overflow: hidden;
                ">
                    <div style="
                        width: {score}%;
                        height: 100%;
                        background: {color};
                        transition: width 0.5s ease;
                    "></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
