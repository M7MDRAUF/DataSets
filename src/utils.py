"""
CineMatch V1.0.0 - Utility Functions

Explainability features (XAI) and helper functions.
Implements F-05 (Explainable AI) and F-06 (User Taste Profile).

Author: CineMatch Team
Date: October 24, 2025
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

# Import data loading
sys.path.append(str(Path(__file__).parent.parent))


def explain_recommendation(
    user_id: int,
    movie_id: int,
    movie_title: str,
    movie_genres: List[str],
    user_history: pd.DataFrame,
    ratings_df: pd.DataFrame,
    n_similar: int = 3
) -> str:
    """
    Generate an explanation for why a movie is recommended.
    
    Implements F-05: Explainable AI (XAI)
    
    Uses multiple strategies:
    1. Content-based: Similar genres to highly-rated movies
    2. Collaborative: Users with similar taste loved this
    3. Genre preference: Matches user's top genres
    4. Fallback: High global rating
    
    Args:
        user_id: User ID
        movie_id: Recommended movie ID
        movie_title: Movie title
        movie_genres: List of movie genres
        user_history: User's rating history DataFrame
        ratings_df: Full ratings DataFrame
        n_similar: Number of similar movies to mention
    
    Returns:
        Explanation string
    """
    explanations = []
    
    # Strategy 1: Content-Based Similarity (Genre Match)
    # Find user's top-rated movies in the same genres
    matching_rated = user_history[
        user_history['genres_list'].apply(
            lambda x: any(genre in movie_genres for genre in x)
        )
    ].sort_values('rating', ascending=False).head(n_similar)
    
    if len(matching_rated) > 0 and matching_rated['rating'].iloc[0] >= 4.0:
        similar_titles = matching_rated['title'].head(3).tolist()
        if len(similar_titles) == 1:
            explanations.append(f"Because you highly rated **{similar_titles[0]}**, which shares similar themes")
        elif len(similar_titles) == 2:
            explanations.append(f"Because you loved **{similar_titles[0]}** and **{similar_titles[1]}**")
        else:
            explanations.append(f"Because you enjoyed **{similar_titles[0]}**, **{similar_titles[1]}**, and **{similar_titles[2]}**")
    
    # Strategy 2: Collaborative Filtering
    # Find users who rated this movie highly and see if they share taste with current user
    movie_ratings = ratings_df[ratings_df['movieId'] == movie_id]
    high_raters = movie_ratings[movie_ratings['rating'] >= 4.5]['userId'].unique()
    
    if len(high_raters) > 10:  # Significant number of high raters
        # Check if any of these users also highly rated movies the current user likes
        user_liked = user_history[user_history['rating'] >= 4.0]['movieId'].values
        
        similar_users_count = 0
        for other_user in high_raters[:50]:  # Sample to avoid performance issues
            other_user_ratings = ratings_df[
                (ratings_df['userId'] == other_user) &
                (ratings_df['movieId'].isin(user_liked))
            ]
            if len(other_user_ratings) >= 3:  # At least 3 movies in common
                similar_users_count += 1
        
        if similar_users_count >= 10:
            explanations.append(f"Users with similar taste to yours loved this movie")
    
    # Strategy 3: Genre Preference
    # Check if movie matches user's top genres
    user_genre_counts = user_history.explode('genres_list')['genres_list'].value_counts()
    top_genres = user_genre_counts.head(3).index.tolist()
    
    matching_top_genres = [g for g in movie_genres if g in top_genres]
    if matching_top_genres:
        genre_str = ", ".join(matching_top_genres[:2])
        explanations.append(f"Matches your interest in {genre_str}")
    
    # Strategy 4: High Global Rating (Fallback)
    if len(explanations) == 0:
        avg_rating = movie_ratings['rating'].mean()
        if avg_rating >= 4.0:
            explanations.append(f"This critically acclaimed film has an average rating of {avg_rating:.1f}/5.0")
        else:
            explanations.append("Based on your unique taste profile, we think you'll enjoy this hidden gem")
    
    # Combine explanations
    if len(explanations) == 1:
        return explanations[0]
    elif len(explanations) == 2:
        return f"{explanations[0]}, and {explanations[1].lower()}"
    else:
        return f"{explanations[0]}, {explanations[1].lower()}, and {explanations[2].lower()}"


def get_user_taste_profile(user_history: pd.DataFrame) -> Dict:
    """
    Generate a user's taste profile summary.
    
    Implements F-06: User Taste Profile
    
    Args:
        user_history: User's rating history DataFrame
    
    Returns:
        Dictionary with taste profile information:
        - top_genres: List of (genre, percentage) tuples
        - avg_rating: User's average rating
        - num_ratings: Total number of ratings
        - top_rated_movies: List of user's highest-rated movies
        - rating_distribution: Count of ratings by value
    """
    if len(user_history) == 0:
        return {
            'top_genres': [],
            'avg_rating': 0.0,
            'num_ratings': 0,
            'top_rated_movies': [],
            'rating_distribution': {}
        }
    
    # Genre preferences
    genre_counts = user_history.explode('genres_list')['genres_list'].value_counts()
    total_genre_mentions = genre_counts.sum()
    
    top_genres = [
        {
            'genre': genre,
            'count': int(count),
            'percentage': round((count / total_genre_mentions) * 100, 1)
        }
        for genre, count in genre_counts.head(5).items()
    ]
    
    # Rating statistics
    avg_rating = round(user_history['rating'].mean(), 2)
    num_ratings = len(user_history)
    
    # Top-rated movies
    top_rated = user_history.nlargest(5, 'rating')[['title', 'rating', 'genres']].to_dict('records')
    
    # Rating distribution
    rating_dist = user_history['rating'].value_counts().sort_index(ascending=False).to_dict()
    
    return {
        'top_genres': top_genres,
        'avg_rating': avg_rating,
        'num_ratings': num_ratings,
        'top_rated_movies': top_rated,
        'rating_distribution': rating_dist
    }


def format_genres(genres: str, max_genres: int = 3) -> str:
    """
    Format genre string for display.
    
    Args:
        genres: Pipe-separated genre string (e.g., "Action|Sci-Fi|Thriller")
        max_genres: Maximum number of genres to display
    
    Returns:
        Formatted genre string (e.g., "Action, Sci-Fi, Thriller")
    """
    # Defensive: handle None, empty, or invalid input
    if not genres or pd.isna(genres) or genres == "(no genres listed)" or str(genres).strip() == "":
        return "Unknown"
    
    try:
        genre_list = str(genres).split('|')
        # Filter empty strings
        genre_list = [g.strip() for g in genre_list if g.strip()]
        
        if not genre_list:
            return "Unknown"
        
        if len(genre_list) <= max_genres:
            return ", ".join(genre_list)
        else:
            displayed = ", ".join(genre_list[:max_genres])
            remaining = len(genre_list) - max_genres
            return f"{displayed} (+{remaining} more)"
    except Exception:
        return "Unknown"


def format_rating(rating: float, style: str = "stars") -> str:
    """
    Format a rating for display.
    
    Args:
        rating: Numeric rating (0.5 to 5.0)
        style: Display style - "stars" or "number"
    
    Returns:
        Formatted rating string
    """
    if style == "stars":
        full_stars = int(rating)
        half_star = 1 if (rating - full_stars) >= 0.5 else 0
        empty_stars = 5 - full_stars - half_star
        
        stars = "⭐" * full_stars
        if half_star:
            stars += "✨"
        
        return f"{stars} ({rating:.1f}/5.0)"
    else:
        return f"{rating:.1f}/5.0"


def calculate_diversity_score(recommendations: pd.DataFrame) -> float:
    """
    Calculate diversity score for a recommendation list.
    
    A higher score means more genre diversity in recommendations.
    
    Args:
        recommendations: DataFrame with 'genres_list' column
    
    Returns:
        Diversity score (0.0 to 1.0)
    """
    if len(recommendations) == 0:
        return 0.0
    
    # Get all unique genres in recommendations
    all_genres = set()
    for genres in recommendations['genres_list']:
        all_genres.update(genres)
    
    # Diversity = unique genres / total genre mentions
    total_mentions = sum(len(genres) for genres in recommendations['genres_list'])
    diversity = len(all_genres) / total_mentions if total_mentions > 0 else 0.0
    
    return round(diversity, 2)


def get_recommendation_stats(recommendations: pd.DataFrame) -> Dict:
    """
    Calculate statistics about a set of recommendations.
    
    Args:
        recommendations: DataFrame with recommendations
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        'count': len(recommendations),
        'avg_predicted_rating': round(recommendations['predicted_rating'].mean(), 2),
        'min_predicted_rating': round(recommendations['predicted_rating'].min(), 2),
        'max_predicted_rating': round(recommendations['predicted_rating'].max(), 2),
        'diversity_score': calculate_diversity_score(recommendations)
    }
    
    # Genre distribution
    genre_counts = recommendations.explode('genres_list')['genres_list'].value_counts().head(5)
    stats['top_genres'] = genre_counts.to_dict()
    
    return stats


def extract_year_from_title(title: str) -> Optional[int]:
    """
    Extract release year from movie title.
    
    MovieLens titles typically end with (YEAR)
    
    Args:
        title: Movie title string
    
    Returns:
        Year as integer, or None if not found
    """
    import re
    match = re.search(r'\((\d{4})\)$', title)
    if match:
        return int(match.group(1))
    return None


def format_movie_title(title: str, include_year: bool = True) -> str:
    """
    Format movie title for display.
    
    Args:
        title: Raw movie title
        include_year: Whether to include year in output
    
    Returns:
        Formatted title
    """
    if not include_year:
        # Remove year from title
        import re
        title = re.sub(r'\s*\(\d{4}\)$', '', title)
    
    return title


def create_genre_color_map() -> Dict[str, str]:
    """
    Create a color map for movie genres (for visualizations).
    
    Returns:
        Dictionary mapping genre names to color codes
    """
    return {
        'Action': '#FF6B6B',
        'Adventure': '#4ECDC4',
        'Animation': '#FFE66D',
        'Children': '#95E1D3',
        'Comedy': '#F38181',
        'Crime': '#AA96DA',
        'Documentary': '#FCBAD3',
        'Drama': '#A8D8EA',
        'Fantasy': '#FFAAA5',
        'Film-Noir': '#7F8C8D',
        'Horror': '#C0392B',
        'IMAX': '#34495E',
        'Musical': '#E8DAEF',
        'Mystery': '#5DADE2',
        'Romance': '#F1948A',
        'Sci-Fi': '#85C1E2',
        'Thriller': '#641E16',
        'War': '#566573',
        'Western': '#D4AC0D'
    }


def get_genre_emoji(genre: str) -> str:
    """
    Get emoji representation for a genre.
    
    Args:
        genre: Genre name
    
    Returns:
        Emoji string
    """
    emoji_map = {
        'Action': '💥',
        'Adventure': '🗺️',
        'Animation': '🎨',
        'Children': '👶',
        'Comedy': '😂',
        'Crime': '🔫',
        'Documentary': '📽️',
        'Drama': '🎭',
        'Fantasy': '🧙',
        'Film-Noir': '🎩',
        'Horror': '👻',
        'IMAX': '🎬',
        'Musical': '🎵',
        'Mystery': '🔍',
        'Romance': '❤️',
        'Sci-Fi': '🚀',
        'Thriller': '😱',
        'War': '⚔️',
        'Western': '🤠'
    }
    
    return emoji_map.get(genre, '🎬')


# Streamlit-specific helpers

def create_rating_stars(rating: float) -> str:
    """
    Create star rating HTML for Streamlit.
    
    Args:
        rating: Rating value (0.5 to 5.0)
    
    Returns:
        HTML string with star rating
    """
    # Defensive: handle invalid ratings
    try:
        rating = float(rating)
        # Clamp to valid range
        rating = max(0.0, min(5.0, rating))
    except (ValueError, TypeError):
        rating = 0.0
    
    full_stars = int(rating)
    half_star = 1 if (rating - full_stars) >= 0.5 else 0
    empty_stars = 5 - full_stars - half_star
    
    html = '<span style="color: gold; font-size: 20px;">'
    html += '⭐' * full_stars
    if half_star:
        html += '✨'
    html += '☆' * empty_stars
    html += f'</span> <span style="font-size: 16px; color: #666;">({rating:.1f})</span>'
    
    return html


# ============================================================================
# SEARCH PAGE UTILITY FUNCTIONS (Added V2.1.2 - November 20, 2025)
# ============================================================================

def format_rating_history(
    user_ratings: pd.DataFrame,
    max_display: int = 50
) -> pd.DataFrame:
    """
    Format user rating history for display in Search page.
    
    Args:
        user_ratings: DataFrame from search_engine.get_user_ratings()
        max_display: Maximum number of rows to return
        
    Returns:
        Formatted DataFrame with display-friendly columns
    """
    if user_ratings.empty:
        return pd.DataFrame()
    
    display_df = user_ratings.head(max_display).copy()
    
    # Format timestamp
    display_df['Date'] = display_df['timestamp'].dt.strftime('%Y-%m-%d')
    
    # Format genres
    display_df['Genres'] = display_df['genres'].apply(
        lambda x: format_genres(x, max_genres=3)
    )
    
    # Select and rename columns
    display_df = display_df[['title', 'Genres', 'rating', 'Date']]
    display_df.columns = ['Movie Title', 'Genres', 'Rating', 'Rated On']
    
    return display_df


def calculate_user_engagement_score(stats: Dict) -> float:
    """
    Calculate an engagement score (0-100) based on user statistics.
    
    Args:
        stats: Dictionary from search_engine.get_user_statistics()
        
    Returns:
        Engagement score (0-100)
    """
    if stats['total_ratings'] == 0:
        return 0.0
    
    # Factors:
    # 1. Volume: Number of ratings (0-40 points, log scale)
    # 2. Diversity: Genre variety (0-30 points)
    # 3. Consistency: Rating std deviation (0-30 points, lower is more consistent)
    
    # Volume score (log scale, capped at 1000 ratings)
    volume_score = min(40, (np.log10(stats['total_ratings'] + 1) / np.log10(1000)) * 40)
    
    # Diversity score
    genre_diversity = len(stats['top_genres'])
    diversity_score = min(30, (genre_diversity / 20) * 30)  # Max at 20 unique genres
    
    # Consistency score (inverse of std dev)
    consistency_score = max(0, 30 - (stats['std_rating'] * 10))
    
    total_score = volume_score + diversity_score + consistency_score
    return min(100, max(0, total_score))


def create_user_profile_summary(stats: Dict, user_id: int) -> str:
    """
    Generate a natural language summary of user profile.
    
    Args:
        stats: Dictionary from search_engine.get_user_statistics()
        user_id: User identifier
        
    Returns:
        HTML-formatted summary string
    """
    if stats['total_ratings'] == 0:
        return f"<p>User {user_id} has not rated any movies yet.</p>"
    
    engagement = calculate_user_engagement_score(stats)
    
    # Determine user type
    if stats['total_ratings'] < 20:
        user_type = "Casual Viewer"
        type_color = "#FFA500"
    elif stats['total_ratings'] < 100:
        user_type = "Regular User"
        type_color = "#4169E1"
    elif stats['total_ratings'] < 500:
        user_type = "Active Cinephile"
        type_color = "#32CD32"
    else:
        user_type = "Power User"
        type_color = "#E50914"
    
    # Determine rating tendency
    if stats['avg_rating'] >= 4.0:
        tendency = "generous rater"
    elif stats['avg_rating'] >= 3.0:
        tendency = "balanced critic"
    else:
        tendency = "selective viewer"
    
    # Get top genre
    top_genre = stats['top_genres'][0][0] if stats['top_genres'] else "Various"
    top_genre_emoji = get_genre_emoji(top_genre)
    
    summary = f"""
    <div style="background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%); 
                padding: 1.5rem; border-radius: 10px; border-left: 4px solid {type_color};">
        <h3 style="color: {type_color};">👤 {user_type}</h3>
        <p><strong>Engagement Score:</strong> {engagement:.0f}/100</p>
        <p><strong>Rating Style:</strong> {tendency.title()} ({stats['avg_rating']:.2f}⭐ average)</p>
        <p><strong>Favorite Genre:</strong> {top_genre_emoji} {top_genre}</p>
        <p><strong>Activity Period:</strong> {(stats['last_rating_date'] - stats['first_rating_date']).days if stats['last_rating_date'] and stats['first_rating_date'] else 0} days</p>
    </div>
    """
    
    return summary


# =============================================================================
# TMDB Image Utilities
# =============================================================================

# TMDB Image Base URL (w500 is optimal for card displays)
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
TMDB_BACKDROP_BASE_URL = "https://image.tmdb.org/t/p/w1280"

# Placeholder image for movies without posters
PLACEHOLDER_POSTER = "https://via.placeholder.com/200x300/1a1a2e/ffffff?text=No+Poster"
PLACEHOLDER_BACKDROP = "https://via.placeholder.com/1280x720/1a1a2e/ffffff?text=No+Backdrop"


def get_tmdb_poster_url(poster_path: Optional[str], size: str = "w500") -> str:
    """
    Generate full TMDB poster URL from path.
    
    Args:
        poster_path: TMDB poster path (e.g., '/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg')
        size: Image size (w92, w154, w185, w342, w500, w780, original)
    
    Returns:
        Full URL to poster image or placeholder
    """
    if not poster_path or pd.isna(poster_path) or str(poster_path).strip() == '':
        return PLACEHOLDER_POSTER
    
    # Ensure path starts with /
    path = str(poster_path).strip()
    if not path.startswith('/'):
        path = '/' + path
    
    return f"https://image.tmdb.org/t/p/{size}{path}"


def get_tmdb_backdrop_url(backdrop_path: Optional[str], size: str = "w1280") -> str:
    """
    Generate full TMDB backdrop URL from path.
    
    Args:
        backdrop_path: TMDB backdrop path (e.g., '/3Rfvhy1Nl6sSGJwyjb0QiZzZYlB.jpg')
        size: Image size (w300, w780, w1280, original)
    
    Returns:
        Full URL to backdrop image or placeholder
    """
    if not backdrop_path or pd.isna(backdrop_path) or str(backdrop_path).strip() == '':
        return PLACEHOLDER_BACKDROP
    
    # Ensure path starts with /
    path = str(backdrop_path).strip()
    if not path.startswith('/'):
        path = '/' + path
    
    return f"https://image.tmdb.org/t/p/{size}{path}"


def create_movie_card_html(
    title: str,
    genres: str,
    rating: float = None,
    poster_path: str = None,
    movie_id: int = None,
    explanation: str = None,
    predicted_rating: float = None,
    show_poster: bool = True
) -> str:
    """
    Create a styled movie card HTML with poster image.
    
    Args:
        title: Movie title
        genres: Pipe-separated genre string
        rating: User's rating or average rating
        poster_path: TMDB poster path
        movie_id: Movie ID for linking
        explanation: XAI explanation text
        predicted_rating: Predicted rating from algorithm
        show_poster: Whether to show poster image
    
    Returns:
        HTML string for the movie card
    """
    poster_url = get_tmdb_poster_url(poster_path) if show_poster else None
    formatted_genres = format_genres(genres, max_genres=3)
    year = extract_year_from_title(title)
    
    # Get genre emoji for the first genre
    first_genre = genres.split('|')[0] if genres and '|' in genres else (genres or 'Drama')
    genre_emoji = get_genre_emoji(first_genre)
    
    # Rating display
    rating_html = ""
    if rating is not None:
        rating_html = f"""
        <div style="display: flex; align-items: center; gap: 5px; margin-top: 8px;">
            <span style="color: #FFD700; font-size: 1.1em;">{'⭐' * int(round(rating))}</span>
            <span style="color: #888;">({rating:.1f})</span>
        </div>
        """
    
    if predicted_rating is not None:
        rating_html += f"""
        <div style="color: #4CAF50; font-size: 0.9em; margin-top: 4px;">
            📊 Predicted: {predicted_rating:.2f}
        </div>
        """
    
    # Explanation display
    explanation_html = ""
    if explanation:
        explanation_html = f"""
        <div style="background: rgba(229, 9, 20, 0.1); padding: 10px; border-radius: 5px; 
                    margin-top: 10px; border-left: 3px solid #E50914; font-size: 0.85em;">
            💡 {explanation}
        </div>
        """
    
    # Poster HTML
    poster_html = ""
    if show_poster and poster_url:
        poster_html = f"""
        <div style="flex-shrink: 0; width: 120px;">
            <img src="{poster_url}" alt="{title}" 
                 style="width: 120px; height: 180px; object-fit: cover; border-radius: 8px; 
                        box-shadow: 0 4px 8px rgba(0,0,0,0.3);"
                 onerror="this.src='{PLACEHOLDER_POSTER}'">
        </div>
        """
    
    card_html = f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                padding: 15px; border-radius: 12px; margin-bottom: 15px;
                border: 1px solid rgba(255,255,255,0.1);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                display: flex; gap: 15px; align-items: flex-start;">
        {poster_html}
        <div style="flex-grow: 1;">
            <h4 style="margin: 0 0 8px 0; color: #fff; font-size: 1.1em;">
                🎬 {title}
            </h4>
            <div style="color: #aaa; font-size: 0.9em; margin-bottom: 5px;">
                {genre_emoji} {formatted_genres}
                {f' • 📅 {year}' if year else ''}
            </div>
            {rating_html}
            {explanation_html}
        </div>
    </div>
    """
    
    return card_html


def create_compact_movie_card_html(
    title: str,
    genres: str,
    rating: float = None,
    poster_path: str = None
) -> str:
    """
    Create a compact movie card for grid layouts.
    
    Args:
        title: Movie title
        genres: Pipe-separated genre string
        rating: Rating value
        poster_path: TMDB poster path
    
    Returns:
        HTML string for compact movie card
    """
    poster_url = get_tmdb_poster_url(poster_path)
    formatted_genres = format_genres(genres, max_genres=2)
    
    # Truncate title if too long
    display_title = title if len(title) <= 30 else title[:27] + "..."
    
    rating_html = f"""
        <div style="color: #FFD700; font-size: 0.85em; margin-top: 5px;">
            ⭐ {rating:.1f}
        </div>
    """ if rating else ""
    
    card_html = f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                border-radius: 10px; overflow: hidden; text-align: center;
                border: 1px solid rgba(255,255,255,0.1);
                transition: transform 0.3s ease;">
        <img src="{poster_url}" alt="{title}" 
             style="width: 100%; height: 200px; object-fit: cover;"
             onerror="this.src='{PLACEHOLDER_POSTER}'">
        <div style="padding: 10px;">
            <div style="color: #fff; font-size: 0.9em; font-weight: 500; 
                        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                {display_title}
            </div>
            <div style="color: #888; font-size: 0.75em; margin-top: 3px;">
                {formatted_genres}
            </div>
            {rating_html}
        </div>
    </div>
    """
    
    return card_html


if __name__ == "__main__":
    """
    Test utility functions.
    """
    print("CineMatch V1.0.0 - Utility Functions Test\n")
    print("=" * 70)
    
    # Test genre formatting
    test_genres = "Action|Adventure|Sci-Fi|Thriller|Mystery"
    print(f"Genre formatting test:")
    print(f"  Input: {test_genres}")
    print(f"  Output: {format_genres(test_genres, max_genres=3)}")
    
    # Test rating formatting
    test_rating = 4.7
    print(f"\nRating formatting test:")
    print(f"  Input: {test_rating}")
    print(f"  Stars: {format_rating(test_rating, style='stars')}")
    print(f"  Number: {format_rating(test_rating, style='number')}")
    
    # Test year extraction
    test_title = "The Matrix (1999)"
    year = extract_year_from_title(test_title)
    print(f"\nYear extraction test:")
    print(f"  Input: {test_title}")
    print(f"  Year: {year}")
    
    # Test genre emojis
    print(f"\nGenre emoji test:")
    for genre in ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror']:
        print(f"  {genre}: {get_genre_emoji(genre)}")
    
    print("\n✅ Utility functions test successful!")
