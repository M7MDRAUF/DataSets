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
    if pd.isna(genres) or genres == "(no genres listed)":
        return "Unknown"
    
    genre_list = genres.split('|')
    
    if len(genre_list) <= max_genres:
        return ", ".join(genre_list)
    else:
        displayed = ", ".join(genre_list[:max_genres])
        remaining = len(genre_list) - max_genres
        return f"{displayed} (+{remaining} more)"


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
