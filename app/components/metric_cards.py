"""
CineMatch V2.1 - Metric Cards Component

Beautiful metric displays with optional deltas and icons.
"""

import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards
from typing import Optional, Union


def render_metric_card(
    label: str,
    value: Union[str, int, float],
    delta: Optional[Union[str, int, float]] = None,
    delta_color: str = "normal",
    icon: str = "📊",
    help_text: Optional[str] = None
) -> None:
    """
    Render an enhanced metric card with icon and styling.
    
    Args:
        label: Metric label
        value: Metric value (will be formatted)
        delta: Change/delta value (optional)
        delta_color: 'normal', 'inverse', or 'off' for delta color behavior
        icon: Emoji icon for the metric
        help_text: Tooltip help text
    """
    # Format value if numeric
    if isinstance(value, float):
        if value >= 1000:
            formatted_value = f"{value:,.0f}"
        else:
            formatted_value = f"{value:.2f}"
    elif isinstance(value, int):
        formatted_value = f"{value:,}"
    else:
        formatted_value = str(value)
    
    # Format delta if provided
    formatted_delta = None
    if delta is not None:
        if isinstance(delta, (int, float)):
            sign = "+" if delta > 0 else ""
            formatted_delta = f"{sign}{delta:.2f}" if isinstance(delta, float) else f"{sign}{delta}"
        else:
            formatted_delta = str(delta)
    
    # Render metric with icon in label
    st.metric(
        label=f"{icon} {label}",
        value=formatted_value,
        delta=formatted_delta,
        delta_color=delta_color,
        help=help_text
    )


def render_metric_grid(metrics: list, columns: int = 4) -> None:
    """
    Render multiple metrics in a grid layout.
    
    Args:
        metrics: List of metric dictionaries with keys:
            - label, value, delta (optional), icon (optional), help_text (optional)
        columns: Number of columns in grid
    """
    # Create columns
    cols = st.columns(columns)
    
    # Render each metric
    for idx, metric in enumerate(metrics):
        with cols[idx % columns]:
            render_metric_card(
                label=metric['label'],
                value=metric['value'],
                delta=metric.get('delta'),
                delta_color=metric.get('delta_color', 'normal'),
                icon=metric.get('icon', '📊'),
                help_text=metric.get('help_text')
            )
    
    # Apply custom styling from streamlit-extras
    style_metric_cards(
        background_color="#222222",
        border_left_color="#E50914",
        border_color="#333333",
        box_shadow=True
    )


def render_algorithm_metrics(
    rmse: float,
    coverage: float,
    training_time: float,
    memory_mb: float
) -> None:
    """
    Render standard algorithm performance metrics.
    
    Args:
        rmse: Root Mean Square Error
        coverage: Coverage percentage (0-100)
        training_time: Training time in seconds
        memory_mb: Memory usage in MB
    """
    metrics = [
        {
            'label': 'RMSE',
            'value': rmse,
            'icon': '🎯',
            'help_text': 'Root Mean Square Error - lower is better'
        },
        {
            'label': 'Coverage',
            'value': f"{coverage:.1f}%",
            'icon': '📊',
            'help_text': 'Percentage of items that can be recommended'
        },
        {
            'label': 'Training Time',
            'value': f"{training_time:.2f}s",
            'icon': '⏱️',
            'help_text': 'Time taken to train the model'
        },
        {
            'label': 'Memory Usage',
            'value': f"{memory_mb:.1f} MB",
            'icon': '💾',
            'help_text': 'Memory consumed by the model'
        }
    ]
    
    render_metric_grid(metrics, columns=4)


def render_recommendation_stats(
    total_recommendations: int,
    avg_rating: float,
    unique_genres: int,
    generation_time: float
) -> None:
    """
    Render recommendation statistics.
    
    Args:
        total_recommendations: Number of recommendations generated
        avg_rating: Average predicted rating
        unique_genres: Number of unique genres in recommendations
        generation_time: Time taken to generate recommendations
    """
    metrics = [
        {
            'label': 'Recommendations',
            'value': total_recommendations,
            'icon': '🎬',
            'help_text': 'Total movies recommended'
        },
        {
            'label': 'Avg. Rating',
            'value': avg_rating,
            'icon': '⭐',
            'help_text': 'Average predicted rating'
        },
        {
            'label': 'Genre Diversity',
            'value': unique_genres,
            'icon': '🎭',
            'help_text': 'Number of different genres'
        },
        {
            'label': 'Generation Time',
            'value': f"{generation_time:.2f}s",
            'icon': '⚡',
            'help_text': 'Time taken to generate results'
        }
    ]
    
    render_metric_grid(metrics, columns=4)


def render_dataset_stats(
    total_movies: int,
    total_ratings: int,
    total_users: int,
    sparsity: float
) -> None:
    """
    Render dataset statistics.
    
    Args:
        total_movies: Total number of movies
        total_ratings: Total number of ratings
        total_users: Total number of users
        sparsity: Dataset sparsity (0-100)
    """
    metrics = [
        {
            'label': 'Movies',
            'value': total_movies,
            'icon': '🎬',
            'help_text': 'Total movies in dataset'
        },
        {
            'label': 'Ratings',
            'value': total_ratings,
            'icon': '⭐',
            'help_text': 'Total ratings in dataset'
        },
        {
            'label': 'Users',
            'value': total_users,
            'icon': '👥',
            'help_text': 'Total users who rated movies'
        },
        {
            'label': 'Sparsity',
            'value': f"{sparsity:.2f}%",
            'icon': '📉',
            'help_text': 'Percentage of missing ratings'
        }
    ]
    
    render_metric_grid(metrics, columns=4)
