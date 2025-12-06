"""
CineMatch V2.1.6 - UI Helper Functions

Streamlit UI utilities for consistent user experience across all pages.
Implements Tasks 2.1-2.14 from the usability improvement plan.

Author: CineMatch Development Team
Date: December 5, 2025

Features:
    - Loading spinners with progress indicators (2.1)
    - User-friendly error messages (2.2)
    - Input validation with helpful feedback (2.3)
    - Contextual help tooltips (2.4)
    - Accessibility support (2.8)
    - Export functionality (2.12)
    - User preference memory (2.11)
"""

import streamlit as st
import pandas as pd
import json
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime
from io import StringIO, BytesIO

logger = logging.getLogger(__name__)

# =============================================================================
# TASK 2.1: Loading Spinners & Progress Indicators
# =============================================================================

def show_loading_spinner(
    message: str = "Loading...",
    emoji: str = "⏳"
) -> st.spinner:
    """
    Create a styled loading spinner.
    
    Args:
        message: Loading message to display
        emoji: Emoji prefix for the message
        
    Returns:
        Streamlit spinner context manager
    """
    return st.spinner(f"{emoji} {message}")


def show_progress_bar(
    total_steps: int,
    current_step: int,
    message: str = "Processing",
    step_descriptions: Optional[List[str]] = None
) -> None:
    """
    Display a progress bar with step information.
    
    Args:
        total_steps: Total number of steps
        current_step: Current step (1-indexed)
        message: Overall progress message
        step_descriptions: Optional list of descriptions for each step
    """
    progress = current_step / total_steps
    
    # Create progress bar
    progress_bar = st.progress(progress)
    
    # Show step info
    step_msg = f"Step {current_step}/{total_steps}"
    if step_descriptions and current_step <= len(step_descriptions):
        step_msg += f": {step_descriptions[current_step - 1]}"
    
    st.caption(f"{message} - {step_msg}")


def show_model_loading_status(
    algorithm_name: str,
    status: str = "loading",
    metrics: Optional[Dict[str, Any]] = None
) -> None:
    """
    Display model loading status with detailed information.
    
    Args:
        algorithm_name: Name of the algorithm being loaded
        status: Current status ('loading', 'training', 'ready', 'error')
        metrics: Optional metrics to display (training time, memory, etc.)
    """
    status_config = {
        'loading': {'emoji': '⏳', 'color': 'blue', 'text': 'Loading pre-trained model...'},
        'training': {'emoji': '🔧', 'color': 'orange', 'text': 'Training model...'},
        'ready': {'emoji': '✅', 'color': 'green', 'text': 'Model ready!'},
        'error': {'emoji': '❌', 'color': 'red', 'text': 'Failed to load model'},
        'cached': {'emoji': '⚡', 'color': 'green', 'text': 'Using cached model'},
    }
    
    config = status_config.get(status, status_config['loading'])
    
    # Status message
    st.markdown(f"""
    <div style="
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid {_get_status_border_color(config['color'])};
        background: {_get_status_bg_color(config['color'])};
        margin: 0.5rem 0;
    ">
        <strong>{config['emoji']} {algorithm_name}</strong><br>
        <span style="color: #666;">{config['text']}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Show metrics if available and model is ready
    if metrics and status in ('ready', 'cached'):
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'training_time' in metrics:
                st.metric("Training Time", f"{metrics['training_time']:.1f}s")
        with col2:
            if 'memory_mb' in metrics:
                st.metric("Memory", f"{metrics['memory_mb']:.1f}MB")
        with col3:
            if 'rmse' in metrics:
                st.metric("RMSE", f"{metrics['rmse']:.4f}")


def _get_status_border_color(color_name: str) -> str:
    """Get border color for status display."""
    colors = {
        'blue': '#4A90D9',
        'orange': '#F5A623',
        'green': '#28A745',
        'red': '#DC3545',
    }
    return colors.get(color_name, '#4A90D9')


def _get_status_bg_color(color_name: str) -> str:
    """Get background color for status display."""
    colors = {
        'blue': 'rgba(74, 144, 217, 0.1)',
        'orange': 'rgba(245, 166, 35, 0.1)',
        'green': 'rgba(40, 167, 69, 0.1)',
        'red': 'rgba(220, 53, 69, 0.1)',
    }
    return colors.get(color_name, 'rgba(74, 144, 217, 0.1)')


# =============================================================================
# TASK 2.2: User-Friendly Error Messages
# =============================================================================

# Error message mappings for common technical errors
ERROR_MESSAGES = {
    'FileNotFoundError': {
        'title': '📁 File Not Found',
        'message': 'The required data file could not be found.',
        'suggestions': [
            'Ensure the MovieLens dataset is in the `data/ml-32m/` directory',
            'Download the dataset from https://grouplens.org/datasets/movielens/',
            'Check file permissions'
        ]
    },
    'MemoryError': {
        'title': '💾 Insufficient Memory',
        'message': 'The system ran out of memory while processing.',
        'suggestions': [
            'Use a smaller dataset sample (Performance Settings)',
            'Close other applications',
            'Restart the application',
            'Increase Docker container memory limits'
        ]
    },
    'ValueError': {
        'title': '⚠️ Invalid Value',
        'message': 'An invalid value was provided.',
        'suggestions': [
            'Check the input format',
            'Ensure all required fields are filled',
            'Verify the value is within the expected range'
        ]
    },
    'KeyError': {
        'title': '🔑 Missing Data',
        'message': 'Required data field was not found.',
        'suggestions': [
            'Refresh the page and try again',
            'The data format may have changed',
            'Contact support if the issue persists'
        ]
    },
    'ConnectionError': {
        'title': '🌐 Connection Failed',
        'message': 'Could not connect to the required service.',
        'suggestions': [
            'Check your internet connection',
            'The server may be temporarily unavailable',
            'Try again in a few moments'
        ]
    },
    'TimeoutError': {
        'title': '⏰ Request Timed Out',
        'message': 'The operation took too long to complete.',
        'suggestions': [
            'Try a smaller dataset or fewer recommendations',
            'The system may be under heavy load',
            'Try again later'
        ]
    },
}


def show_friendly_error(
    exception: Exception,
    context: str = "",
    show_technical: bool = False
) -> None:
    """
    Display a user-friendly error message based on exception type.
    
    Args:
        exception: The caught exception
        context: Optional context about what was being done
        show_technical: Whether to show technical details (for debugging)
    """
    error_type = type(exception).__name__
    error_info = ERROR_MESSAGES.get(error_type, {
        'title': '❌ An Error Occurred',
        'message': 'Something went wrong.',
        'suggestions': [
            'Try refreshing the page',
            'Clear your browser cache',
            'Contact support if the issue persists'
        ]
    })
    
    # Main error display
    st.error(f"**{error_info['title']}**")
    
    if context:
        st.markdown(f"*While: {context}*")
    
    st.markdown(error_info['message'])
    
    # Suggestions
    with st.expander("💡 What you can try", expanded=True):
        for suggestion in error_info['suggestions']:
            st.markdown(f"• {suggestion}")
    
    # Technical details (hidden by default)
    if show_technical:
        with st.expander("🔧 Technical Details", expanded=False):
            st.code(f"{error_type}: {str(exception)}")
    
    logger.error(f"User-facing error: {error_type} - {str(exception)}", exc_info=True)


def show_validation_error(
    field_name: str,
    value: Any,
    expected: str,
    suggestion: Optional[str] = None
) -> None:
    """
    Display a validation error for a specific field.
    
    Args:
        field_name: Name of the field that failed validation
        value: The invalid value that was provided
        expected: Description of what was expected
        suggestion: Optional suggestion for fixing the error
    """
    st.error(f"❌ Invalid {field_name}")
    
    st.markdown(f"""
    **Provided value:** `{value}`  
    **Expected:** {expected}
    """)
    
    if suggestion:
        st.info(f"💡 **Suggestion:** {suggestion}")


def show_system_error(
    error_code: str,
    message: str,
    recovery_action: Optional[str] = None
) -> None:
    """
    Display a system error with error code for support reference.
    
    Args:
        error_code: CineMatch error code (e.g., CME-2001)
        message: User-friendly error message
        recovery_action: Suggested action to recover
    """
    st.error(f"**System Error** ({error_code})")
    st.markdown(message)
    
    if recovery_action:
        st.info(f"🔄 **Recovery:** {recovery_action}")
    
    st.caption(f"Error reference: {error_code} | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# =============================================================================
# TASK 2.3: Input Validation with UI Feedback
# =============================================================================

def validated_number_input(
    label: str,
    min_value: int,
    max_value: int,
    default_value: int,
    key: str,
    help_text: Optional[str] = None,
    show_range: bool = True
) -> Tuple[int, bool]:
    """
    Number input with built-in validation and feedback.
    
    Args:
        label: Input label
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        default_value: Default value
        key: Streamlit widget key
        help_text: Optional help text
        show_range: Whether to show valid range info
        
    Returns:
        Tuple of (value, is_valid)
    """
    if show_range:
        label_with_range = f"{label} ({min_value:,} - {max_value:,})"
    else:
        label_with_range = label
    
    value = st.number_input(
        label_with_range,
        min_value=min_value,
        max_value=max_value,
        value=default_value,
        step=1,
        key=key,
        help=help_text
    )
    
    is_valid = min_value <= value <= max_value
    
    if not is_valid:
        st.warning(f"⚠️ Value must be between {min_value:,} and {max_value:,}")
    
    return value, is_valid


def validated_text_input(
    label: str,
    key: str,
    max_length: int = 500,
    pattern: Optional[str] = None,
    help_text: Optional[str] = None,
    placeholder: Optional[str] = None
) -> Tuple[str, bool]:
    """
    Text input with validation and feedback.
    
    Args:
        label: Input label
        key: Streamlit widget key
        max_length: Maximum allowed length
        pattern: Optional regex pattern for validation
        help_text: Optional help text
        placeholder: Placeholder text
        
    Returns:
        Tuple of (sanitized_value, is_valid)
    """
    import re
    import html
    
    value = st.text_input(
        label,
        key=key,
        max_chars=max_length,
        help=help_text,
        placeholder=placeholder
    )
    
    # Sanitize
    sanitized = html.escape(value.strip()) if value else ""
    
    # Validate length
    if len(sanitized) > max_length:
        st.warning(f"⚠️ Maximum length is {max_length} characters")
        return sanitized[:max_length], False
    
    # Validate pattern if provided
    if pattern and sanitized:
        if not re.match(pattern, sanitized):
            st.warning("⚠️ Invalid format")
            return sanitized, False
    
    return sanitized, True


# =============================================================================
# TASK 2.4: Contextual Help Tooltips
# =============================================================================

ALGORITHM_HELP = {
    'SVD': {
        'name': 'Singular Value Decomposition (SVD)',
        'icon': '🔮',
        'simple_explanation': 'Finds hidden patterns in your ratings to predict what you\'ll like.',
        'best_for': 'Users with many ratings who want diverse recommendations',
        'strengths': ['Fast predictions', 'Good accuracy', 'Memory efficient'],
        'weaknesses': ['Needs many ratings', 'Cold start problem'],
        'technical': 'Matrix factorization technique that decomposes the user-item matrix.'
    },
    'User KNN': {
        'name': 'User-Based Collaborative Filtering',
        'icon': '👥',
        'simple_explanation': 'Finds users similar to you and recommends their favorites.',
        'best_for': 'Users who want recommendations based on similar people\'s tastes',
        'strengths': ['Intuitive explanations', 'Serendipitous discoveries'],
        'weaknesses': ['Can be slow with many users', 'Needs overlap in ratings'],
        'technical': 'K-Nearest Neighbors on user-user similarity matrix.'
    },
    'Item KNN': {
        'name': 'Item-Based Collaborative Filtering',
        'icon': '🎬',
        'simple_explanation': 'Recommends movies similar to ones you\'ve already liked.',
        'best_for': 'Users who want "more like this" recommendations',
        'strengths': ['Stable predictions', 'Fast for new users', 'Clear explanations'],
        'weaknesses': ['Less diverse', 'Popularity bias'],
        'technical': 'K-Nearest Neighbors on item-item similarity matrix.'
    },
    'Content-Based': {
        'name': 'Content-Based Filtering',
        'icon': '📝',
        'simple_explanation': 'Recommends movies based on genres and features you enjoy.',
        'best_for': 'New users or those with specific genre preferences',
        'strengths': ['No cold start', 'Transparent', 'Works with few ratings'],
        'weaknesses': ['Limited diversity', 'No serendipity'],
        'technical': 'TF-IDF vectorization of movie features with cosine similarity.'
    },
    'Hybrid': {
        'name': 'Hybrid Ensemble',
        'icon': '🚀',
        'simple_explanation': 'Combines all algorithms for the best possible recommendations.',
        'best_for': 'Most users - provides balanced, high-quality results',
        'strengths': ['Best accuracy', 'Balanced diversity', 'Robust'],
        'weaknesses': ['Slower', 'More memory'],
        'technical': 'Weighted ensemble of SVD, User KNN, Item KNN, and Content-Based.'
    }
}


def show_algorithm_help(algorithm_name: str, expanded: bool = False) -> None:
    """
    Display contextual help for a recommendation algorithm.
    
    Args:
        algorithm_name: Name of the algorithm
        expanded: Whether the expander should be expanded by default
    """
    algo_info = ALGORITHM_HELP.get(algorithm_name, ALGORITHM_HELP['Hybrid'])
    
    with st.expander(f"{algo_info['icon']} About {algo_info['name']}", expanded=expanded):
        st.markdown(f"**{algo_info['simple_explanation']}**")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**✅ Strengths:**")
            for strength in algo_info['strengths']:
                st.markdown(f"• {strength}")
        
        with col2:
            st.markdown("**⚠️ Limitations:**")
            for weakness in algo_info['weaknesses']:
                st.markdown(f"• {weakness}")
        
        st.markdown("---")
        st.markdown(f"**Best for:** {algo_info['best_for']}")
        
        with st.expander("🔬 Technical Details"):
            st.caption(algo_info['technical'])


def show_feature_tooltip(
    feature_name: str,
    description: str,
    example: Optional[str] = None
) -> None:
    """
    Display a tooltip for a feature.
    
    Args:
        feature_name: Name of the feature
        description: Description of the feature
        example: Optional example usage
    """
    with st.popover(f"ℹ️ {feature_name}"):
        st.markdown(description)
        if example:
            st.code(example)


def create_info_callout(
    title: str,
    message: str,
    type: str = "info"
) -> None:
    """
    Create an informational callout box.
    
    Args:
        title: Callout title
        message: Callout message
        type: Callout type ('info', 'success', 'warning', 'error')
    """
    icon_map = {
        'info': 'ℹ️',
        'success': '✅',
        'warning': '⚠️',
        'error': '❌'
    }
    
    color_map = {
        'info': ('#4A90D9', 'rgba(74, 144, 217, 0.1)'),
        'success': ('#28A745', 'rgba(40, 167, 69, 0.1)'),
        'warning': ('#F5A623', 'rgba(245, 166, 35, 0.1)'),
        'error': ('#DC3545', 'rgba(220, 53, 69, 0.1)')
    }
    
    icon = icon_map.get(type, 'ℹ️')
    border_color, bg_color = color_map.get(type, color_map['info'])
    
    st.markdown(f"""
    <div style="
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid {border_color};
        background: {bg_color};
        margin: 1rem 0;
    ">
        <strong>{icon} {title}</strong><br>
        <span style="color: #333;">{message}</span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TASK 2.8: Accessibility Features (Enhanced with ARIA labels)
# =============================================================================

def render_accessible_section(
    title: str,
    content: str,
    section_id: str,
    role: str = "region",
    level: int = 2
) -> None:
    """
    Render an accessible section with proper ARIA landmark.
    
    Args:
        title: Section title
        content: HTML content to render
        section_id: Unique ID for the section
        role: ARIA role (region, navigation, complementary, etc.)
        level: Heading level (1-6)
    """
    st.markdown(f"""
    <section id="{section_id}" role="{role}" aria-labelledby="{section_id}-heading">
        <h{level} id="{section_id}-heading" class="sr-only">{title}</h{level}>
        {content}
    </section>
    """, unsafe_allow_html=True)


def render_accessible_card(
    title: str,
    content: str,
    card_id: str,
    is_interactive: bool = False
) -> None:
    """
    Render an accessible card component with proper ARIA attributes.
    
    Args:
        title: Card title
        content: Card content (HTML)
        card_id: Unique ID for the card
        is_interactive: Whether the card is clickable/interactive
    """
    interactive_attrs = 'tabindex="0" role="button"' if is_interactive else ''
    st.markdown(f"""
    <article 
        id="{card_id}" 
        role="article" 
        aria-labelledby="{card_id}-title"
        {interactive_attrs}
        class="movie-card"
    >
        <h3 id="{card_id}-title">{title}</h3>
        <div aria-describedby="{card_id}-content">
            <span id="{card_id}-content">{content}</span>
        </div>
    </article>
    """, unsafe_allow_html=True)


def render_accessible_alert(
    message: str,
    alert_type: str = "info",
    dismissible: bool = False
) -> None:
    """
    Render an accessible alert with proper ARIA live region.
    
    Args:
        message: Alert message
        alert_type: Type of alert ('info', 'success', 'warning', 'error')
        dismissible: Whether the alert can be dismissed
    """
    role = "alert" if alert_type in ("error", "warning") else "status"
    live = "assertive" if alert_type == "error" else "polite"
    
    type_config = {
        'info': {'emoji': 'ℹ️', 'color': '#4A90D9'},
        'success': {'emoji': '✅', 'color': '#28A745'},
        'warning': {'emoji': '⚠️', 'color': '#F5A623'},
        'error': {'emoji': '❌', 'color': '#DC3545'},
    }
    
    config = type_config.get(alert_type, type_config['info'])
    
    st.markdown(f"""
    <div 
        role="{role}" 
        aria-live="{live}"
        aria-atomic="true"
        style="
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid {config['color']};
            background: rgba(0,0,0,0.05);
            margin: 0.5rem 0;
        "
    >
        <span aria-hidden="true">{config['emoji']}</span>
        <span>{message}</span>
    </div>
    """, unsafe_allow_html=True)


def render_accessible_list(
    items: List[str],
    list_label: str,
    ordered: bool = False
) -> None:
    """
    Render an accessible list with proper labeling.
    
    Args:
        items: List items to render
        list_label: Accessible label for the list
        ordered: Whether to use ordered list
    """
    tag = "ol" if ordered else "ul"
    items_html = "\n".join(f'<li>{item}</li>' for item in items)
    
    st.markdown(f"""
    <{tag} aria-label="{list_label}">
        {items_html}
    </{tag}>
    """, unsafe_allow_html=True)


def render_accessible_table(
    df: pd.DataFrame,
    caption: str,
    summary: str
) -> None:
    """
    Render an accessible table with caption and summary.
    
    Args:
        df: DataFrame to render
        caption: Table caption
        summary: Table summary for screen readers
    """
    # Build table HTML with accessibility features
    headers = "".join(f'<th scope="col">{col}</th>' for col in df.columns)
    
    rows = []
    for _, row in df.iterrows():
        cells = "".join(f'<td>{val}</td>' for val in row)
        rows.append(f'<tr>{cells}</tr>')
    
    rows_html = "\n".join(rows)
    
    st.markdown(f"""
    <table role="table" aria-describedby="table-summary">
        <caption>{caption}</caption>
        <span id="table-summary" class="sr-only">{summary}</span>
        <thead>
            <tr>{headers}</tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    """, unsafe_allow_html=True)


def inject_accessibility_css() -> None:
    """
    Inject CSS for accessibility features.
    """
    st.markdown("""
    <style>
    /* Screen reader only class */
    .sr-only {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border: 0;
    }
    
    /* Focus visible for keyboard navigation */
    *:focus-visible {
        outline: 3px solid #4A90D9;
        outline-offset: 2px;
    }
    
    /* High contrast mode support */
    @media (prefers-contrast: high) {
        .movie-card {
            border: 2px solid currentColor;
        }
    }
    
    /* Reduced motion support */
    @media (prefers-reduced-motion: reduce) {
        * {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def accessible_button(
    label: str,
    key: str,
    help_text: str,
    icon: Optional[str] = None,
    type: str = "secondary"
) -> bool:
    """
    Create an accessible button with ARIA-style support.
    
    Args:
        label: Button label
        key: Widget key
        help_text: Accessible description
        icon: Optional icon prefix
        type: Button type ('primary' or 'secondary')
        
    Returns:
        Whether the button was clicked
    """
    full_label = f"{icon} {label}" if icon else label
    return st.button(
        full_label,
        key=key,
        help=help_text,
        type=type
    )


def accessible_metric(
    label: str,
    value: str,
    help_text: str,
    delta: Optional[str] = None,
    delta_color: str = "normal"
) -> None:
    """
    Display a metric with accessible labeling.
    
    Args:
        label: Metric label
        value: Metric value
        help_text: Accessible description
        delta: Optional delta value
        delta_color: Delta color ('normal', 'inverse', 'off')
    """
    st.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color=delta_color,
        help=help_text
    )


def screen_reader_text(text: str) -> None:
    """
    Add text that's only visible to screen readers.
    
    Args:
        text: Text for screen readers
    """
    st.markdown(f"""
    <span style="
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border: 0;
    " aria-label="{text}">{text}</span>
    """, unsafe_allow_html=True)


# =============================================================================
# TASK 2.11: User Preference Memory
# =============================================================================

PREFERENCE_KEYS = {
    'algorithm': 'user_pref_algorithm',
    'num_recommendations': 'user_pref_num_recs',
    'dark_mode': 'user_pref_dark_mode',
    'show_explanations': 'user_pref_explanations',
    'dataset_size': 'user_pref_dataset_size',
    'sort_order': 'user_pref_sort_order',
}


def save_user_preference(key: str, value: Any) -> None:
    """
    Save a user preference to session state.
    
    Args:
        key: Preference key
        value: Preference value
    """
    pref_key = PREFERENCE_KEYS.get(key, f'user_pref_{key}')
    st.session_state[pref_key] = value
    logger.debug(f"Saved preference: {key}={value}")


def load_user_preference(key: str, default: Any = None) -> Any:
    """
    Load a user preference from session state.
    
    Args:
        key: Preference key
        default: Default value if not set
        
    Returns:
        The preference value or default
    """
    pref_key = PREFERENCE_KEYS.get(key, f'user_pref_{key}')
    return st.session_state.get(pref_key, default)


def get_default_algorithm() -> str:
    """
    Get the user's preferred algorithm or system default.
    
    Returns:
        Algorithm name
    """
    return load_user_preference('algorithm', 'Hybrid')


# =============================================================================
# TASK 2.12: Export Functionality
# =============================================================================

def export_to_csv(df: pd.DataFrame, filename: str = "recommendations.csv") -> bytes:
    """
    Export DataFrame to CSV bytes.
    
    Args:
        df: DataFrame to export
        filename: Suggested filename
        
    Returns:
        CSV data as bytes
    """
    return df.to_csv(index=False).encode('utf-8')


def export_to_json(
    data: Union[pd.DataFrame, Dict, List],
    filename: str = "recommendations.json"
) -> bytes:
    """
    Export data to JSON bytes.
    
    Args:
        data: Data to export (DataFrame, dict, or list)
        filename: Suggested filename
        
    Returns:
        JSON data as bytes
    """
    if isinstance(data, pd.DataFrame):
        return data.to_json(orient='records', indent=2).encode('utf-8')
    else:
        return json.dumps(data, indent=2, default=str).encode('utf-8')


def create_download_button(
    data: Union[pd.DataFrame, Dict, List, bytes],
    filename: str,
    label: str = "📥 Download",
    file_type: str = "csv",
    key: Optional[str] = None
) -> bool:
    """
    Create a download button for data export.
    
    Args:
        data: Data to download
        filename: Download filename
        label: Button label
        file_type: File type ('csv', 'json')
        key: Widget key
        
    Returns:
        Whether download was initiated
    """
    if isinstance(data, bytes):
        file_data = data
    elif file_type == "csv":
        if isinstance(data, pd.DataFrame):
            file_data = export_to_csv(data)
        else:
            file_data = pd.DataFrame(data).to_csv(index=False).encode('utf-8')
    else:
        file_data = export_to_json(data)
    
    mime_types = {
        'csv': 'text/csv',
        'json': 'application/json',
    }
    
    return st.download_button(
        label=label,
        data=file_data,
        file_name=filename,
        mime=mime_types.get(file_type, 'application/octet-stream'),
        key=key
    )


# =============================================================================
# TASK 2.5: Mobile Responsive CSS
# =============================================================================

MOBILE_RESPONSIVE_CSS = """
<style>
/* Mobile-first responsive design */
@media (max-width: 768px) {
    /* Stack columns vertically on mobile */
    .stColumns > div {
        flex: 0 0 100% !important;
        max-width: 100% !important;
    }
    
    /* Reduce padding on mobile */
    .main .block-container {
        padding: 1rem 0.5rem !important;
    }
    
    /* Larger touch targets */
    .stButton > button {
        min-height: 48px !important;
        font-size: 16px !important;
    }
    
    /* Readable font sizes */
    .movie-card h3 {
        font-size: 1.1rem !important;
    }
    
    .movie-card p {
        font-size: 0.9rem !important;
    }
    
    /* Hide less important elements on mobile */
    .hide-mobile {
        display: none !important;
    }
}

@media (max-width: 480px) {
    /* Extra small screens */
    .stMetric {
        padding: 0.5rem !important;
    }
    
    .stMetric label {
        font-size: 0.8rem !important;
    }
}

/* Touch-friendly controls */
@media (hover: none) {
    .stButton > button:hover {
        background-color: inherit !important;
    }
    
    /* Remove hover effects on touch devices */
    .movie-card:hover {
        transform: none !important;
    }
}

/* High contrast mode support */
@media (prefers-contrast: high) {
    .movie-card {
        border: 2px solid white !important;
    }
    
    .stButton > button {
        border: 2px solid currentColor !important;
    }
}

/* Reduced motion preference */
@media (prefers-reduced-motion: reduce) {
    * {
        animation: none !important;
        transition: none !important;
    }
}
</style>
"""


def inject_mobile_responsive_css() -> None:
    """Inject mobile-responsive CSS into the page."""
    st.markdown(MOBILE_RESPONSIVE_CSS, unsafe_allow_html=True)


# =============================================================================
# TASK 2.14: Dark Mode Support
# =============================================================================

DARK_MODE_CSS = """
<style>
/* Dark mode styles */
[data-theme="dark"] .movie-card,
.dark-mode .movie-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
    color: white !important;
}

[data-theme="dark"] .stat-card,
.dark-mode .stat-card {
    background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%) !important;
    color: white !important;
}

[data-theme="dark"] .info-box,
.dark-mode .info-box {
    background: #2a2a3e !important;
    color: #e0e0e0 !important;
}

/* Light mode styles */
[data-theme="light"] .movie-card,
.light-mode .movie-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
    color: #333 !important;
}
</style>
"""


def inject_theme_css() -> None:
    """Inject theme-related CSS into the page."""
    st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)


def toggle_dark_mode() -> bool:
    """
    Create a dark mode toggle and return current state.
    
    Returns:
        True if dark mode is enabled
    """
    current_mode = load_user_preference('dark_mode', False)
    
    col1, col2 = st.columns([3, 1])
    with col2:
        new_mode = st.toggle(
            "🌙 Dark Mode",
            value=current_mode,
            key="dark_mode_toggle",
            help="Toggle between light and dark themes"
        )
    
    if new_mode != current_mode:
        save_user_preference('dark_mode', new_mode)
    
    return new_mode


# =============================================================================
# TASK 2.13: Improved Visualization Labels
# =============================================================================

def create_chart_title(
    title: str,
    subtitle: Optional[str] = None,
    data_source: Optional[str] = None
) -> str:
    """
    Create a properly formatted chart title with subtitle.
    
    Args:
        title: Main chart title
        subtitle: Optional subtitle
        data_source: Optional data source attribution
        
    Returns:
        Formatted title string
    """
    full_title = f"<b>{title}</b>"
    
    if subtitle:
        full_title += f"<br><span style='font-size:12px; color:#666;'>{subtitle}</span>"
    
    if data_source:
        full_title += f"<br><span style='font-size:10px; color:#999;'>Source: {data_source}</span>"
    
    return full_title


def format_axis_label(label: str, unit: Optional[str] = None) -> str:
    """
    Format an axis label with optional unit.
    
    Args:
        label: Axis label
        unit: Optional unit (e.g., 'seconds', 'MB')
        
    Returns:
        Formatted label
    """
    if unit:
        return f"{label} ({unit})"
    return label


# =============================================================================
# ERROR BOUNDARY & SAFE COMPONENT RENDERING
# =============================================================================

def with_error_boundary(
    component_func: Callable,
    component_name: str = "Component",
    fallback_message: str = "This section is temporarily unavailable.",
    show_retry: bool = True
) -> Callable:
    """
    Wrap a component function with error boundary to prevent crashes.
    
    Args:
        component_func: The component function to wrap
        component_name: Name for error messages
        fallback_message: Message to show on error
        show_retry: Whether to show retry button
        
    Returns:
        Wrapped function with error handling
    """
    def wrapper(*args, **kwargs):
        try:
            return component_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {component_name}: {e}", exc_info=True)
            st.warning(f"⚠️ **{component_name}**: {fallback_message}")
            
            with st.expander("Error Details", expanded=False):
                st.code(f"{type(e).__name__}: {str(e)}")
            
            if show_retry:
                if st.button(f"🔄 Retry {component_name}", key=f"retry_{component_name}_{id(component_func)}"):
                    st.rerun()
            
            return None
    
    return wrapper


def safe_render(
    render_func: Callable,
    component_key: str,
    fallback: Any = None,
    error_message: str = "Failed to render component"
) -> Any:
    """
    Safely render a component with error handling.
    
    Args:
        render_func: Function to call for rendering
        component_key: Unique key for the component
        fallback: Value to return on error
        error_message: Error message to display
        
    Returns:
        Result of render_func or fallback on error
    """
    try:
        return render_func()
    except Exception as e:
        logger.warning(f"Safe render failed for {component_key}: {e}")
        st.caption(f"⚠️ {error_message}")
        return fallback


def render_with_loading(
    render_func: Callable,
    loading_message: str = "Loading...",
    component_key: str = "component"
) -> Any:
    """
    Render a component with loading indicator.
    
    Args:
        render_func: Function to call for rendering
        loading_message: Message to show while loading
        component_key: Unique key for tracking
        
    Returns:
        Result of render_func
    """
    placeholder = st.empty()
    
    with placeholder:
        with st.spinner(loading_message):
            result = render_func()
    
    return result


def prevent_duplicate_render(component_key: str) -> bool:
    """
    Check if a component has already been rendered this run.
    
    Use this to prevent duplicate component rendering in Streamlit.
    
    Args:
        component_key: Unique key for the component
        
    Returns:
        True if this is the first render, False if duplicate
    """
    render_key = f"_rendered_{component_key}"
    
    if render_key in st.session_state:
        return False
    
    st.session_state[render_key] = True
    return True


def clear_render_tracking() -> None:
    """Clear all render tracking flags. Call at start of page."""
    keys_to_remove = [k for k in st.session_state.keys() if k.startswith("_rendered_")]
    for key in keys_to_remove:
        del st.session_state[key]

