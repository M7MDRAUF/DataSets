"""
CineMatch V2.1 - Loading Animation Component

Lottie animation loader for smooth UX.
"""

import streamlit as st
from streamlit_lottie import st_lottie
import json
from pathlib import Path
from typing import Optional


def load_lottie_file(filename: str) -> Optional[dict]:
    """
    Load Lottie animation JSON from assets folder.
    
    Args:
        filename: Name of JSON file (e.g., 'loading.json')
        
    Returns:
        Lottie animation dict or None if error
    """
    try:
        file_path = Path(__file__).parent.parent / 'assets' / 'animations' / filename
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load animation {filename}: {e}")
        return None


def render_loading_animation(
    animation_type: str = 'loading',
    message: str = "Loading...",
    height: int = 200,
    key: Optional[str] = None
) -> None:
    """
    Render a loading animation with message.
    
    Args:
        animation_type: Type of animation ('loading', 'recommendation', 'training')
        message: Message to display below animation
        height: Animation height in pixels
        key: Unique key for Streamlit widget
    """
    # Animation file mapping
    animation_files = {
        'loading': 'loading.json',
        'recommendation': 'recommendation.json',
        'training': 'training.json'
    }
    
    filename = animation_files.get(animation_type, 'loading.json')
    lottie_animation = load_lottie_file(filename)
    
    # Container for centering
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if lottie_animation:
            st_lottie(
                lottie_animation,
                height=height,
                key=key,
                speed=1.0,
                loop=True
            )
        else:
            # Fallback spinner if animation fails to load
            with st.spinner():
                st.write("")
        
        # Message
        st.markdown(
            f"""
            <div class="loading-text" style="
                text-align: center;
                color: #999;
                font-size: 1.2rem;
                margin-top: 1rem;
            ">
                {message}
            </div>
            """,
            unsafe_allow_html=True
        )


def render_inline_spinner(message: str = "Processing...") -> None:
    """
    Render a simple inline spinner (no Lottie).
    
    Args:
        message: Message to display
    """
    with st.spinner(message):
        st.write("")  # Empty write to trigger spinner


def loading_section(message: str = "Loading data...", animation: str = 'loading'):
    """
    Context manager for loading sections.
    
    Usage:
        with loading_section("Training model...", 'training'):
            # Your code here
            train_model()
    
    Args:
        message: Loading message
        animation: Animation type
    """
    class LoadingContext:
        def __enter__(self):
            render_loading_animation(animation, message)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    
    return LoadingContext()
