"""
CineMatch V2.1 - Algorithm Selector Component

Enhanced algorithm selection with visual menu and descriptions.
"""

import streamlit as st
from streamlit_option_menu import option_menu
from typing import Optional, Dict


# Algorithm metadata
ALGORITHM_INFO = {
    'SVD': {
        'icon': '🎯',
        'name': 'SVD',
        'full_name': 'Singular Value Decomposition',
        'description': 'Matrix factorization technique that predicts ratings by decomposing the user-item matrix.',
        'best_for': 'Large datasets with collaborative filtering',
        'pros': ['High accuracy', 'Handles sparsity well', 'Scalable'],
        'cons': ['Requires training time', 'Cold start problem']
    },
    'KNN': {
        'icon': '👥',
        'name': 'KNN',
        'full_name': 'K-Nearest Neighbors',
        'description': 'Recommends items based on similar users or items using nearest neighbors approach.',
        'best_for': 'Finding similar users/items',
        'pros': ['Interpretable', 'No training needed', 'Simple'],
        'cons': ['Slower predictions', 'Memory intensive']
    },
    'Content-Based': {
        'icon': '🎬',
        'name': 'Content-Based',
        'full_name': 'Content-Based Filtering',
        'description': 'Recommends movies similar to ones you liked based on genres, tags, and content features.',
        'best_for': 'New users, genre-based recommendations',
        'pros': ['No cold start', 'Explainable', 'Fast'],
        'cons': ['Limited novelty', 'Requires item features']
    }
}


def render_algorithm_selector(
    default_algorithm: str = 'SVD',
    horizontal: bool = True,
    key: str = "algo_selector"
) -> str:
    """
    Render enhanced algorithm selector with visual menu.
    
    Args:
        default_algorithm: Default selected algorithm
        horizontal: Use horizontal menu layout
        key: Unique key for the widget
        
    Returns:
        Selected algorithm name
    """
    # Get algorithm names and icons
    algorithms = list(ALGORITHM_INFO.keys())
    icons = [ALGORITHM_INFO[algo]['icon'] for algo in algorithms]
    
    # Default index
    default_index = algorithms.index(default_algorithm) if default_algorithm in algorithms else 0
    
    # Render option menu
    selected = option_menu(
        menu_title="Select Algorithm",
        options=algorithms,
        icons=icons,
        default_index=default_index,
        orientation="horizontal" if horizontal else "vertical",
        key=key,
        styles={
            "container": {
                "padding": "0.5rem",
                "background-color": "#222222",
                "border-radius": "8px"
            },
            "icon": {
                "color": "#E50914",
                "font-size": "1.5rem"
            },
            "nav-link": {
                "font-size": "1rem",
                "text-align": "center",
                "margin": "0.25rem",
                "background-color": "#333333",
                "border-radius": "4px",
                "color": "white"
            },
            "nav-link-selected": {
                "background-color": "#E50914",
                "color": "white",
                "font-weight": "600"
            }
        }
    )
    
    return selected


def render_algorithm_info(algorithm: str) -> None:
    """
    Render detailed information about selected algorithm.
    
    Args:
        algorithm: Algorithm name
    """
    if algorithm not in ALGORITHM_INFO:
        st.warning(f"Unknown algorithm: {algorithm}")
        return
    
    info = ALGORITHM_INFO[algorithm]
    
    # Info card
    st.markdown(
        f"""
        <div class="algorithm-info-card" style="
            background: linear-gradient(135deg, #222222 0%, #333333 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            border-left: 4px solid #E50914;
        ">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <span style="font-size: 2.5rem; margin-right: 1rem;">{info['icon']}</span>
                <div>
                    <h3 style="margin: 0; color: white;">{info['full_name']}</h3>
                    <p style="margin: 0; color: #999; font-size: 0.9rem;">Best for: {info['best_for']}</p>
                </div>
            </div>
            
            <p style="color: #DDD; font-size: 1rem; line-height: 1.6; margin-bottom: 1rem;">
                {info['description']}
            </p>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                <div>
                    <div style="color: #4CAF50; font-weight: 600; margin-bottom: 0.5rem;">✓ Pros:</div>
                    <ul style="color: #CCC; font-size: 0.9rem; margin: 0; padding-left: 1.5rem;">
                        {''.join([f'<li>{pro}</li>' for pro in info['pros']])}
                    </ul>
                </div>
                <div>
                    <div style="color: #F44336; font-weight: 600; margin-bottom: 0.5rem;">⚠ Cons:</div>
                    <ul style="color: #CCC; font-size: 0.9rem; margin: 0; padding-left: 1.5rem;">
                        {''.join([f'<li>{con}</li>' for con in info['cons']])}
                    </ul>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_algorithm_comparison() -> None:
    """
    Render comparison table of all algorithms.
    """
    st.markdown("### Algorithm Comparison")
    
    # Build comparison HTML
    comparison_html = """
    <table style="
        width: 100%;
        background: #222222;
        border-radius: 8px;
        overflow: hidden;
        border-collapse: collapse;
    ">
        <thead>
            <tr style="background: #E50914;">
                <th style="padding: 1rem; color: white; text-align: left;">Algorithm</th>
                <th style="padding: 1rem; color: white; text-align: center;">Accuracy</th>
                <th style="padding: 1rem; color: white; text-align: center;">Speed</th>
                <th style="padding: 1rem; color: white; text-align: center;">Scalability</th>
                <th style="padding: 1rem; color: white; text-align: center;">Cold Start</th>
            </tr>
        </thead>
        <tbody>
    """
    
    # Algorithm ratings
    ratings = {
        'SVD': {'accuracy': '⭐⭐⭐⭐⭐', 'speed': '⭐⭐⭐', 'scalability': '⭐⭐⭐⭐⭐', 'cold_start': '⭐⭐'},
        'KNN': {'accuracy': '⭐⭐⭐⭐', 'speed': '⭐⭐', 'scalability': '⭐⭐⭐', 'cold_start': '⭐⭐⭐'},
        'Content-Based': {'accuracy': '⭐⭐⭐', 'speed': '⭐⭐⭐⭐⭐', 'scalability': '⭐⭐⭐⭐', 'cold_start': '⭐⭐⭐⭐⭐'}
    }
    
    for algo, info in ALGORITHM_INFO.items():
        rating = ratings[algo]
        comparison_html += f"""
        <tr style="border-bottom: 1px solid #333;">
            <td style="padding: 1rem; color: white;">
                <strong>{info['icon']} {info['name']}</strong>
                <div style="font-size: 0.85rem; color: #999;">{info['full_name']}</div>
            </td>
            <td style="padding: 1rem; text-align: center; color: #FFD700;">{rating['accuracy']}</td>
            <td style="padding: 1rem; text-align: center; color: #FFD700;">{rating['speed']}</td>
            <td style="padding: 1rem; text-align: center; color: #FFD700;">{rating['scalability']}</td>
            <td style="padding: 1rem; text-align: center; color: #FFD700;">{rating['cold_start']}</td>
        </tr>
        """
    
    comparison_html += """
        </tbody>
    </table>
    """
    
    st.markdown(comparison_html, unsafe_allow_html=True)
    
    # Legend
    st.markdown(
        """
        <div style="margin-top: 1rem; color: #999; font-size: 0.85rem;">
            <strong>Legend:</strong> ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Very Good | ⭐⭐⭐ Good | ⭐⭐ Fair
        </div>
        """,
        unsafe_allow_html=True
    )
