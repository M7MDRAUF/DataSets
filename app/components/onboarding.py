"""
CineMatch V2.1.6 - Onboarding Module

First-time user tutorial and walkthrough functionality.
Implements Task 2.7: Create onboarding flow.

Author: CineMatch Development Team
Date: December 5, 2025
"""

import streamlit as st
from typing import Optional, List, Dict

# =============================================================================
# ONBOARDING CONFIGURATION
# =============================================================================

ONBOARDING_STEPS = [
    {
        "title": "Welcome to CineMatch! 🎬",
        "content": """
        CineMatch is an AI-powered movie recommendation system that learns your taste
        and suggests movies you'll love.
        
        **What makes CineMatch special:**
        - 5 different AI algorithms to find your perfect match
        - 32 million ratings from real users
        - 87,000+ movies to discover
        """,
        "icon": "👋"
    },
    {
        "title": "Choose Your Algorithm 🤖",
        "content": """
        CineMatch offers 5 recommendation algorithms:
        
        | Algorithm | Best For |
        |-----------|----------|
        | 🔮 **SVD** | Overall accuracy |
        | 👥 **User KNN** | Finding similar users |
        | 🎬 **Item KNN** | Movies like your favorites |
        | 📝 **Content** | New users, specific genres |
        | 🚀 **Hybrid** | Best overall results |
        
        *Tip: Start with Hybrid for the best experience!*
        """,
        "icon": "🤖"
    },
    {
        "title": "Get Recommendations 🎯",
        "content": """
        To get personalized recommendations:
        
        1. Go to the **🎬 Recommend** page
        2. Enter a User ID (try: 10, 15, 22, or 33)
        3. Select your preferred algorithm
        4. Click **Get Recommendations**
        
        *Each recommendation shows the movie, predicted rating, and why it was chosen!*
        """,
        "icon": "🎯"
    },
    {
        "title": "Explore User History 🔍",
        "content": """
        Want to see what movies a user has rated?
        
        1. Go to the **🔍 Search** page
        2. Enter a User ID
        3. View complete rating history
        4. Analyze genre preferences
        5. Export data to CSV/JSON
        
        *Try User ID 26 to answer the professor's question!*
        """,
        "icon": "🔍"
    },
    {
        "title": "View Analytics 📊",
        "content": """
        The Analytics page provides insights into:
        
        - **Dataset statistics** - Ratings, users, movies
        - **Genre distribution** - Most popular genres
        - **Rating patterns** - How users rate movies
        - **Algorithm performance** - Compare all 5 algorithms
        
        *Use analytics to understand the data behind recommendations!*
        """,
        "icon": "📊"
    },
    {
        "title": "You're Ready! 🚀",
        "content": """
        **Quick Start Checklist:**
        
        ✅ Go to **🎬 Recommend** page  
        ✅ Enter User ID **10** (a great test user)  
        ✅ Select **🚀 Hybrid** algorithm  
        ✅ Click **Get Recommendations**  
        ✅ Explore and enjoy!
        
        *Need help? Look for the ❓ tooltips throughout the app.*
        """,
        "icon": "🚀"
    }
]


# =============================================================================
# ONBOARDING FUNCTIONS
# =============================================================================

def show_onboarding_modal() -> None:
    """Display the onboarding tutorial as a modal dialog."""
    # Check if user has seen onboarding
    if 'onboarding_complete' not in st.session_state:
        st.session_state.onboarding_complete = False
    
    if 'onboarding_step' not in st.session_state:
        st.session_state.onboarding_step = 0
    
    # Only show if not complete
    if st.session_state.onboarding_complete:
        return
    
    current_step = ONBOARDING_STEPS[st.session_state.onboarding_step]
    total_steps = len(ONBOARDING_STEPS)
    
    # Create onboarding container
    with st.container():
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        ">
            <div style="text-align: center;">
                <span style="font-size: 3rem;">{current_step['icon']}</span>
                <h2 style="margin: 1rem 0;">{current_step['title']}</h2>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(current_step['content'])
        
        # Progress indicator
        progress = (st.session_state.onboarding_step + 1) / total_steps
        st.progress(progress)
        st.caption(f"Step {st.session_state.onboarding_step + 1} of {total_steps}")
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.session_state.onboarding_step > 0:
                if st.button("← Previous", key="onboard_prev"):
                    st.session_state.onboarding_step -= 1
                    st.rerun()
        
        with col2:
            if st.button("Skip Tutorial", key="onboard_skip"):
                st.session_state.onboarding_complete = True
                st.rerun()
        
        with col3:
            if st.session_state.onboarding_step < total_steps - 1:
                if st.button("Next →", key="onboard_next", type="primary"):
                    st.session_state.onboarding_step += 1
                    st.rerun()
            else:
                if st.button("Get Started! 🚀", key="onboard_finish", type="primary"):
                    st.session_state.onboarding_complete = True
                    st.rerun()


def show_feature_tour(feature_name: str) -> None:
    """Show a mini-tutorial for a specific feature."""
    feature_tours = {
        'algorithm_selection': {
            'title': '🤖 Algorithm Selection',
            'tips': [
                "SVD is best for users with many ratings",
                "Content-Based works well for new users",
                "Hybrid combines all algorithms for best results"
            ]
        },
        'recommendations': {
            'title': '🎯 Getting Recommendations',
            'tips': [
                "Enter any User ID from 1 to 330,000",
                "Click 'Explain' to understand why movies were recommended",
                "Export your recommendations to CSV or JSON"
            ]
        },
        'search': {
            'title': '🔍 User Search',
            'tips': [
                "Search for any user's complete rating history",
                "Filter by rating range and sort order",
                "Export data for further analysis"
            ]
        }
    }
    
    if feature_name in feature_tours:
        tour = feature_tours[feature_name]
        with st.expander(f"💡 Tips: {tour['title']}", expanded=False):
            for tip in tour['tips']:
                st.markdown(f"• {tip}")


def check_first_visit() -> bool:
    """Check if this is the user's first visit."""
    if 'first_visit' not in st.session_state:
        st.session_state.first_visit = True
        return True
    return False


def show_welcome_banner() -> None:
    """Show a welcome banner for first-time users."""
    if check_first_visit():
        st.info("""
        👋 **Welcome to CineMatch!** 
        
        Looks like this is your first visit. Would you like a quick tour?
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📚 Start Tutorial", key="start_tour", type="primary"):
                st.session_state.onboarding_step = 0
                st.session_state.onboarding_complete = False
                st.rerun()
        with col2:
            if st.button("⏭️ Skip, I know what I'm doing", key="skip_tour"):
                st.session_state.first_visit = False
                st.session_state.onboarding_complete = True
                st.rerun()


def reset_onboarding() -> None:
    """Reset the onboarding state to show the tutorial again."""
    st.session_state.onboarding_complete = False
    st.session_state.onboarding_step = 0
    st.session_state.first_visit = True


# =============================================================================
# KEYBOARD SHORTCUTS HELP (Task 2.6)
# =============================================================================

def show_keyboard_shortcuts() -> None:
    """Display keyboard shortcuts help."""
    with st.expander("⌨️ Keyboard Shortcuts", expanded=False):
        st.markdown("""
        | Shortcut | Action |
        |----------|--------|
        | `Ctrl + Enter` | Submit form / Get recommendations |
        | `Tab` | Navigate between inputs |
        | `Escape` | Close dialogs |
        | `?` | Show help |
        
        *Note: Some shortcuts may vary by browser.*
        """)
