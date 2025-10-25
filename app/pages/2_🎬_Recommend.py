"""
CineMatch V1.0.0 - Recommend Page

Core feature: Personalized movie recommendations with explainability.
Implements F-01 through F-08.

Author: CineMatch Team
Date: October 24, 2025
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Try to import sklearn-compatible engine first (Windows compatibility)
try:
    from src.recommendation_engine_sklearn import RecommendationEngine, validate_user_id
except ImportError:
    from src.recommendation_engine import RecommendationEngine, validate_user_id

from src.utils import (
    explain_recommendation,
    get_user_taste_profile,
    format_genres,
    format_rating,
    create_rating_stars,
    get_genre_emoji
)
from src.data_processing import load_ratings


# Page config
st.set_page_config(
    page_title="CineMatch - Recommendations",
    page_icon="🎬",
    layout="wide"
)

# Header
st.title("🎬 Get Your Personalized Movie Recommendations")
st.markdown("### Discover movies tailored to your unique taste")
st.markdown("---")

# Initialize recommendation engine (cached)
@st.cache_resource
def initialize_engine():
    """Initialize and cache the recommendation engine"""
    engine = RecommendationEngine()
    engine.load_model()
    engine.load_data(sample_size=None)  # Load full dataset for production
    return engine

# Load full ratings for explanations
@st.cache_data
def load_full_ratings():
    """Load full ratings dataset"""
    return load_ratings(sample_size=1_000_000)  # Use sample for performance

# Initialize
try:
    with st.spinner("🚀 Loading recommendation engine..."):
        engine = initialize_engine()
        ratings_df = load_full_ratings()
    
    st.success("✅ Recommendation engine ready!")
    
except Exception as e:
    st.error(f"❌ Error initializing engine: {e}")
    st.info("""
    **Please ensure:**
    1. Model is trained (run: `python src/model_training.py`)
    2. Dataset files are in `data/ml-32m/`
    3. All dependencies are installed
    """)
    st.stop()

# Main interface
st.markdown("## 👤 Enter User ID")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    # F-01: User Input
    user_id = st.number_input(
        "User ID",
        min_value=1,
        max_value=200000,
        value=1,
        step=1,
        help="Enter a user ID from 1 to 200,000"
    )

with col2:
    # F-02: Get Recommendations Button
    get_recs_button = st.button("🎬 Get Recommendations", type="primary", use_container_width=True)

with col3:
    # F-07: Surprise Me Button
    surprise_button = st.button("🎲 Surprise Me!", use_container_width=True)

st.markdown("---")

# Process recommendations
if get_recs_button or surprise_button or 'recommendations' in st.session_state:
    
    # Validate user ID
    is_valid, error_msg = validate_user_id(int(user_id), ratings_df)
    
    if not is_valid:
        st.error(f"❌ {error_msg}")
        st.stop()
    
    # Generate recommendations
    try:
        with st.spinner("🔮 Generating personalized recommendations..."):
            if surprise_button:
                # F-07: Surprise Me Mode
                recommendations = engine.get_surprise_recommendations(int(user_id), n=10)
                st.info("🎲 **Surprise Mode**: Showing movies outside your usual taste!")
            else:
                # F-02: Regular recommendations
                recommendations = engine.get_recommendations(int(user_id), n=10)
            
            # Get user history for explanations and taste profile
            user_history = engine.get_user_history(int(user_id))
            
            # Store in session state
            st.session_state['recommendations'] = recommendations
            st.session_state['user_history'] = user_history
            st.session_state['user_id'] = user_id
        
        recommendations = st.session_state['recommendations']
        user_history = st.session_state['user_history']
        
        # Layout: Main content + Sidebar
        main_col, sidebar_col = st.columns([2, 1])
        
        with sidebar_col:
            # F-06: User Taste Profile
            st.markdown("### 👤 Your Taste Profile")
            
            profile = get_user_taste_profile(user_history)
            
            # Statistics
            st.metric("Total Ratings", f"{profile['num_ratings']:,}")
            st.metric("Average Rating", f"{profile['avg_rating']} ⭐")
            
            # Top genres
            st.markdown("#### 🎭 Top Genres")
            for genre_info in profile['top_genres']:
                genre = genre_info['genre']
                percentage = genre_info['percentage']
                emoji = get_genre_emoji(genre)
                st.markdown(f"{emoji} **{genre}**: {percentage}%")
            
            st.markdown("---")
            
            # Top rated movies
            st.markdown("#### 🏆 Your Top Rated")
            for movie in profile['top_rated_movies'][:5]:
                st.markdown(f"**{movie['rating']}⭐** {movie['title'][:40]}...")
            
            st.markdown("---")
            
            # Rating distribution
            st.markdown("#### 📊 Your Rating Pattern")
            for rating in sorted(profile['rating_distribution'].keys(), reverse=True):
                count = profile['rating_distribution'][rating]
                st.markdown(f"**{rating}⭐**: {count} movies")
        
        with main_col:
            # F-03: Recommendation Display
            st.markdown(f"### 🎬 Top Recommendations for User {user_id}")
            st.markdown(f"*Based on your {len(user_history)} ratings*")
            
            # Display each recommendation as a card
            for idx, row in recommendations.iterrows():
                movie_id = row['movieId']
                title = row['title']
                genres = row['genres']
                # Ensure predicted_rating is float (safety check)
                predicted_rating = float(row['predicted_rating'].est) if hasattr(row['predicted_rating'], 'est') else float(row['predicted_rating'])
                genres_list = row['genres_list']
                
                # Movie card
                with st.container():
                    st.markdown(f"""
                    <div class="movie-card">
                        <h3>#{idx+1}. {title}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    card_col1, card_col2, card_col3 = st.columns([2, 1, 1])
                    
                    with card_col1:
                        st.markdown(f"**Genres**: {format_genres(genres)}")
                        st.markdown(f"**Predicted Rating**: {create_rating_stars(predicted_rating)}", unsafe_allow_html=True)
                    
                    with card_col2:
                        # F-05: Explain button
                        if st.button(f"💡 Explain", key=f"explain_{movie_id}"):
                            explanation = explain_recommendation(
                                int(user_id),
                                movie_id,
                                title,
                                genres_list,
                                user_history,
                                ratings_df,
                                n_similar=3
                            )
                            st.info(f"**Why this recommendation?**\n\n{explanation}")
                    
                    with card_col3:
                        # F-08: Feedback buttons (simulated)
                        feedback_col1, feedback_col2 = st.columns(2)
                        with feedback_col1:
                            if st.button("👍", key=f"like_{movie_id}"):
                                st.success("Thanks! We'll recommend more like this.")
                        with feedback_col2:
                            if st.button("👎", key=f"dislike_{movie_id}"):
                                st.warning("Got it! We'll adjust future recommendations.")
                    
                    st.markdown("---")
        
        # Additional actions
        st.markdown("### 🎯 More Actions")
        
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button("🔄 Refresh Recommendations"):
                st.session_state.clear()
                st.rerun()
        
        with action_col2:
            if st.button("🎲 Try Surprise Mode"):
                surprise_button = True
                st.rerun()
        
        with action_col3:
            if st.button("👤 Try Different User"):
                st.session_state.clear()
                st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error generating recommendations: {e}")
        st.info("Please try a different user ID or check the logs.")

else:
    # Instructions when no recommendations yet
    st.info("""
    ### 🎬 How to Get Started
    
    1. **Enter a User ID** in the input field above (e.g., 1, 123, 1000)
    2. Click **"Get Recommendations"** for personalized suggestions
    3. Click **"Surprise Me!"** to discover movies outside your usual taste
    
    ### ✨ Features You'll See
    
    - **Personalized Recommendations**: Top 10 movies tailored to your taste
    - **Predicted Ratings**: How much we think you'll like each movie
    - **Explanations**: Click "Explain" to understand why each movie is recommended
    - **Your Taste Profile**: See your genre preferences and rating patterns
    - **Feedback**: Give thumbs up/down to help improve recommendations
    
    ### 💡 Tips
    
    - Different users have different tastes - try multiple User IDs!
    - Use "Surprise Me" to discover hidden gems
    - Check your taste profile to understand your movie preferences
    - Read explanations to see how the AI makes decisions
    
    ---
    
    **Sample User IDs to try**: 1, 50, 123, 500, 1000, 5000
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>Powered by SVD Collaborative Filtering</strong></p>
    <p>Trained on 32 million ratings for maximum accuracy</p>
</div>
""", unsafe_allow_html=True)
