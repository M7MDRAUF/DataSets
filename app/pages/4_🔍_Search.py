"""
CineMatch V2.1.6 - Search Page

User rating history lookup and advanced search functionality.
Answer the professor's question: "How can I see all movies rated by User ID 26?"

Author: CineMatch Development Team
Date: December 5, 2025

Features:
    - User ID validation with bounds checking
    - Rate limiting for search requests
    - Export functionality (Task 2.12)
    - Improved error messages (Task 2.2)
    - Result pagination (Task 2.10)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import html
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_processing import load_ratings, load_movies
from src.search_engine import (
    validate_user_existence,
    get_user_ratings,
    get_user_statistics,
    get_user_genre_preferences,
    get_rating_timeline,
    search_movies_by_criteria
)
from src.utils import format_genres, create_rating_stars, get_genre_emoji, get_tmdb_poster_url, PLACEHOLDER_POSTER
from src.utils.input_validation import validate_user_id, InputValidationError

# Import UI components for improved UX
try:
    from app.components.ui_helpers import (
        show_friendly_error,
        create_download_button,
        create_info_callout,
    )
    UI_COMPONENTS_AVAILABLE = True
except ImportError:
    UI_COMPONENTS_AVAILABLE = False


# Page config
st.set_page_config(
    page_title="CineMatch V2.1.6 - Search",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS with mobile responsiveness (Task 2.5)
st.markdown("""
<style>
.search-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.stat-card {
    background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #E50914;
    margin: 0.5rem 0;
    color: white;
}

.movie-row {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 8px;
    border-left: 3px solid #667eea;
    color: white;
}

.movie-row:hover {
    border-left-color: #E50914;
    transform: translateX(5px);
    transition: all 0.2s;
}

.genre-badge {
    background: linear-gradient(90deg, #E50914, #8B0000);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 15px;
    margin: 0.2rem;
    display: inline-block;
    font-size: 0.85rem;
}

.info-box {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 1rem;
    border-radius: 8px;
    color: white;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="search-header">
    <h1>🔍 User Rating Search</h1>
    <p>Explore rating history and preferences for any user in the system</p>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load ratings and movies data with caching."""
    ratings = load_ratings()
    movies = load_movies()
    return ratings, movies

ratings_df, movies_df = load_data()

# Sidebar - Search Controls
st.sidebar.markdown("## 🔎 Search Options")

# User ID Input
user_id_input = st.sidebar.number_input(
    "Enter User ID",
    min_value=1,
    max_value=330000,
    value=26,  # Default to User 26 per professor's question
    step=1,
    help="Enter a User ID between 1 and 330,000"
)

# Search button
search_clicked = st.sidebar.button("🔍 Search User History", type="primary")

# Additional filters
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Display Options")

show_stats = st.sidebar.checkbox("Show Statistics", value=True)
show_timeline = st.sidebar.checkbox("Show Rating Timeline", value=True)
show_genre_analysis = st.sidebar.checkbox("Show Genre Analysis", value=True)

# Sort options
sort_by = st.sidebar.selectbox(
    "Sort Ratings By",
    ["Most Recent", "Oldest First", "Highest Rating", "Lowest Rating"],
    index=0
)

# Filter by rating
rating_filter = st.sidebar.slider(
    "Filter by Rating",
    min_value=0.5,
    max_value=5.0,
    value=(0.5, 5.0),
    step=0.5,
    help="Show only ratings within this range"
)

# Main content
if search_clicked or user_id_input:
    # Security: Validate user ID with bounds checking
    is_valid, validated_user_id, validation_message = validate_user_id(user_id_input)
    
    if not is_valid:
        st.error(f"❌ Invalid User ID: {validation_message}")
        st.info("💡 Enter a valid positive integer User ID")
        st.stop()
    
    user_id_input = validated_user_id
    
    # Validate user existence in dataset
    if not validate_user_existence(user_id_input, ratings_df):
        st.error(f"❌ User ID {user_id_input} not found in the dataset.")
        st.info("💡 Try a different User ID. Valid range: 1 to 330,000")
        st.stop()
    
    # Get user ratings
    with st.spinner(f"🔎 Loading rating history for User ID {user_id_input}..."):
        user_ratings = get_user_ratings(user_id_input, ratings_df, movies_df)
    
    if user_ratings.empty:
        st.warning(f"⚠️ User ID {user_id_input} exists but has no ratings.")
        st.stop()
    
    # Get statistics
    stats = get_user_statistics(user_ratings)
    
    # Apply rating filter
    user_ratings_filtered = user_ratings[
        (user_ratings['rating'] >= rating_filter[0]) & 
        (user_ratings['rating'] <= rating_filter[1])
    ]
    
    # Apply sorting
    if sort_by == "Most Recent":
        user_ratings_filtered = user_ratings_filtered.sort_values('timestamp', ascending=False)
    elif sort_by == "Oldest First":
        user_ratings_filtered = user_ratings_filtered.sort_values('timestamp', ascending=True)
    elif sort_by == "Highest Rating":
        user_ratings_filtered = user_ratings_filtered.sort_values('rating', ascending=False)
    elif sort_by == "Lowest Rating":
        user_ratings_filtered = user_ratings_filtered.sort_values('rating', ascending=True)
    
    # Display header with user info
    st.markdown(f"""
    <div class="info-box">
        <h2>👤 User ID: {user_id_input}</h2>
        <p><strong>Total Ratings:</strong> {stats['total_ratings']} movies</p>
        <p><strong>Average Rating:</strong> {stats['avg_rating']:.2f} ⭐ | 
           <strong>Median:</strong> {stats['median_rating']:.1f} ⭐</p>
        <p><strong>Active Period:</strong> {stats['first_rating_date'].strftime('%Y-%m-%d') if stats['first_rating_date'] else 'N/A'} 
           to {stats['last_rating_date'].strftime('%Y-%m-%d') if stats['last_rating_date'] else 'N/A'}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistics Section
    if show_stats:
        st.markdown("---")
        st.markdown("## 📊 User Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <h3>{stats['total_ratings']}</h3>
                <p>Total Ratings</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <h3>{stats['avg_rating']:.2f} ⭐</h3>
                <p>Average Rating</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <h3>{stats['std_rating']:.2f}</h3>
                <p>Rating Std Dev</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <h3>{len(stats['top_genres'])}</h3>
                <p>Unique Genres</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Rating Distribution
        st.markdown("### 📈 Rating Distribution")
        rating_dist_df = pd.DataFrame(
            list(stats['rating_distribution'].items()),
            columns=['Rating', 'Count']
        ).sort_values('Rating')
        
        fig_dist = px.bar(
            rating_dist_df,
            x='Rating',
            y='Count',
            title=f'Rating Distribution for User {user_id_input}',
            color='Count',
            color_continuous_scale='Reds'
        )
        fig_dist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig_dist)
    
    # Genre Analysis Section
    if show_genre_analysis and stats['top_genres']:
        st.markdown("---")
        st.markdown("## 🎭 Genre Preferences")
        
        genre_prefs = get_user_genre_preferences(user_ratings)
        
        if not genre_prefs.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Genre preference chart
                fig_genres = px.bar(
                    genre_prefs.head(10),
                    x='preference_score',
                    y='genre',
                    orientation='h',
                    title='Top 10 Genre Preferences (Score = 60% Avg Rating + 40% Frequency)',
                    color='avg_rating',
                    color_continuous_scale='RdYlGn',
                    hover_data=['count', 'avg_rating']
                )
                fig_genres.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig_genres)
            
            with col2:
                st.markdown("### 🏆 Top Genres")
                for idx, row in genre_prefs.head(5).iterrows():
                    emoji = get_genre_emoji(row['genre'])
                    st.markdown(f"""
                    <div class="genre-badge">
                        {emoji} {row['genre']}: {row['count']} movies ({row['avg_rating']:.1f}⭐)
                    </div>
                    """, unsafe_allow_html=True)
    
    # Timeline Section
    if show_timeline:
        st.markdown("---")
        st.markdown("## 📅 Rating Timeline")
        
        timeline = get_rating_timeline(user_ratings)
        
        if not timeline.empty:
            fig_timeline = go.Figure()
            
            fig_timeline.add_trace(go.Scatter(
                x=timeline['date'],
                y=timeline['cumulative_count'],
                mode='lines+markers',
                name='Cumulative Ratings',
                line=dict(color='#E50914', width=2),
                fill='tozeroy'
            ))
            
            fig_timeline.update_layout(
                title=f'Rating Activity Over Time for User {user_id_input}',
                xaxis_title='Date',
                yaxis_title='Cumulative Ratings',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_timeline)
    
    # Ratings Table
    st.markdown("---")
    st.markdown(f"## 🎬 All Ratings ({len(user_ratings_filtered)} movies)")
    
    if len(user_ratings_filtered) != len(user_ratings):
        st.info(f"📌 Showing {len(user_ratings_filtered)} of {len(user_ratings)} ratings (filtered by rating range {rating_filter[0]}-{rating_filter[1]})")
    
    # Export option
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**Sorted by:** {sort_by}")
    with col2:
        # CSV Export
        csv = user_ratings_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"user_{user_id_input}_ratings.csv",
            mime="text/csv"
        )
    with col3:
        # JSON Export (Task 2.12)
        json_data = user_ratings_filtered[['movieId', 'title', 'genres', 'rating', 'timestamp']].copy()
        json_data['timestamp'] = json_data['timestamp'].astype(str)
        st.download_button(
            label="📥 Download JSON",
            data=json_data.to_json(orient='records', indent=2),
            file_name=f"user_{user_id_input}_ratings.json",
            mime="application/json"
        )
    
    # Pagination (Task 2.10)
    ITEMS_PER_PAGE = 20
    total_items = len(user_ratings_filtered)
    total_pages = max(1, (total_items + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
    
    # Initialize page number in session state
    page_key = f"search_page_{user_id_input}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    
    # Pagination controls
    if total_pages > 1:
        st.markdown("---")
        page_col1, page_col2, page_col3, page_col4, page_col5 = st.columns([1, 1, 2, 1, 1])
        
        with page_col1:
            if st.button("⏮️ First", disabled=st.session_state[page_key] == 1, key="first_page"):
                st.session_state[page_key] = 1
                st.rerun()
        
        with page_col2:
            if st.button("◀️ Prev", disabled=st.session_state[page_key] == 1, key="prev_page"):
                st.session_state[page_key] = max(1, st.session_state[page_key] - 1)
                st.rerun()
        
        with page_col3:
            st.markdown(f"<div style='text-align: center; padding-top: 0.5rem;'><strong>Page {st.session_state[page_key]} of {total_pages}</strong></div>", unsafe_allow_html=True)
        
        with page_col4:
            if st.button("Next ▶️", disabled=st.session_state[page_key] == total_pages, key="next_page"):
                st.session_state[page_key] = min(total_pages, st.session_state[page_key] + 1)
                st.rerun()
        
        with page_col5:
            if st.button("Last ⏭️", disabled=st.session_state[page_key] == total_pages, key="last_page"):
                st.session_state[page_key] = total_pages
                st.rerun()
        
        # Show items range
        start_idx = (st.session_state[page_key] - 1) * ITEMS_PER_PAGE
        end_idx = min(start_idx + ITEMS_PER_PAGE, total_items)
        st.caption(f"Showing items {start_idx + 1} - {end_idx} of {total_items}")
    else:
        start_idx = 0
        end_idx = total_items
    
    # Get current page data
    page_data = user_ratings_filtered.iloc[start_idx:end_idx]
    
    # Display ratings in a nice format
    for idx, row in page_data.iterrows():
        poster_path = row.get('poster_path', None)
        poster_url = get_tmdb_poster_url(poster_path)
        
        # HTML escape title and genres
        title_escaped = html.escape(str(row['title']))
        genres_escaped = html.escape(str(row['genres']))
        
        st.markdown(f"""
        <div class="movie-row" style="display: flex; gap: 15px; align-items: center; padding: 10px; 
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                    border-radius: 10px; margin-bottom: 10px;">
            <div style="flex-shrink: 0;">
                <img src="{poster_url}" alt="{title_escaped}" 
                     style="width: 60px; height: 90px; object-fit: cover; border-radius: 6px;"
                     onerror="this.src='{PLACEHOLDER_POSTER}'">
            </div>
            <div style="flex-grow: 1;">
                <h4 style="margin: 0; color: #fff;">{title_escaped}</h4>
                <p style="margin: 5px 0; color: #aaa;">{format_genres(genres_escaped)}</p>
            </div>
            <div style="text-align: center; padding: 0 15px;">
                <div style="color: #FFD700;">{create_rating_stars(row['rating'])}</div>
                <p style="margin: 5px 0; color: #fff;">{row['rating']:.1f} / 5.0</p>
            </div>
            <div style="text-align: center; min-width: 100px;">
                <p style="margin: 0; color: #888; font-size: 0.85em;">Rated On</p>
                <p style="margin: 5px 0; color: #fff;">{row['timestamp'].strftime('%Y-%m-%d')}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show total at bottom
    st.markdown("---")
    st.success(f"✅ Successfully loaded {len(user_ratings_filtered)} ratings for User ID {user_id_input}")

else:
    # Welcome screen
    st.markdown("""
    ## 👋 Welcome to the User Rating Search
    
    This tool allows you to explore the complete rating history of any user in the CineMatch system.
    
    ### 🔍 What You Can Do:
    - **View All Ratings**: See every movie a user has rated
    - **Analyze Preferences**: Discover genre preferences and rating patterns
    - **Track Activity**: Visualize rating timeline and trends
    - **Export Data**: Download rating history as CSV
    
    ### 📝 How to Use:
    1. Enter a User ID (1-330,000) in the sidebar
    2. Click "Search User History"
    3. Explore the comprehensive rating profile
    
    ### 💡 Example: User ID 26
    Try searching for User ID **26** to answer the professor's question!
    """)
    
    # Quick stats about the dataset
    st.markdown("---")
    st.markdown("## 📊 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_users = ratings_df['userId'].nunique()
        st.metric("Total Users", f"{total_users:,}")
    
    with col2:
        total_ratings = len(ratings_df)
        st.metric("Total Ratings", f"{total_ratings:,}")
    
    with col3:
        total_movies = len(movies_df)
        st.metric("Total Movies", f"{total_movies:,}")
