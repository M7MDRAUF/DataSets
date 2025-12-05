"""
CineMatch V2.1.2 - Movie Search Page

Search for movies and view all users who rated/liked them.
Answer the professor's question: "Show me all users who liked Movie X"

Author: CineMatch Development Team
Date: November 20, 2025
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import html
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_processing import load_ratings, load_movies
from src.utils import format_genres, create_rating_stars, get_genre_emoji, get_tmdb_poster_url, PLACEHOLDER_POSTER


# Page config
st.set_page_config(
    page_title="CineMatch V2.1.2 - Movie Search",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.movie-search-header {
    background: linear-gradient(135deg, #E50914 0%, #8B0000 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.movie-detail-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 2rem;
    border-radius: 10px;
    border-left: 5px solid #E50914;
    color: white;
    margin: 1rem 0;
}

.user-card {
    background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 3px solid #667eea;
    color: white;
}

.user-card:hover {
    border-left-color: #E50914;
    transform: translateX(5px);
    transition: all 0.2s;
}

.stat-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    color: white;
    margin: 0.5rem;
}

.rating-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: bold;
    margin: 0.2rem;
}

.rating-high {
    background: linear-gradient(90deg, #32CD32, #228B22);
    color: white;
}

.rating-medium {
    background: linear-gradient(90deg, #FFA500, #FF8C00);
    color: white;
}

.rating-low {
    background: linear-gradient(90deg, #DC143C, #8B0000);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="movie-search-header">
    <h1>🎬 Movie Search</h1>
    <p>Find movies and discover who loved them</p>
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
st.sidebar.markdown("## 🔍 Search Options")

# Movie search input
search_query = st.sidebar.text_input(
    "Search Movie Title",
    placeholder="e.g., Inception, Matrix, Godfather...",
    help="Enter partial or full movie title"
)

# Minimum rating filter
min_rating = st.sidebar.slider(
    "Minimum User Rating",
    min_value=0.5,
    max_value=5.0,
    value=4.0,
    step=0.5,
    help="Show only users who rated the movie at or above this threshold"
)

# Sort options
sort_by = st.sidebar.selectbox(
    "Sort Users By",
    ["Highest Rating", "Lowest Rating", "Most Recent", "Oldest First"],
    index=0
)

# Limit results
max_users_display = st.sidebar.slider(
    "Max Users to Display",
    min_value=10,
    max_value=200,
    value=50,
    step=10,
    help="Limit number of users shown"
)

# Search button
search_clicked = st.sidebar.button("🔍 Search Movie", type="primary")

# Main content
if search_clicked and search_query:
    # Search for movies matching the query
    matching_movies = movies_df[
        movies_df['title'].str.contains(search_query, case=False, na=False)
    ]
    
    if matching_movies.empty:
        st.error(f"❌ No movies found matching '{search_query}'")
        st.info("💡 Try a different search term or check your spelling")
        st.stop()
    
    # If multiple matches, show selection
    if len(matching_movies) > 1:
        st.markdown("## 🎯 Multiple Matches Found")
        st.info(f"Found {len(matching_movies)} movies matching '{search_query}'. Select one below:")
        
        # Create selection list
        movie_options = {}
        for idx, row in matching_movies.head(20).iterrows():
            display_name = f"{row['title']} - {format_genres(row['genres'], max_genres=3)}"
            movie_options[display_name] = row['movieId']
        
        selected_movie_display = st.selectbox(
            "Select a movie:",
            options=list(movie_options.keys())
        )
        
        selected_movie_id = movie_options[selected_movie_display]
        selected_movie = movies_df[movies_df['movieId'] == selected_movie_id].iloc[0]
    else:
        selected_movie = matching_movies.iloc[0]
        selected_movie_id = selected_movie['movieId']
    
    # Get poster URL
    poster_path = selected_movie.get('poster_path', None)
    poster_url = get_tmdb_poster_url(poster_path)
    
    # HTML escape title and genres
    title_escaped = html.escape(str(selected_movie['title']))
    genres_escaped = html.escape(str(selected_movie['genres']))
    
    # Display movie details with poster
    st.markdown("---")
    st.markdown(f"""
    <div class="movie-detail-card" style="display: flex; gap: 20px; align-items: flex-start;">
        <div style="flex-shrink: 0;">
            <img src="{poster_url}" alt="{title_escaped}" 
                 style="width: 150px; height: 225px; object-fit: cover; border-radius: 10px; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.4);"
                 onerror="this.src='{PLACEHOLDER_POSTER}'">
        </div>
        <div style="flex-grow: 1;">
            <h2 style="margin: 0 0 10px 0;">🎬 {title_escaped}</h2>
            <p style="margin: 5px 0;"><strong>Genres:</strong> {format_genres(genres_escaped)}</p>
            <p style="margin: 5px 0;"><strong>Movie ID:</strong> {selected_movie_id}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get all ratings for this movie
    movie_ratings = ratings_df[ratings_df['movieId'] == selected_movie_id].copy()
    
    if movie_ratings.empty:
        st.warning(f"⚠️ No ratings found for '{selected_movie['title']}'")
        st.stop()
    
    # Filter by minimum rating
    movie_ratings_filtered = movie_ratings[movie_ratings['rating'] >= min_rating]
    
    if movie_ratings_filtered.empty:
        st.warning(f"⚠️ No users rated this movie {min_rating}⭐ or higher")
        st.info(f"💡 Try lowering the minimum rating filter (currently {min_rating}⭐)")
        st.stop()
    
    # Convert timestamp
    movie_ratings_filtered['timestamp'] = pd.to_datetime(
        movie_ratings_filtered['timestamp'], 
        unit='s'
    )
    
    # Apply sorting
    if sort_by == "Highest Rating":
        movie_ratings_filtered = movie_ratings_filtered.sort_values('rating', ascending=False)
    elif sort_by == "Lowest Rating":
        movie_ratings_filtered = movie_ratings_filtered.sort_values('rating', ascending=True)
    elif sort_by == "Most Recent":
        movie_ratings_filtered = movie_ratings_filtered.sort_values('timestamp', ascending=False)
    elif sort_by == "Oldest First":
        movie_ratings_filtered = movie_ratings_filtered.sort_values('timestamp', ascending=True)
    
    # Calculate statistics
    total_ratings = len(movie_ratings)
    total_likes = len(movie_ratings_filtered)
    avg_rating = movie_ratings['rating'].mean()
    median_rating = movie_ratings['rating'].median()
    
    # Display statistics
    st.markdown("---")
    st.markdown("## 📊 Movie Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <h3>{total_ratings:,}</h3>
            <p>Total Ratings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <h3>{total_likes:,}</h3>
            <p>Users (≥{min_rating}⭐)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <h3>{avg_rating:.2f} ⭐</h3>
            <p>Average Rating</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-box">
            <h3>{median_rating:.1f} ⭐</h3>
            <p>Median Rating</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Rating distribution
    st.markdown("### 📈 Rating Distribution")
    
    rating_dist = movie_ratings['rating'].value_counts().sort_index()
    rating_dist_df = pd.DataFrame({
        'Rating': rating_dist.index,
        'Count': rating_dist.values
    })
    
    fig_dist = px.bar(
        rating_dist_df,
        x='Rating',
        y='Count',
        title=f'Rating Distribution for "{selected_movie["title"]}"',
        color='Count',
        color_continuous_scale='RdYlGn'
    )
    fig_dist.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_dist)
    
    # Users list
    st.markdown("---")
    st.markdown(f"## 👥 Users Who Rated This Movie (Showing {min(len(movie_ratings_filtered), max_users_display)} of {len(movie_ratings_filtered)})")
    
    if len(movie_ratings_filtered) != total_ratings:
        st.info(f"📌 Filtered to show only users who rated ≥{min_rating}⭐ (sorted by {sort_by})")
    
    # Export option
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**Total Users Listed:** {min(len(movie_ratings_filtered), max_users_display)}")
    with col2:
        csv = movie_ratings_filtered.head(max_users_display).to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"movie_{selected_movie_id}_users.csv",
            mime="text/csv"
        )
    
    # Display users in a nice format
    for idx, (_, row) in enumerate(movie_ratings_filtered.head(max_users_display).iterrows()):
        # Determine rating badge style
        if row['rating'] >= 4.5:
            badge_class = "rating-high"
            emoji = "😍"
        elif row['rating'] >= 3.5:
            badge_class = "rating-medium"
            emoji = "👍"
        else:
            badge_class = "rating-low"
            emoji = "😐"
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="user-card">
                <h4>👤 User ID: {row['userId']}</h4>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="user-card" style="text-align: center;">
                <div class="rating-badge {badge_class}">
                    {emoji} {row['rating']:.1f} ⭐
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="user-card" style="text-align: center;">
                <p><strong>Rated On:</strong></p>
                <p>{row['timestamp'].strftime('%Y-%m-%d')}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Show total at bottom
    st.markdown("---")
    st.success(f"✅ Successfully found {len(movie_ratings_filtered)} users who rated '{selected_movie['title']}' ≥{min_rating}⭐")
    
    # Timeline visualization
    if len(movie_ratings_filtered) > 10:
        st.markdown("---")
        st.markdown("## 📅 Rating Timeline")
        
        timeline_df = movie_ratings_filtered.copy()
        timeline_df['date'] = timeline_df['timestamp'].dt.date
        timeline_data = timeline_df.groupby('date').size().reset_index(name='count')
        timeline_data = timeline_data.sort_values('date')
        timeline_data['cumulative'] = timeline_data['count'].cumsum()
        
        fig_timeline = go.Figure()
        
        fig_timeline.add_trace(go.Scatter(
            x=timeline_data['date'],
            y=timeline_data['cumulative'],
            mode='lines+markers',
            name='Cumulative Ratings',
            line=dict(color='#E50914', width=2),
            fill='tozeroy'
        ))
        
        fig_timeline.update_layout(
            title=f'Cumulative Ratings Over Time',
            xaxis_title='Date',
            yaxis_title='Total Ratings',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_timeline)

else:
    # Welcome screen
    st.markdown("""
    ## 👋 Welcome to Movie Search
    
    This tool allows you to search for any movie and discover all users who rated it.
    
    ### 🔍 What You Can Do:
    - **Search Movies**: Find any movie by title (partial matches supported)
    - **View User Ratings**: See all users who rated the movie
    - **Filter by Rating**: Focus on users who loved it (≥4⭐)
    - **Analyze Patterns**: View rating distribution and timeline
    - **Export Data**: Download user lists as CSV
    
    ### 📝 How to Use:
    1. Enter a movie title in the sidebar (e.g., "Inception", "Matrix")
    2. Click "Search Movie"
    3. If multiple matches, select the correct movie
    4. Explore user ratings and statistics
    
    ### 💡 Example Searches:
    - "The Dark Knight"
    - "Forrest Gump"
    - "Star Wars"
    - "Pulp Fiction"
    """)
    
    # Quick stats about the dataset
    st.markdown("---")
    st.markdown("## 📊 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_movies = len(movies_df)
        st.metric("Total Movies", f"{total_movies:,}")
    
    with col2:
        total_ratings = len(ratings_df)
        st.metric("Total Ratings", f"{total_ratings:,}")
    
    with col3:
        avg_ratings_per_movie = total_ratings / total_movies
        st.metric("Avg Ratings/Movie", f"{avg_ratings_per_movie:.0f}")
