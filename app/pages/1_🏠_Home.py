"""
CineMatch V1.0.0 - Home Page

Overview, statistics, and dataset visualizations.

Author: CineMatch Team
Date: October 24, 2025
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing import load_movies, load_ratings


# Page config
st.set_page_config(
    page_title="CineMatch - Home",
    page_icon="🏠",
    layout="wide"
)

# Header
st.title("🏠 CineMatch - Home")
st.markdown("### Intelligent Movie Recommendation Engine")
st.markdown("---")

# Load data (cached)
@st.cache_data
def load_data_cached():
    """Load and cache dataset"""
    movies_df = load_movies()
    # Load a sample of ratings for faster stats (not the full 32M)
    ratings_df = load_ratings(sample_size=1_000_000)
    return movies_df, ratings_df

try:
    with st.spinner("Loading dataset..."):
        movies_df, ratings_df = load_data_cached()
    
    st.success("✅ Dataset loaded successfully")
    
    # Overview statistics
    st.markdown("## 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Movies",
            value=f"{len(movies_df):,}",
            delta="In catalog"
        )
    
    with col2:
        st.metric(
            label="Total Ratings",
            value="32,000,000+",
            delta="Full dataset"
        )
    
    with col3:
        st.metric(
            label="Unique Users",
            value=f"{ratings_df['userId'].nunique():,}",
            delta="Active users"
        )
    
    with col4:
        avg_rating = ratings_df['rating'].mean()
        st.metric(
            label="Average Rating",
            value=f"{avg_rating:.2f} ⭐",
            delta="Out of 5.0"
        )
    
    st.markdown("---")
    
    # Genre distribution
    st.markdown("## 🎭 Genre Distribution")
    st.markdown("Explore the most popular movie genres in our catalog:")
    
    # Process genres
    genres_exploded = movies_df['genres'].str.split('|', expand=True).stack()
    genre_counts = genres_exploded.value_counts().head(15)
    
    # Create bar chart
    fig_genres = px.bar(
        x=genre_counts.values,
        y=genre_counts.index,
        orientation='h',
        labels={'x': 'Number of Movies', 'y': 'Genre'},
        title='Top 15 Movie Genres',
        color=genre_counts.values,
        color_continuous_scale='Viridis'
    )
    
    fig_genres.update_layout(
        height=500,
        showlegend=False,
        xaxis_title="Number of Movies",
        yaxis_title="Genre"
    )
    
    st.plotly_chart(fig_genres, use_container_width=True)
    
    st.markdown("---")
    
    # Rating distribution
    st.markdown("## ⭐ Rating Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Distribution of User Ratings")
        
        rating_counts = ratings_df['rating'].value_counts().sort_index()
        
        fig_ratings = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            labels={'x': 'Rating', 'y': 'Count'},
            title='How users rate movies',
            color=rating_counts.index,
            color_continuous_scale='RdYlGn'
        )
        
        fig_ratings.update_layout(height=400)
        st.plotly_chart(fig_ratings, use_container_width=True)
    
    with col2:
        st.markdown("### Rating Statistics")
        
        st.metric("Mean Rating", f"{ratings_df['rating'].mean():.2f}")
        st.metric("Median Rating", f"{ratings_df['rating'].median():.2f}")
        st.metric("Mode (Most Common)", f"{ratings_df['rating'].mode()[0]:.1f}")
        st.metric("Standard Deviation", f"{ratings_df['rating'].std():.2f}")
        
        st.info("""
        **Insight**: Most users tend to rate movies between 3.0 and 4.0, 
        indicating a generally positive viewing experience.
        """)
    
    st.markdown("---")
    
    # Top rated movies
    st.markdown("## 🏆 Top Rated Movies")
    st.markdown("Movies with the highest average ratings (minimum 100 ratings):")
    
    # Calculate average rating per movie
    movie_ratings = ratings_df.groupby('movieId').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    
    movie_ratings.columns = ['movieId', 'avg_rating', 'num_ratings']
    
    # Filter movies with at least 100 ratings
    popular_movies = movie_ratings[movie_ratings['num_ratings'] >= 100]
    
    # Merge with movie info
    popular_movies = popular_movies.merge(
        movies_df[['movieId', 'title', 'genres']],
        on='movieId'
    )
    
    # Sort by average rating
    top_movies = popular_movies.nlargest(10, 'avg_rating')
    
    # Display as table
    display_df = top_movies[['title', 'genres', 'avg_rating', 'num_ratings']].copy()
    display_df.columns = ['Title', 'Genres', 'Avg Rating', '# Ratings']
    display_df['Avg Rating'] = display_df['Avg Rating'].round(2)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    # User engagement
    st.markdown("## 👥 User Engagement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Ratings per User")
        
        user_rating_counts = ratings_df.groupby('userId').size()
        
        fig_user_engagement = px.histogram(
            user_rating_counts,
            nbins=50,
            labels={'value': 'Number of Ratings', 'count': 'Number of Users'},
            title='Distribution of user activity'
        )
        
        fig_user_engagement.update_layout(
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_user_engagement, use_container_width=True)
    
    with col2:
        st.markdown("### Engagement Statistics")
        
        st.metric("Most Active User", f"{user_rating_counts.max()} ratings")
        st.metric("Average per User", f"{user_rating_counts.mean():.0f} ratings")
        st.metric("Median per User", f"{user_rating_counts.median():.0f} ratings")
        
        # Power users
        power_users = (user_rating_counts >= 500).sum()
        st.metric("Power Users (500+ ratings)", f"{power_users:,}")
        
        st.info("""
        **Insight**: A core group of engaged users provides the majority of ratings,
        helping create a rich collaborative filtering dataset.
        """)
    
    st.markdown("---")
    
    # Project Information
    st.markdown("## 🎓 About This Project")
    
    st.markdown("""
    ### CineMatch V1.0.0
    
    This is a production-grade movie recommendation engine built as a master's thesis demonstration.
    
    **Key Technologies:**
    - **Algorithm**: SVD (Singular Value Decomposition) for collaborative filtering
    - **Dataset**: MovieLens 32M (32 million ratings)
    - **Framework**: Streamlit for interactive web interface
    - **Deployment**: Docker for containerized deployment
    
    **Features:**
    - ✅ Personalized movie recommendations
    - ✅ Explainable AI (understand why movies are recommended)
    - ✅ User taste profiling
    - ✅ "Surprise Me" serendipity mode
    - ✅ Movie similarity explorer
    
    **Performance Goal:**
    - Target RMSE: < 0.87 on test set
    - Response Time: < 2 seconds for recommendations
    
    ---
    
    ### 🚀 Get Started
    
    Navigate to the **🎬 Recommend** page to get personalized movie recommendations!
    
    Enter any User ID from 1 to {max_user} to see recommendations tailored to that user's taste.
    """.format(max_user=ratings_df['userId'].max()))
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>CineMatch V1.0.0</strong> | Built with ❤️ for Master's Thesis Defense</p>
        <p>Powered by Streamlit, Surprise, and MovieLens 32M Dataset</p>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.info("""
    **Possible Solutions:**
    1. Ensure the MovieLens dataset is downloaded and placed in `data/ml-32m/`
    2. Check that all required CSV files are present
    3. See README.md for setup instructions
    """)
