"""
CineMatch V2.0 - Enhanced Home Page

Overview, statistics, and multi-algorithm recommendation system integration.
Now supports SVD, User KNN, Item KNN, and Hybrid algorithms.

Author: CineMatch Development Team  
Date: November 7, 2025
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.algorithms.algorithm_manager import get_algorithm_manager, AlgorithmType
from src.data_processing import load_movies, load_ratings
from src.utils import (
    format_genres,
    create_rating_stars,
    get_genre_emoji
)


# Page config
st.set_page_config(
    page_title="CineMatch V2.0 - Home",
    page_icon="🏠",
    layout="wide"
)

# V2.0 CSS Styling
st.markdown("""
<style>
.algorithm-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
    border-left: 4px solid #E50914;
}

.metrics-container {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.movie-card {
    background: linear-gradient(135deg, #000000 0%, #c3cfe2 100%);
    border-radius: 10px;
    padding: 1.5rem;
    margin: 1rem 0;
    border-left: 5px solid #E50914;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    color: white;
}

.recommendation-header {
    background: linear-gradient(90deg, #E50914, #8B0000);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}

.performance-card {
    background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🏠 CineMatch V2.0 - Home")
st.markdown("### Multi-Algorithm Movie Recommendation Engine")
st.markdown("---")

# Algorithm Manager Integration
@st.cache_resource
def get_manager():
    """Get the algorithm manager singleton"""
    return get_algorithm_manager()

# Load data with configurable sampling (V2.0 feature)
@st.cache_data
def load_data(sample_size):
    """Load and cache the dataset with configurable sampling"""
    print("Loading MovieLens dataset...")
    ratings_df = load_ratings(sample_size=sample_size)
    movies_df = load_movies()
    return ratings_df, movies_df

# V2.0 Performance Settings and Algorithm Selection
st.markdown("## ⚙️ Algorithm & Performance Settings")

with st.expander("🚀 Dataset Size Configuration", expanded=True):
    st.markdown("### 📊 Choose Dataset Size for Optimal Performance")
    
    dataset_options = [
        ("Fast Demo", 100000),
        ("Balanced", 500000), 
        ("High Quality", 1000000),
        ("Full Dataset", None)
    ]
    
    dataset_size = st.selectbox(
        "Choose dataset size for optimal performance:",
        options=dataset_options,
        index=3,  # Default to Full Dataset for pre-trained models
        format_func=lambda x: f"{x[0]} ({x[1]:,} ratings)" if x[1] else f"{x[0]} (32M ratings)"
    )
    
    selected_sample_size = dataset_size[1]
    
    # Check if dataset size changed and clear cache if needed
    if 'last_dataset_size' in st.session_state and st.session_state.last_dataset_size != selected_sample_size:
        st.cache_data.clear()
        st.info(f"🔄 Dataset changed to {dataset_size[0]}. Cache cleared and algorithms reset.")
    
    st.session_state.last_dataset_size = selected_sample_size
    
    st.info(f"""
    **Selected**: {dataset_size[0]}
    - **Training Speed**: {'⚡ Very Fast' if selected_sample_size and selected_sample_size <= 100000 else '🔥 Fast' if selected_sample_size and selected_sample_size <= 500000 else '⏱️ Medium' if selected_sample_size and selected_sample_size <= 1000000 else '🐌 Slow'}
    - **Accuracy**: {'📊 Good' if selected_sample_size and selected_sample_size <= 500000 else '🎯 High' if selected_sample_size and selected_sample_size <= 1000000 else '🏆 Maximum'}
    - **Memory Usage**: {'💚 Low' if selected_sample_size and selected_sample_size <= 500000 else '🟡 Medium' if selected_sample_size and selected_sample_size <= 1000000 else '🔴 High'}
    """)

try:
    with st.spinner(f"🚀 Loading MovieLens dataset... ({dataset_size[0]} mode)"):
        st.markdown("Running load_data(...).")
        ratings_df, movies_df = load_data(selected_sample_size)
    
    # Initialize algorithm manager
    manager = get_manager()
    
    # Always reinitialize with current dataset (fixes dataset size mismatch)
    manager.initialize_data(ratings_df, movies_df)
    
    st.success(f"✅ System ready! Loaded {len(ratings_df):,} ratings and {len(movies_df):,} movies")
    
    # Performance mode indicator
    if selected_sample_size and selected_sample_size <= 100000:
        st.info("⚡ **Fast Demo Mode**: Algorithms will train in seconds!")
    elif selected_sample_size and selected_sample_size <= 500000:
        st.info("🔥 **Balanced Mode**: Good balance of speed and accuracy")
    elif selected_sample_size and selected_sample_size <= 1000000:
        st.info("🎯 **High Quality Mode**: Excellent accuracy with reasonable speed")
    else:
        st.warning("🐌 **Full Dataset Mode**: Maximum accuracy but slow training times")
    
    st.markdown("---")
    
    # Algorithm Selection & Recommendations
    st.markdown("## 🤖 AI Recommendation Engine")
    st.markdown("Experience our advanced multi-algorithm recommendation system")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Choose Your Algorithm")
        
        # Algorithm selection
        algorithm_options = {
            "SVD": "🎯 SVD (Singular Value Decomposition) - Best overall accuracy",
            "User KNN": "👥 User-Based Collaborative Filtering - Find similar users",
            "Item KNN": "🎬 Item-Based Collaborative Filtering - Find similar movies",
            "Hybrid": "🚀 Hybrid Algorithm - Combines multiple approaches"
        }
        
        selected_algorithm = st.selectbox(
            "Select recommendation algorithm:",
            options=list(algorithm_options.keys()),
            format_func=lambda x: algorithm_options[x],
            help="Each algorithm uses different approaches to generate recommendations"
        )
        
        # User ID input
        st.markdown("### Enter User ID")
        max_user_id = ratings_df['userId'].max()
        min_user_id = ratings_df['userId'].min()
        
        # Use a safe default value that's within the valid range
        default_user_id = max(int(min_user_id), 10)  # Prefer 10 if valid, otherwise use min
        
        user_id = st.number_input(
            f"Enter User ID ({min_user_id} - {max_user_id}):",
            min_value=int(min_user_id),
            max_value=int(max_user_id),
            value=default_user_id,
            help=f"Choose any user ID to see personalized recommendations"
        )
        
        # Number of recommendations
        num_recommendations = st.slider(
            "Number of recommendations:",
            min_value=5,
            max_value=20,
            value=10,
            help="How many movie recommendations to generate"
        )
    
    with col2:
        st.markdown("### Algorithm Performance")
        
        # Get algorithm manager and show current status
        manager = get_manager()
        manager.initialize_data(ratings_df, movies_df)
        
        # Show algorithm info cards
        algorithm_info = {
            "SVD": {"emoji": "🎯", "color": "#1f77b4", "strength": "Accuracy"},
            "User KNN": {"emoji": "👥", "color": "#ff7f0e", "strength": "Similarity"},
            "Item KNN": {"emoji": "🎬", "color": "#2ca02c", "strength": "Item Relations"},
            "Hybrid": {"emoji": "🚀", "color": "#d62728", "strength": "Combined Power"}
        }
        
        for algo, info in algorithm_info.items():
            is_selected = algo == selected_algorithm
            border_style = f"border: 3px solid {info['color']};" if is_selected else "border: 1px solid #333;"
            
            st.markdown(f"""
            <div class="algorithm-card" style="{border_style}">
                <div style="font-size: 2rem; text-align: center;">{info['emoji']}</div>
                <div style="font-weight: bold; text-align: center; margin: 0.5rem 0;">{algo}</div>
                <div style="text-align: center; color: #888; font-size: 0.9rem;">{info['strength']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Generate recommendations button
    if st.button("🎬 Generate Recommendations", type="primary", use_container_width=True):
        try:
            with st.spinner(f"Training {selected_algorithm} algorithm and generating recommendations..."):
                
                # Map selected algorithm to AlgorithmType
                algorithm_map = {
                    "SVD": AlgorithmType.SVD,
                    "User KNN": AlgorithmType.USER_KNN,
                    "Item KNN": AlgorithmType.ITEM_KNN,
                    "Hybrid": AlgorithmType.HYBRID
                }
                
                # Get the algorithm instance
                algorithm_type = algorithm_map[selected_algorithm]
                algorithm = manager.switch_algorithm(algorithm_type)
                
                # Generate recommendations
                recommendations = algorithm.get_recommendations(user_id, n=num_recommendations, exclude_rated=True)
                
                st.success(f"✅ Generated {len(recommendations)} recommendations using {selected_algorithm}!")
                
                # Display recommendations
                st.markdown(f"### 🎬 Recommendations for User {user_id}")
                st.markdown(f"*Powered by {selected_algorithm} algorithm*")
                
                # Show recommendations in a nice grid
                cols = st.columns(2)
                for idx, (_, movie) in enumerate(recommendations.iterrows()):
                    col = cols[idx % 2]
                    
                    with col:
                        st.markdown(f"""
                        <div class="movie-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <span style="font-weight: bold; font-size: 1.1rem;">#{idx + 1}</span>
                                <span style="color: #ffd700; font-size: 1.2rem;">⭐ {movie.get('predicted_rating', 'N/A')}</span>
                            </div>
                            <h4 style="margin: 0.5rem 0; color: white;">{movie['title']}</h4>
                            <p style="color: #ccc; font-size: 0.9rem; margin: 0.25rem 0;">
                                <strong>Genres:</strong> {movie['genres']}
                            </p>
                            {f'<p style="color: #ddd; font-size: 0.8rem; margin-top: 0.5rem;"><strong>Why recommended:</strong> {movie.get("explanation", "Based on your preferences")}</p>' if movie.get("explanation") else ''}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show algorithm performance metrics if available
                if hasattr(manager, 'get_algorithm_metrics'):
                    try:
                        metrics = manager.get_algorithm_metrics(algorithm_type)
                        if metrics:
                            st.markdown("### 📊 Algorithm Performance")
                            
                            perf_cols = st.columns(4)
                            with perf_cols[0]:
                                st.metric("RMSE", f"{metrics.rmse:.4f}" if hasattr(metrics, 'rmse') else "N/A")
                            with perf_cols[1]:
                                st.metric("MAE", f"{metrics.mae:.4f}" if hasattr(metrics, 'mae') else "N/A")
                            with perf_cols[2]:
                                st.metric("Training Time", f"{metrics.training_time:.2f}s" if hasattr(metrics, 'training_time') else "N/A")
                            with perf_cols[3]:
                                st.metric("Prediction Time", f"{metrics.prediction_time:.4f}s" if hasattr(metrics, 'prediction_time') else "N/A")
                    except Exception as e:
                        st.debug(f"Metrics display error: {e}")
                
        except Exception as e:
            st.error(f"❌ Error generating recommendations: {str(e)}")
            st.info("""
            **Possible Solutions:**
            1. Try a different algorithm
            2. Ensure the dataset is properly loaded
            3. Check if the user ID exists in the dataset
            """)
    
    st.markdown("---")
    
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
    ### CineMatch V2.0.0
    
    This is a production-grade movie recommendation engine built as a master's thesis demonstration.
    
    **Key Technologies:**
    - **Algorithms**: Multi-algorithm support (SVD, User KNN, Item KNN, Hybrid)
    - **Dataset**: MovieLens 32M (32 million ratings) with configurable sampling
    - **Framework**: Streamlit for interactive web interface
    - **Deployment**: Docker for containerized deployment
    
    **Performance Goals:**
    - Target RMSE: < 0.87 on test set
    - Response Time: < 2 seconds for recommendations
    - Multiple algorithms for different use cases
    
    ---
    
    ### 🚀 Get Started
    
    **Try the algorithm selection above** to get personalized movie recommendations with different AI approaches!
    
    Enter any User ID from 1 to {max_user} to see recommendations tailored to that user's taste.
    """.format(max_user=ratings_df['userId'].max()))
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>CineMatch V2.0.0</strong> | Built with ❤️ for Master's Thesis Defense</p>
        <p>Powered by Streamlit, Multiple AI Algorithms, and MovieLens 32M Dataset</p>
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
