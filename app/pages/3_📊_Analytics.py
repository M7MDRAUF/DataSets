"""
CineMatch V1.0.0 - Analytics Page

Data visualizations and insights.
Implements F-09 (Data Visualization) and F-10 (Movie Similarity Explorer).

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
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_processing import load_movies, load_ratings
# Try to import sklearn-compatible engine first (Windows compatibility)
try:
    from src.recommendation_engine_sklearn import RecommendationEngine
except ImportError:
    from src.recommendation_engine import RecommendationEngine
from src.utils import extract_year_from_title, create_genre_color_map


# Page config
st.set_page_config(
    page_title="CineMatch - Analytics",
    page_icon="📊",
    layout="wide"
)

# Header
st.title("📊 Dataset Analytics & Insights")
st.markdown("### Explore the MovieLens 32M dataset through interactive visualizations")
st.markdown("---")

# Load data (cached)
@st.cache_data
def load_data_cached():
    """Load and cache dataset"""
    movies_df = load_movies()
    ratings_df = load_ratings(sample_size=2_000_000)  # 2M sample for analytics
    return movies_df, ratings_df

@st.cache_resource
def initialize_engine():
    """Initialize recommendation engine for similarity"""
    try:
        engine = RecommendationEngine()
        engine.load_model()
        engine.load_data()
        return engine
    except:
        return None

try:
    with st.spinner("Loading dataset..."):
        movies_df, ratings_df = load_data_cached()
        engine = initialize_engine()
    
    st.success("✅ Dataset loaded successfully")
    
    # Tab navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Genre Analysis",
        "📅 Temporal Trends",
        "🔥 Popularity Metrics",
        "🔍 Movie Similarity Explorer"
    ])
    
    # TAB 1: Genre Analysis
    with tab1:
        st.markdown("## 🎭 Genre Distribution & Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Genre Frequency")
            
            # Process genres
            genres_exploded = movies_df['genres'].str.split('|', expand=True).stack()
            genre_counts = genres_exploded.value_counts().head(20)
            
            # Create bar chart
            fig_genres = px.bar(
                x=genre_counts.index,
                y=genre_counts.values,
                labels={'x': 'Genre', 'y': 'Number of Movies'},
                title='Top 20 Genres by Movie Count',
                color=genre_counts.values,
                color_continuous_scale='Viridis'
            )
            
            fig_genres.update_layout(
                height=500,
                showlegend=False,
                xaxis={'tickangle': -45}
            )
            
            st.plotly_chart(fig_genres, use_container_width=True)
        
        with col2:
            st.markdown("### Genre Combinations")
            
            # Most common genre combinations
            genre_combo_counts = movies_df['genres'].value_counts().head(15)
            
            fig_combos = px.bar(
                y=genre_combo_counts.index,
                x=genre_combo_counts.values,
                orientation='h',
                labels={'x': 'Count', 'y': 'Genre Combination'},
                title='Most Common Genre Combinations',
                color=genre_combo_counts.values,
                color_continuous_scale='Plasma'
            )
            
            fig_combos.update_layout(
                height=500,
                showlegend=False,
                yaxis={'tickmode': 'linear'}
            )
            
            st.plotly_chart(fig_combos, use_container_width=True)
        
        # Genre ratings analysis
        st.markdown("### 📊 Average Ratings by Genre")
        
        # Merge ratings with movies
        merged = ratings_df.merge(movies_df[['movieId', 'genres']], on='movieId')
        
        # Explode genres
        merged_exploded = merged.assign(
            genre=merged['genres'].str.split('|')
        ).explode('genre')
        
        # Calculate average rating per genre
        genre_ratings = merged_exploded.groupby('genre')['rating'].agg(['mean', 'count']).reset_index()
        genre_ratings = genre_ratings[genre_ratings['count'] >= 1000]  # Filter for significance
        genre_ratings = genre_ratings.sort_values('mean', ascending=False)
        
        # Create plot
        fig_genre_ratings = px.bar(
            genre_ratings,
            x='genre',
            y='mean',
            labels={'mean': 'Average Rating', 'genre': 'Genre'},
            title='Average Rating by Genre (min 1000 ratings)',
            color='mean',
            color_continuous_scale='RdYlGn',
            range_color=[3.0, 4.5]
        )
        
        fig_genre_ratings.update_layout(height=400, xaxis={'tickangle': -45})
        st.plotly_chart(fig_genre_ratings, use_container_width=True)
    
    # TAB 2: Temporal Trends
    with tab2:
        st.markdown("## 📅 Movie Release Trends Over Time")
        
        # Extract years from titles
        movies_df['year'] = movies_df['title'].apply(extract_year_from_title)
        movies_with_years = movies_df.dropna(subset=['year'])
        movies_with_years['year'] = movies_with_years['year'].astype(int)
        
        # Filter reasonable years
        movies_with_years = movies_with_years[
            (movies_with_years['year'] >= 1900) &
            (movies_with_years['year'] <= 2025)
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Movies Released by Year")
            
            year_counts = movies_with_years['year'].value_counts().sort_index()
            
            fig_years = px.line(
                x=year_counts.index,
                y=year_counts.values,
                labels={'x': 'Year', 'y': 'Number of Movies'},
                title='Movie Production Over Time'
            )
            
            fig_years.update_layout(height=400)
            st.plotly_chart(fig_years, use_container_width=True)
        
        with col2:
            st.markdown("### Movies by Decade")
            
            # Group by decade
            movies_with_years['decade'] = (movies_with_years['year'] // 10) * 10
            decade_counts = movies_with_years['decade'].value_counts().sort_index()
            
            fig_decades = px.bar(
                x=decade_counts.index,
                y=decade_counts.values,
                labels={'x': 'Decade', 'y': 'Number of Movies'},
                title='Movie Production by Decade',
                color=decade_counts.values,
                color_continuous_scale='Blues'
            )
            
            fig_decades.update_layout(height=400)
            st.plotly_chart(fig_decades, use_container_width=True)
        
        # Rating trends over time
        st.markdown("### ⭐ Rating Trends Over Time")
        
        # Merge with ratings
        ratings_with_years = ratings_df.merge(
            movies_with_years[['movieId', 'year']],
            on='movieId'
        )
        
        # Group by year
        yearly_ratings = ratings_with_years.groupby('year')['rating'].agg(['mean', 'count']).reset_index()
        yearly_ratings = yearly_ratings[yearly_ratings['count'] >= 100]  # Filter for significance
        
        fig_rating_trends = px.scatter(
            yearly_ratings,
            x='year',
            y='mean',
            size='count',
            labels={'mean': 'Average Rating', 'year': 'Release Year', 'count': 'Number of Ratings'},
            title='Average Movie Ratings by Release Year',
            color='mean',
            color_continuous_scale='RdYlGn'
        )
        
        fig_rating_trends.update_layout(height=450)
        st.plotly_chart(fig_rating_trends, use_container_width=True)
        
        st.info("""
        **Insight**: Older movies tend to have slightly higher ratings, possibly due to 
        survivorship bias (only the best old movies remain popular).
        """)
    
    # TAB 3: Popularity Metrics
    with tab3:
        st.markdown("## 🔥 Movie Popularity Analysis")
        
        # Calculate movie statistics
        movie_stats = ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count', 'std']
        }).reset_index()
        
        movie_stats.columns = ['movieId', 'avg_rating', 'num_ratings', 'std_rating']
        
        # Merge with movie info
        movie_stats = movie_stats.merge(
            movies_df[['movieId', 'title', 'genres']],
            on='movieId'
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏆 Most Rated Movies")
            
            most_rated = movie_stats.nlargest(20, 'num_ratings')
            
            fig_most_rated = px.bar(
                most_rated,
                x='num_ratings',
                y='title',
                orientation='h',
                labels={'num_ratings': 'Number of Ratings', 'title': 'Movie'},
                title='Top 20 Most Rated Movies',
                color='avg_rating',
                color_continuous_scale='Viridis'
            )
            
            fig_most_rated.update_layout(
                height=600,
                yaxis={'tickmode': 'linear'}
            )
            
            st.plotly_chart(fig_most_rated, use_container_width=True)
        
        with col2:
            st.markdown("### ⭐ Highest Rated Movies (100+ ratings)")
            
            popular_and_good = movie_stats[movie_stats['num_ratings'] >= 100]
            highest_rated = popular_and_good.nlargest(20, 'avg_rating')
            
            fig_highest_rated = px.bar(
                highest_rated,
                x='avg_rating',
                y='title',
                orientation='h',
                labels={'avg_rating': 'Average Rating', 'title': 'Movie'},
                title='Top 20 Highest Rated Movies',
                color='num_ratings',
                color_continuous_scale='Oranges'
            )
            
            fig_highest_rated.update_layout(
                height=600,
                yaxis={'tickmode': 'linear'}
            )
            
            st.plotly_chart(fig_highest_rated, use_container_width=True)
        
        # Popularity vs Quality scatter
        st.markdown("### 📊 Popularity vs Quality")
        
        # Filter for visualization
        viz_data = movie_stats[movie_stats['num_ratings'] >= 10]
        
        fig_scatter = px.scatter(
            viz_data,
            x='num_ratings',
            y='avg_rating',
            hover_data=['title'],
            labels={'num_ratings': 'Number of Ratings (Popularity)', 'avg_rating': 'Average Rating (Quality)'},
            title='Movie Popularity vs Quality',
            color='avg_rating',
            color_continuous_scale='Viridis',
            opacity=0.6
        )
        
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # TAB 4: Movie Similarity Explorer
    with tab4:
        st.markdown("## 🔍 Movie Similarity Explorer")
        st.markdown("### F-10: Find movies similar to one you love")
        
        if engine is None:
            st.warning("⚠️ Model not loaded. Please train the model first.")
        else:
            # Movie search
            st.markdown("#### Search for a Movie")
            
            search_query = st.text_input(
                "Enter movie title (partial match works)",
                placeholder="e.g., Matrix, Inception, Shawshank"
            )
            
            if search_query:
                # Search for movies
                search_results = movies_df[
                    movies_df['title'].str.contains(search_query, case=False, na=False)
                ].head(10)
                
                if len(search_results) > 0:
                    st.markdown(f"Found {len(search_results)} matching movies:")
                    
                    # Display results
                    for idx, row in search_results.iterrows():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**{row['title']}** - {row['genres']}")
                        
                        with col2:
                            if st.button("Find Similar", key=f"sim_{row['movieId']}"):
                                # Find similar movies
                                try:
                                    with st.spinner("Finding similar movies..."):
                                        similar = engine.get_similar_movies(row['movieId'], n=10)
                                    
                                    st.markdown(f"### Movies Similar to **{row['title']}**")
                                    
                                    # Display similar movies
                                    for sim_idx, sim_row in similar.iterrows():
                                        st.markdown(f"""
                                        **{sim_idx+1}.** {sim_row['title']}  
                                        *Similarity Score*: {sim_row['similarity']:.3f} | *Genres*: {sim_row['genres']}
                                        """)
                                    
                                except Exception as e:
                                    st.error(f"Error finding similar movies: {e}")
                else:
                    st.warning("No movies found. Try a different search term.")
            
            st.markdown("---")
            
            st.info("""
            ### 💡 How it Works
            
            Movie similarity is calculated using the **latent factor vectors** learned by the SVD model.
            
            - Movies with similar factor patterns are considered similar
            - This captures both content similarity (genre) and collaborative patterns (what users watch together)
            - Higher similarity scores indicate stronger relationships
            
            **Try it**: Search for a movie you love and discover similar films!
            """)

except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.info("""
    **Possible Solutions:**
    1. Ensure the MovieLens dataset is in `data/ml-32m/`
    2. Check that all CSV files are present
    3. See README.md for setup instructions
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>Analytics powered by Plotly & Pandas</strong></p>
    <p>Exploring 32 million ratings across thousands of movies</p>
</div>
""", unsafe_allow_html=True)
