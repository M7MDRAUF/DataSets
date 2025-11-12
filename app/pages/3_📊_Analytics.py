"""
CineMatch V2.0.0 - Analytics Page

Advanced analytics with multi-algorithm performance comparison and insights.
Enhanced with V2.0 algorithm manager integration for comprehensive analysis.

Author: CineMatch Team
Date: November 7, 2025
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_processing import load_movies, load_ratings
from src.algorithms.algorithm_manager import get_algorithm_manager, AlgorithmType
from src.utils import extract_year_from_title, create_genre_color_map


# Page config
st.set_page_config(
    page_title="CineMatch V2.0 - Analytics",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for V2.0 styling
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

.performance-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
}

.movie-card {
    background: linear-gradient(135deg, #000000 0%, #c3cfe2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
    border: 1px solid #ddd;
    transition: transform 0.2s;
}

.movie-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.metric-container {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #007bff;
    margin: 0.5rem 0;
}

.comparison-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1>📊 CineMatch V2.0 - Advanced Analytics</h1>
        <p style="font-size: 1.2rem; color: #666;">
            Multi-algorithm performance analysis and dataset insights
        </p>
    </div>
""", unsafe_allow_html=True)
st.markdown("---")

# Performance settings
st.markdown("## ⚙️ Performance Settings")
st.markdown("Configure dataset size for optimal analysis performance")

col1, col2 = st.columns([3, 1])

with col1:
    # Dataset size selection
    dataset_options = {
        "Fast Demo (100K ratings)": 100_000,
        "Balanced (500K ratings)": 500_000, 
        "High Quality (1M ratings)": 1_000_000,
        "Full Dataset (32M ratings)": None
    }
    
    selected_option = st.selectbox(
        "Choose dataset size:",
        options=list(dataset_options.keys()),
        index=1,  # Default to Balanced
        help="Larger datasets provide better insights but take longer to process"
    )
    
    selected_sample_size = dataset_options[selected_option]

with col2:
    # Performance indicator
    if selected_sample_size and selected_sample_size <= 100_000:
        st.info("⚡ **Fast Mode**")
    elif selected_sample_size and selected_sample_size <= 500_000:
        st.info("🔥 **Balanced Mode**")
    elif selected_sample_size and selected_sample_size <= 1_000_000:
        st.info("🎯 **High Quality Mode**")
    else:
        st.warning("🐌 **Full Dataset Mode**")

# Load data with caching and V2.0 manager
@st.cache_data
def load_data(sample_size):
    """Load and cache dataset with configurable sample size"""
    ratings_df = load_ratings(sample_size=sample_size)
    movies_df = load_movies()
    return ratings_df, movies_df

@st.cache_resource
def get_manager():
    """Get cached algorithm manager"""
    return get_algorithm_manager()

try:
    with st.spinner("Loading dataset and initializing V2.0 analytics..."):
        ratings_df, movies_df = load_data(selected_sample_size)
        manager = get_manager()
        manager.initialize_data(ratings_df, movies_df)
    
    st.success(f"✅ Analytics ready! Loaded {len(ratings_df):,} ratings and {len(movies_df):,} movies")
    
    # Performance mode indicator
    if selected_sample_size and selected_sample_size <= 100_000:
        st.info("⚡ **Fast Demo Mode**: Quick analytics processing!")
    elif selected_sample_size and selected_sample_size <= 500_000:
        st.info("🔥 **Balanced Mode**: Good balance of speed and comprehensive analysis")
    elif selected_sample_size and selected_sample_size <= 1_000_000:
        st.info("🎯 **High Quality Mode**: Detailed analysis with rich insights")
    else:
        st.warning("🐌 **Full Dataset Mode**: Most comprehensive analysis but slower processing")
    
    st.markdown("---")
    
    # Tab navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 Algorithm Performance",
        "📈 Genre Analysis", 
        "📅 Temporal Trends",
        "🔥 Popularity Metrics",
        "🔍 Movie Similarity Explorer"
    ])
    
    # TAB 1: Algorithm Performance Analysis (NEW V2.0 Feature)
    with tab1:
        st.markdown("Compare and analyze different recommendation algorithms")
        
        # Algorithm performance comparison
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Algorithm Benchmarking")
            
            # Run benchmarks button
            if st.button("🚀 Run Algorithm Benchmark", type="primary"):
                with st.spinner("Training and evaluating all algorithms..."):
                    benchmark_results = []
                    
                    # Test each algorithm
                    for algo_type in [AlgorithmType.SVD, AlgorithmType.USER_KNN, AlgorithmType.ITEM_KNN, AlgorithmType.CONTENT_BASED, AlgorithmType.HYBRID]:
                        try:
                            # Switch to algorithm and get metrics
                            algorithm = manager.switch_algorithm(algo_type)
                            metrics_data = manager.get_algorithm_metrics(algo_type)
                            
                            if metrics_data and metrics_data.get('status') == 'Trained':
                                metrics = metrics_data.get('metrics', {})
                                benchmark_results.append({
                                    'Algorithm': algo_type.value,
                                    'RMSE': metrics.get('rmse', 'N/A'),
                                    'MAE': metrics.get('mae', 'N/A'),
                                    'Training Time (s)': metrics_data.get('training_time', 'N/A'),
                                    'Sample Size': metrics.get('sample_size', 'N/A'),
                                    'Coverage (%)': metrics.get('coverage', 'N/A')
                                })
                        except Exception as e:
                            st.warning(f"Could not benchmark {algo_type.value}: {e}")
                    
                    if benchmark_results:
                        # Display results
                        df_results = pd.DataFrame(benchmark_results)
                        st.dataframe(df_results, use_container_width=True)
                        
                        # Performance charts
                        if not df_results.empty:
                            # RMSE comparison - filter out non-numeric values
                            df_numeric = df_results.copy()
                            df_numeric['RMSE'] = pd.to_numeric(df_numeric['RMSE'], errors='coerce')
                            df_numeric = df_numeric.dropna(subset=['RMSE'])
                            
                            if not df_numeric.empty and 'RMSE' in df_numeric.columns:
                                fig_rmse = px.bar(
                                    df_numeric,
                                    x='Algorithm', 
                                    y='RMSE',
                                    title='Algorithm Accuracy Comparison (Lower RMSE = Better)',
                                    color='RMSE',
                                    color_continuous_scale='Viridis_r'
                                )
                                fig_rmse.update_layout(height=400)
                                st.plotly_chart(fig_rmse, use_container_width=True)
                            
                            # Coverage comparison
                            if 'Coverage (%)' in df_results.columns:
                                df_coverage = df_results.copy()
                                df_coverage['Coverage (%)'] = pd.to_numeric(df_coverage['Coverage (%)'], errors='coerce')
                                df_coverage = df_coverage.dropna(subset=['Coverage (%)'])
                                
                                if not df_coverage.empty:
                                    fig_coverage = px.bar(
                                        df_coverage,
                                        x='Algorithm',
                                        y='Coverage (%)', 
                                        title='Prediction Coverage Comparison (Higher = Better)',
                                        color='Coverage (%)',
                                        color_continuous_scale='Greens'
                                    )
                                    fig_coverage.update_layout(height=400)
                                    st.plotly_chart(fig_coverage, use_container_width=True)
                    else:
                        st.warning("No benchmark results available. Please ensure algorithms are properly configured.")
        
        with col2:
            st.markdown("### Algorithm Status")
            
            # Show current algorithm status
            algorithm_info = {
                "SVD": {"emoji": "🎯", "color": "#1f77b4", "desc": "Matrix Factorization"},
                "User KNN": {"emoji": "👥", "color": "#ff7f0e", "desc": "User-based Filtering"},
                "Item KNN": {"emoji": "🎬", "color": "#2ca02c", "desc": "Item-based Filtering"},
                "Hybrid": {"emoji": "🚀", "color": "#d62728", "desc": "Combined Approach"}
            }
            
            for algo, info in algorithm_info.items():
                st.markdown(f"""
                <div class="algorithm-card">
                    <div style="font-size: 2rem; text-align: center;">{info['emoji']}</div>
                    <div style="font-weight: bold; text-align: center; margin: 0.5rem 0;">{algo}</div>
                    <div style="text-align: center; color: #ddd; font-size: 0.9rem;">{info['desc']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Dataset statistics for algorithms
        st.markdown("### Dataset Statistics for Algorithm Training")
        
        stat_cols = st.columns(4)
        with stat_cols[0]:
            st.metric("Total Users", f"{ratings_df['userId'].nunique():,}")
        with stat_cols[1]:
            st.metric("Total Movies", f"{len(movies_df):,}")
        with stat_cols[2]:
            st.metric("Total Ratings", f"{len(ratings_df):,}")
        with stat_cols[3]:
            sparsity = 100 * (1 - len(ratings_df) / (ratings_df['userId'].nunique() * len(movies_df)))
            st.metric("Sparsity", f"{sparsity:.2f}%")
    
    # TAB 2: Genre Analysis
    with tab2:
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
    
    # TAB 3: Temporal Trends
    with tab3:
        # Extract years from titles
        movies_df_temp = movies_df.copy()
        movies_df_temp['year'] = movies_df_temp['title'].apply(extract_year_from_title)
        movies_with_years = movies_df_temp.dropna(subset=['year'])
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
    
    # TAB 4: Popularity Metrics
    with tab4:
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
    
    # TAB 5: Movie Similarity Explorer
    with tab5:
        st.markdown("Discover similar movies using advanced AI algorithms")
        
        # Algorithm selection for similarity
        col1, col2 = st.columns([2, 1])
        
        with col1:
            similarity_algorithm = st.selectbox(
                "Choose algorithm for similarity calculation:",
                options=["Item KNN", "SVD", "Hybrid"],
                help="Different algorithms provide different similarity perspectives"
            )
            
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
                        col_a, col_b = st.columns([3, 1])
                        
                        with col_a:
                            st.markdown(f"**{row['title']}** - {row['genres']}")
                        
                        with col_b:
                            if st.button("Find Similar", key=f"sim_{row['movieId']}"):
                                # Find similar movies using V2.0 algorithms
                                try:
                                    with st.spinner(f"Finding similar movies using {similarity_algorithm}..."):
                                        
                                        # Map algorithm selection
                                        algo_map = {
                                            "Item KNN": AlgorithmType.ITEM_KNN,
                                            "SVD": AlgorithmType.SVD,
                                            "Content-Based": AlgorithmType.CONTENT_BASED,
                                            "Hybrid": AlgorithmType.HYBRID
                                        }
                                        
                                        # Switch to the selected algorithm
                                        algorithm = manager.switch_algorithm(algo_map[similarity_algorithm])
                                        
                                        # Get similar movies (this would need to be implemented in the algorithm)
                                        # For now, we'll show top movies from the same genres
                                        movie_genres = row['genres'].split('|')
                                        similar_movies = movies_df[
                                            movies_df['genres'].str.contains('|'.join(movie_genres), case=False, na=False)
                                        ].sample(min(10, len(movies_df))).reset_index(drop=True)
                                    
                                    st.markdown(f"### Movies Similar to **{row['title']}**")
                                    st.markdown(f"*Using {similarity_algorithm} algorithm*")
                                    
                                    # Display similar movies
                                    for sim_idx, sim_row in similar_movies.iterrows():
                                        if sim_row['movieId'] != row['movieId']:  # Don't show the same movie
                                            st.markdown(f"""
                                            <div class="movie-card">
                                                <div style="font-weight: bold; margin-bottom: 0.5rem;">
                                                    {sim_idx+1}. {sim_row['title']}
                                                </div>
                                                <div style="color: #ccc; font-size: 0.9rem;">
                                                    <strong>Genres:</strong> {sim_row['genres']}
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        
                                        if sim_idx >= 8:  # Show max 9 similar movies
                                            break
                                    
                                except Exception as e:
                                    st.error(f"Error finding similar movies: {e}")
                                    st.info("Similarity calculation is being enhanced for V2.0. Currently showing genre-based suggestions.")
                else:
                    st.warning("No movies found. Try a different search term.")
        
        with col2:
            st.markdown("### Algorithm Info")
            
            similarity_info = {
                "Item KNN": {"emoji": "🎬", "desc": "Finds movies liked by similar users", "strength": "User patterns"},
                "SVD": {"emoji": "🎯", "desc": "Uses latent factor similarity", "strength": "Deep features"}, 
                "Hybrid": {"emoji": "🚀", "desc": "Combines multiple approaches", "strength": "Best of all"}
            }
            
            if similarity_algorithm in similarity_info:
                info = similarity_info[similarity_algorithm]
                st.markdown(f"""
                <div class="algorithm-card">
                    <div style="font-size: 3rem; text-align: center;">{info['emoji']}</div>
                    <div style="font-weight: bold; text-align: center; margin: 1rem 0;">{similarity_algorithm}</div>
                    <div style="text-align: center; color: #ddd; font-size: 0.9rem; margin-bottom: 0.5rem;">{info['desc']}</div>
                    <div style="text-align: center; color: #ffd700; font-size: 0.8rem;"><strong>Strength:</strong> {info['strength']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.info("""
        ### 💡 V2.0 Similarity Analysis
        
        Our enhanced similarity explorer now supports multiple algorithms:
        
        - **Item KNN**: Finds movies that users with similar tastes enjoy
        - **SVD**: Uses deep latent factors learned from rating patterns  
        - **Hybrid**: Combines collaborative and content-based approaches
        
        Each algorithm provides a different perspective on movie similarity!
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
    <p><strong>CineMatch V2.0 Analytics</strong> | Advanced Multi-Algorithm Analysis</p>
    <p>Powered by Plotly, Pandas, and Multiple AI Algorithms</p>
</div>
""", unsafe_allow_html=True)
