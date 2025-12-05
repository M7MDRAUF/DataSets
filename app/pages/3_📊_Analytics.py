"""
CineMatch V2.1.0 - Analytics Page

Advanced analytics with multi-algorithm performance comparison and insights.
Features all 5 algorithms with optimized pre-trained model loading.

Author: CineMatch Team
Date: November 13, 2025
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
import html
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_processing import load_movies, load_ratings
from src.algorithms.algorithm_manager import get_algorithm_manager, AlgorithmType
from src.utils import extract_year_from_title, create_genre_color_map, get_tmdb_poster_url, PLACEHOLDER_POSTER


# Page config
st.set_page_config(
    page_title="CineMatch V2.1.0 - Analytics",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for V2.1.0 styling
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
        <h1>📊 CineMatch V2.1.0 - Advanced Analytics</h1>
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
        "Large (5M ratings)": 5_000_000
    }
    
    selected_option = st.selectbox(
        "Choose dataset size:",
        options=list(dataset_options.keys()),
        index=1,  # Default to Balanced
        help="Larger datasets provide better insights but take longer to process",
        key="analytics_dataset_size"
    )
    
    selected_sample_size = dataset_options[selected_option]

with col2:
    # Performance indicator
    if selected_sample_size <= 100_000:
        st.info("⚡ **Fast Mode**\n~5-10s load")
    elif selected_sample_size <= 500_000:
        st.info("🔥 **Balanced Mode**\n~15-30s load")
    elif selected_sample_size <= 1_000_000:
        st.info("🎯 **High Quality**\n~1-2min load")
    else:
        st.warning("⚠️ **Large Dataset**\n~5-10min load")

# Load data with caching and V2.1.0 manager
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
    # Initialize data once per session
    if 'analytics_data_loaded' not in st.session_state:
        with st.spinner("Loading dataset and initializing V2.1.0 analytics..."):
            ratings_df, movies_df = load_data(selected_sample_size)
            manager = get_manager()
            manager.initialize_data(ratings_df, movies_df)
            st.session_state.analytics_data_loaded = True
            st.session_state.ratings_df = ratings_df
            st.session_state.movies_df = movies_df
            st.session_state.manager = manager
    else:
        # Reuse from session state
        ratings_df = st.session_state.ratings_df
        movies_df = st.session_state.movies_df
        manager = st.session_state.manager
    
    st.success(f"✅ Analytics ready! Loaded {len(ratings_df):,} ratings and {len(movies_df):,} movies")
    
    # ========== GLOBAL SESSION STATE INITIALIZATION (CRITICAL) ==========
    # Initialize session state variables BEFORE tab rendering to ensure proper persistence
    # across Streamlit reruns. This prevents state loss when switching between tabs or
    # clicking buttons that trigger page refreshes.
    if 'benchmark_results' not in st.session_state:
        st.session_state.benchmark_results = None
    if 'recommendation_results' not in st.session_state:
        st.session_state.recommendation_results = None
    
    st.markdown("---")
    
    # Tab navigation - rendered fresh on every rerun (normal Streamlit behavior)
    # Note: Tabs may appear duplicated due to spinner text above, but they're actually single instances
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 Algorithm Performance",
        "📈 Genre Analysis", 
        "📅 Temporal Trends",
        "🔥 Popularity Metrics",
        "🔍 Movie Similarity Explorer"
    ])
    
    # TAB 1: Algorithm Performance Analysis (V2.1.0 - All 5 Algorithms)
    with tab1:
        # Algorithm performance comparison
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Run benchmarks button (removed redundant header - tab already says "Algorithm Performance")
            if st.button("🚀 Run Algorithm Benchmark", type="primary"):
                # Use status container to avoid spinner duplication
                status_container = st.empty()
                status_container.info("⏳ Loading algorithm metrics...")
                
                benchmark_results = []
                
                # Get metrics for each algorithm WITHOUT forcing training
                for algo_type in [AlgorithmType.SVD, AlgorithmType.USER_KNN, AlgorithmType.ITEM_KNN, AlgorithmType.CONTENT_BASED, AlgorithmType.HYBRID]:
                    try:
                        # Check if algorithm is already cached/trained
                        metrics_data = manager.get_algorithm_metrics(algo_type)
                        
                        # If not cached, try to load it (will use pre-trained if available)
                        if not metrics_data or metrics_data.get('status') != 'Trained':
                            status_container.info(f"⏳ Loading {algo_type.value}...")
                            algorithm = manager.get_algorithm(algo_type)
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
                
                # Clear status and show results
                status_container.empty()
                
                # Store results in session state
                if benchmark_results:
                    st.session_state.benchmark_results = benchmark_results
                else:
                    st.session_state.benchmark_results = None
        
        # Display benchmark results if available (persistent across reruns)
        if st.session_state.benchmark_results:
            # Display results
            df_results = pd.DataFrame(st.session_state.benchmark_results)
            st.dataframe(df_results, width="stretch")
            
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
                    st.plotly_chart(fig_rmse, width="stretch")
                
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
                        st.plotly_chart(fig_coverage, width="stretch")
        
        # ========== SECTION BELOW BENCHMARK: SAMPLE RECOMMENDATIONS ==========
        st.markdown("---")
        st.markdown("### 🎯 Test Algorithm with Sample Recommendations")
        st.markdown("Generate recommendations for a test user to see how each algorithm performs in practice")
        
        # Test user selection and algorithm picker
        demo_col1, demo_col2, demo_col3 = st.columns([1, 1, 1])
        
        with demo_col1:
            test_user_id = st.number_input(
                "Test User ID:",
                min_value=1,
                max_value=ratings_df['userId'].max(),
                value=1,
                help="Enter a user ID to generate sample recommendations"
            )
        
        with demo_col2:
            demo_algorithm = st.selectbox(
                "Algorithm to Test:",
                options=["SVD Matrix Factorization", "KNN User-Based", "KNN Item-Based", "Content-Based Filtering", "Hybrid (Best of All)"],
                help="Choose which algorithm to generate recommendations from"
            )
        
        with demo_col3:
            num_recs = st.number_input(
                "Number of Recommendations:",
                min_value=5,
                max_value=20,
                value=10,
                help="How many recommendations to generate"
            )
        
        # Generate recommendations button
        if st.button("🎬 Generate Sample Recommendations", type="primary"):
            # Map UI selection to AlgorithmType
            algo_map = {
                "SVD Matrix Factorization": AlgorithmType.SVD,
                "KNN User-Based": AlgorithmType.USER_KNN,
                "KNN Item-Based": AlgorithmType.ITEM_KNN,
                "Content-Based Filtering": AlgorithmType.CONTENT_BASED,
                "Hybrid (Best of All)": AlgorithmType.HYBRID
            }
            
            selected_algo_type = algo_map[demo_algorithm]
            
            # ========== SECTION 1: LOAD ALGORITHM & GENERATE RECOMMENDATIONS ==========
            # Use status container to avoid text appearing above tabs
            status_container = st.empty()
            
            try:
                # Load/switch to the algorithm
                status_container.info(f"⏳ Loading {demo_algorithm}...")
                algorithm = manager.switch_algorithm(selected_algo_type)
                print(f"[DEBUG] Algorithm loaded: {algorithm.name}")
                
                # Check if user exists in sampled dataset (not blocking - per Recommend.py pattern)
                user_exists_in_sample = test_user_id in ratings_df['userId'].values
                if not user_exists_in_sample:
                    status_container.empty()
                    st.warning(f"⚠️ User {test_user_id} not found in the current sample. Generating recommendations for new user profile.")
                    st.info(f"💡 Tip: The dataset is sampled for performance. Try a different sample size or user ID.")
                
                # Generate recommendations
                status_container.info(f"⏳ Generating {num_recs} recommendations for User {test_user_id}...")
                recommendations = algorithm.recommend(
                    user_id=test_user_id,
                    n=num_recs
                )
                print(f"[DEBUG] Recommendations generated: {len(recommendations) if recommendations is not None else 'None'}")
                
                # Clear status container
                status_container.empty()
                
                # ========== DEFENSIVE VALIDATION ==========
                if recommendations is None:
                    st.error(f"❌ Algorithm returned None. This should never happen.")
                    st.info("Please try a different algorithm or user ID.")
                    print(f"[ERROR] Algorithm returned None for user {test_user_id}")
                    st.session_state.recommendation_results = None
                    st.stop()
                
                if not isinstance(recommendations, pd.DataFrame):
                    st.error(f"❌ Algorithm returned {type(recommendations)} instead of DataFrame.")
                    st.info("Please report this bug to the development team.")
                    print(f"[ERROR] Algorithm returned wrong type: {type(recommendations)}")
                    st.session_state.recommendation_results = None
                    st.stop()
                
                if recommendations.empty:
                    st.warning(f"⚠️ No recommendations generated for User {test_user_id}.")
                    st.info("This user might have rated all available movies. Try a different user ID.")
                    print(f"[WARNING] Empty recommendations for user {test_user_id}")
                    st.session_state.recommendation_results = None
                    st.stop()
                
                # Check for required columns
                required_cols = ['movieId', 'predicted_rating', 'title', 'genres']
                missing_cols = [col for col in required_cols if col not in recommendations.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                    st.info(f"Available columns: {list(recommendations.columns)}")
                    print(f"[ERROR] Missing columns: {missing_cols}, Available: {list(recommendations.columns)}")
                    st.session_state.recommendation_results = None
                    st.stop()
                
                # Store in session state
                st.session_state.recommendation_results = {
                    'recommendations': recommendations,
                    'algorithm': demo_algorithm,
                    'user_id': test_user_id
                }
                
            except Exception as e:
                status_container.empty()  # Clear loading message on error
                st.error(f"❌ Error generating recommendations: {e}")
                st.info("""
                **Troubleshooting:**
                1. Try a different user ID
                2. Try a different algorithm
                3. Ensure pre-trained models are available in `models/` folder
                4. Check Docker logs for detailed error information
                """)
                import traceback
                print(f"[ERROR] Recommendation generation failed: {traceback.format_exc()}")
                with st.expander("🐛 Show detailed error"):
                    st.code(traceback.format_exc())
                st.session_state.recommendation_results = None
        
        # Display recommendation results if available (persistent across reruns)
        if st.session_state.recommendation_results:
            recommendations = st.session_state.recommendation_results['recommendations']
            demo_algorithm = st.session_state.recommendation_results['algorithm']
            test_user_id = st.session_state.recommendation_results['user_id']
            
            # ========== DISPLAY RECOMMENDATIONS ==========
            st.success(f"✅ Generated {len(recommendations)} recommendations using {demo_algorithm}!")
            print(f"[SUCCESS] Displaying {len(recommendations)} recommendations")
            
            st.markdown(f"### 🎬 Top {len(recommendations)} Recommendations for User {test_user_id}")
            st.markdown(f"*Powered by {demo_algorithm}*")
            
            # Display each recommendation
            for idx, row in recommendations.iterrows():
                # Format genres with color badges
                genres_list = row['genres'].split('|')
                genre_badges = ' '.join([f"<span style='background-color: #667eea; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; margin-right: 4px;'>{html.escape(g)}</span>" for g in genres_list[:3]])
                
                # HTML escape title for safe rendering
                title_escaped = html.escape(str(row['title']))
                
                # Get poster URL
                poster_path = row.get('poster_path', None)
                poster_url = get_tmdb_poster_url(poster_path)
                
                st.markdown(f"""
                <div class="movie-card">
                    <div style="display: flex; gap: 15px; align-items: flex-start;">
                        <div style="flex-shrink: 0;">
                            <img src="{poster_url}" alt="{title_escaped}" 
                                 style="width: 80px; height: 120px; object-fit: cover; border-radius: 6px; 
                                        box-shadow: 0 2px 6px rgba(0,0,0,0.3);"
                                 onerror="this.src='{PLACEHOLDER_POSTER}'">
                        </div>
                        <div style="flex: 1;">
                            <div style="font-weight: bold; font-size: 1.1rem; margin-bottom: 0.5rem;">
                                {idx + 1}. {title_escaped}
                            </div>
                            <div style="margin-bottom: 0.5rem;">
                                {genre_badges}
                            </div>
                            <div style="color: #aaa; font-size: 0.9rem;">
                                <strong>Predicted Rating:</strong> ⭐ {row['predicted_rating']:.2f}/5.0
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # ========== SECTION 2: USER TASTE PROFILE (INDEPENDENT) ==========
            try:
                print(f"[DEBUG] Rendering User Taste Profile section...")
                
                # ========== USER TASTE PROFILE SECTION ==========
                st.markdown("---")
                st.markdown(f"### 👤 User {test_user_id} Taste Profile")
                
                user_ratings = ratings_df[ratings_df['userId'] == test_user_id]
                
                if not user_ratings.empty:
                    # User statistics
                    profile_col1, profile_col2, profile_col3, profile_col4 = st.columns(4)
                    
                    with profile_col1:
                        st.metric("Total Ratings", len(user_ratings))
                    
                    with profile_col2:
                        st.metric("Average Rating", f"{user_ratings['rating'].mean():.2f}")
                    
                    with profile_col3:
                        st.metric("Highest Rating", f"{user_ratings['rating'].max():.1f}")
                    
                    with profile_col4:
                        st.metric("Lowest Rating", f"{user_ratings['rating'].min():.1f}")
                    
                    # Top rated movies
                    st.markdown("#### 🌟 Top 5 Rated Movies")
                    top_rated = user_ratings.nlargest(5, 'rating').merge(
                        movies_df[['movieId', 'title', 'genres']],
                        on='movieId',
                        how='left'
                    )
                    
                    for idx, row in top_rated.iterrows():
                        st.markdown(f"**{row['title']}** - ⭐ {row['rating']:.1f}/5.0 - {row['genres']}")
                    
                    # Favorite genres
                    st.markdown("#### 🎭 Favorite Genres")
                    user_movies = user_ratings.merge(movies_df[['movieId', 'genres']], on='movieId')
                    all_genres = user_movies['genres'].str.split('|', expand=True).stack()
                    genre_counts = all_genres.value_counts().head(5)
                    
                    genre_col1, genre_col2 = st.columns(2)
                    with genre_col1:
                        for genre, count in genre_counts.items():
                            st.markdown(f"**{genre}**: {count} movies")
                
                print(f"[SUCCESS] User Taste Profile section rendered")
                
            except Exception as e:
                st.error(f"⚠️ Could not load user profile: {e}")
                print(f"[ERROR] User profile section failed: {e}")
                import traceback
                print(traceback.format_exc())
            
            # ========== SECTION 3: RECOMMENDATION EXPLANATION (INDEPENDENT) ==========
            try:
                print(f"[DEBUG] Rendering Recommendation Explanation section...")
                
                # ========== RECOMMENDATION EXPLANATION SECTION ==========
                st.markdown("---")
                st.markdown(f"### 🔍 Why These Recommendations?")
                st.markdown(f"*How {demo_algorithm} decided on these movies*")
                
                # Get explanation from first recommended movie
                if len(recommendations) > 0:
                    first_movie_id = recommendations.iloc[0]['movieId']
                    
                    # Get explanation context with comprehensive error handling
                    explanation_context = None
                    try:
                        if hasattr(algorithm, 'get_explanation_context'):
                            explanation_context = algorithm.get_explanation_context(test_user_id, first_movie_id)
                            print(f"[DEBUG] Explanation context retrieved: {explanation_context.get('method') if explanation_context else 'None'}")
                        else:
                            print(f"[WARNING] Algorithm {algorithm.name} does not have get_explanation_context method")
                    except Exception as e:
                        print(f"[ERROR] Failed to get explanation context: {e}")
                    
                    # Display explanation based on method
                    if explanation_context and explanation_context.get('method'):
                        method = explanation_context.get('method', 'unknown')
                        
                        if method == 'latent_factors':
                            # Safe attribute access with fallback
                            n_components = getattr(algorithm, 'n_components', 100)
                            st.info(f"""
                            **SVD Matrix Factorization** discovered hidden patterns in your rating history:
                            - Predicted rating: **{explanation_context.get('prediction', 0):.2f}/5.0**
                            - Your typical rating bias: **{explanation_context.get('user_bias', 0):+.2f}**
                            - This movie's quality score: **{explanation_context.get('movie_bias', 0):+.2f}**
                            - Combining {n_components} latent factors to match your taste
                            """)
                        
                        elif method == 'similar_users':
                            similar_count = explanation_context.get('similar_users_count', 0)
                            st.info(f"""
                            **User-Based Collaborative Filtering** found {similar_count} users with similar taste:
                            - These users loved the recommended movies
                            - Your ratings align closely with their preferences
                            - Community-validated recommendations
                            """)
                        
                        elif method == 'similar_items':
                            similar_count = explanation_context.get('similar_movies_count', 0)
                            st.info(f"""
                            **Item-Based Collaborative Filtering** analyzed movie similarity:
                            - Found {similar_count} movies similar to ones you loved
                            - Based on rating patterns across all users
                            - Stable, reliable recommendations
                            """)
                        
                        elif method == 'content_based':
                            st.info(f"""
                            **Content-Based Filtering** matched movie features to your preferences:
                            - Analyzed genres, tags, and themes from your top-rated movies
                            - Recommended movies with similar characteristics
                            - No dependency on other users' opinions
                            """)
                        
                        elif method == 'hybrid_ensemble':
                            weights = explanation_context.get('algorithm_weights', {})
                            primary = explanation_context.get('primary_algorithm', 'multiple')
                            st.info(f"""
                            **Hybrid Algorithm** combined multiple approaches:
                            - Primary method: **{primary}**
                            - SVD weight: {weights.get('svd', 0):.2f}
                            - User KNN weight: {weights.get('user_knn', 0):.2f}
                            - Item KNN weight: {weights.get('item_knn', 0):.2f}
                            - Content-Based weight: {weights.get('content_based', 0):.2f}
                            - Best of all worlds approach!
                            """)
                        else:
                            st.info(f"{demo_algorithm} analyzed your rating history and preferences to generate personalized recommendations.")
                    else:
                        # Fallback explanation when context not available
                        st.info(f"{demo_algorithm} analyzed your rating history and preferences to generate personalized recommendations.")
                
                print(f"[SUCCESS] Recommendation Explanation section rendered")
                
            except Exception as e:
                st.warning(f"⚠️ Could not load explanation details: {e}")
                print(f"[ERROR] Explanation section failed: {e}")
                import traceback
                print(traceback.format_exc())
                # Show generic explanation as fallback
                st.info(f"{demo_algorithm} analyzed your rating history to generate these recommendations.")
        
        with col2:
            st.info("""
            **💡 Quick Guide**
            
            1. Run benchmark to see metrics
            2. Select a test user ID
            3. Choose an algorithm
            4. Generate recommendations
            5. Explore results!
            """)
        
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
            
            st.plotly_chart(fig_genres, width="stretch")
        
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
            
            st.plotly_chart(fig_combos, width="stretch")
        
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
        st.plotly_chart(fig_genre_ratings, width="stretch")
    
    # TAB 3: Temporal Trends
    with tab3:
        st.markdown("## 📅 Movie Release Trends Over Time")
        
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
            st.plotly_chart(fig_years, width="stretch")
        
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
            st.plotly_chart(fig_decades, width="stretch")
        
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
        st.plotly_chart(fig_rating_trends, width="stretch")
        
        st.info("""
        **Insight**: Older movies tend to have slightly higher ratings, possibly due to 
        survivorship bias (only the best old movies remain popular).
        """)
    
    # TAB 4: Popularity Metrics
    with tab4:
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
            
            st.plotly_chart(fig_most_rated, width="stretch")
        
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
            
            st.plotly_chart(fig_highest_rated, width="stretch")
        
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
        st.plotly_chart(fig_scatter, width="stretch")
    
    # TAB 5: Movie Similarity Explorer
    with tab5:
        st.markdown("## 🔍 V2.1.0 Movie Similarity Explorer")
        st.markdown("### Discover similar movies using advanced AI algorithms")
        
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
                                # Find similar movies using V2.1.0 algorithms
                                try:
                                    with st.spinner(f"Finding similar movies using {similarity_algorithm}..."):
                                        
                                        # Map algorithm selection
                                        algo_map = {
                                            "Item KNN": AlgorithmType.ITEM_KNN,
                                            "SVD": AlgorithmType.SVD,
                                            "Hybrid": AlgorithmType.HYBRID
                                        }
                                        
                                        # Switch to the selected algorithm
                                        algorithm = manager.switch_algorithm(algo_map[similarity_algorithm])
                                        
                                        # Get similar movies using the algorithm's native method
                                        similar_movies_df = algorithm.get_similar_items(row['movieId'], n=10)
                                    
                                    st.markdown(f"### Movies Similar to **{row['title']}**")
                                    st.markdown(f"*Using {similarity_algorithm} algorithm*")
                                    
                                    # Display similar movies
                                    if similar_movies_df is not None and len(similar_movies_df) > 0:
                                        for sim_idx, sim_row in similar_movies_df.iterrows():
                                            similarity_score = sim_row.get('similarity', 0.0)
                                            sim_poster_path = sim_row.get('poster_path', None)
                                            sim_poster_url = get_tmdb_poster_url(sim_poster_path)
                                            
                                            st.markdown(f"""
                                            <div class="movie-card">
                                                <div style="display: flex; gap: 12px; align-items: flex-start;">
                                                    <div style="flex-shrink: 0;">
                                                        <img src="{sim_poster_url}" alt="{sim_row['title']}" 
                                                             style="width: 60px; height: 90px; object-fit: cover; border-radius: 5px;"
                                                             onerror="this.src='{PLACEHOLDER_POSTER}'">
                                                    </div>
                                                    <div style="flex: 1;">
                                                        <div style="font-weight: bold; margin-bottom: 0.5rem;">
                                                            {sim_idx+1}. {sim_row['title']} 
                                                            <span style="color: #4CAF50;">({similarity_score:.2%} match)</span>
                                                        </div>
                                                        <div style="color: #ccc; font-size: 0.9rem;">
                                                            <strong>Genres:</strong> {sim_row['genres']}
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                            
                                            if sim_idx >= 9:  # Show max 10 similar movies
                                                break
                                    else:
                                        st.info("No similar movies found. Try a different movie or algorithm.")
                                    
                                except Exception as e:
                                    st.error(f"Error finding similar movies: {e}")
                                    import traceback
                                    st.code(traceback.format_exc())
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
        ### 💡 V2.1.0 Similarity Analysis
        
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
    <p><strong>CineMatch V2.1.0 Analytics</strong> | Advanced Multi-Algorithm Analysis</p>
    <p>Powered by Plotly, Pandas, and Multiple AI Algorithms</p>
</div>
""", unsafe_allow_html=True)
