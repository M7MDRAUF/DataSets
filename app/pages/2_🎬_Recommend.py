"""
CineMatch V2.1.0 - Recommend Page

Multi-algorithm movie recommendations with intelligent switching.
Supports all 5 algorithms: SVD, User-KNN, Item-KNN, Content-Based, and Hybrid.

Author: CineMatch Development Team
Date: November 13, 2025
"""

import streamlit as st
import pandas as pd
import sys
import traceback
import logging
from pathlib import Path
from datetime import datetime

# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.algorithms.algorithm_manager import get_algorithm_manager, AlgorithmType
from src.utils import (
    format_genres,
    create_rating_stars,
    get_genre_emoji
)
from src.data_processing import load_ratings, load_movies


# Page config
st.set_page_config(
    page_title="CineMatch V2.1.0 - Recommendations",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for enhanced UI
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
    border: 1px solid #dee2e6;
    margin: 1rem 0;
}

.algorithm-selector {
    background: #ffffff;
    border: 2px solid #E50914;
    border-radius: 10px;
    padding: 1rem;
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
    text-align: center;
    margin: 1rem 0;
}

.explanation-box {
    background: #e8f5e8;
    border-left: 4px solid #28a745;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 0 8px 8px 0;
}

.performance-metric {
    text-align: center;
    padding: 0.5rem;
    border-radius: 8px;
    margin: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="recommendation-header">
    <h1>🎬 CineMatch - Advanced Recommendations</h1>
    <p>Choose your preferred AI algorithm or let our Hybrid system decide for you</p>
</div>
""", unsafe_allow_html=True)

# Performance Configuration
with st.expander("⚙️ Performance Settings", expanded=False):
    st.markdown("### 🚀 Dataset Size Configuration")
    
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
    
    # Cache management
    col_clear, col_info = st.columns([1, 2])
    with col_clear:
        if st.button("🔄 Clear Cache", help="Clear cached data if changing dataset size"):
            st.cache_data.clear()
            st.success("Cache cleared! Refresh page to reload data.")
    with col_info:
        st.caption("💡 Use Fast Demo (100K) for quick testing, Balanced (500K) for good results with reasonable speed.")
        st.caption("⚠️ Changing dataset size will reset algorithm training.")

# Initialize data and algorithm manager
@st.cache_data
def load_data(sample_size):
    """Load and cache the dataset with configurable sampling"""
    logger.info(f"Loading MovieLens dataset with sample_size={sample_size}")
    ratings_df = load_ratings(sample_size=sample_size)
    movies_df = load_movies()
    logger.info(f"Loaded {len(ratings_df):,} ratings and {len(movies_df):,} movies")
    return ratings_df, movies_df

try:
    # CRITICAL FIX: Check if we're in a rerun triggered by button click
    # If manager is already initialized, skip the spinner to prevent UI flicker
    manager = get_algorithm_manager()
    
    if manager._is_initialized and manager._training_data is not None:
        # Already initialized - just get cached data silently
        ratings_df, movies_df = load_data(selected_sample_size)
    else:
        # First load - show spinner
        with st.spinner(f"🚀 Loading MovieLens dataset... ({dataset_size[0]} mode)"):
            ratings_df, movies_df = load_data(selected_sample_size)
        
        # Show dataset statistics
        st.success(f"✅ System ready! Loaded {len(ratings_df):,} ratings and {len(movies_df):,} movies")
        
        # Display performance expectation
        if selected_sample_size and selected_sample_size <= 100000:
            st.info("⚡ **Fast Demo Mode**: Algorithms will train in seconds!")
        elif selected_sample_size and selected_sample_size <= 500000:
            st.info("🔥 **Balanced Mode**: Good balance of speed and accuracy")
        elif selected_sample_size and selected_sample_size <= 1000000:
            st.info("🎯 **High Quality Mode**: Excellent accuracy with reasonable speed")
        else:
            st.warning("🐌 **Full Dataset Mode**: Maximum accuracy but slow training times")
    
    # Initialize data ONLY if not already initialized or dataset size changed
    # This prevents creating 3.3GB copies on every Streamlit rerun
    if not manager._is_initialized or manager._training_data is None:
        manager.initialize_data(ratings_df, movies_df)
        logger.info("Initialized algorithm manager with data")
    elif 'last_dataset_size' in st.session_state:
        # Dataset size changed - reinitialize
        current_size = len(ratings_df)
        stored_size = len(manager._training_data[0]) if manager._training_data else 0
        if current_size != stored_size:
            logger.info(f"Dataset size changed from {stored_size} to {current_size} - reinitializing")
            manager.initialize_data(ratings_df, movies_df)
    
except Exception as e:
    logger.critical(f"Failed to load dataset: {e}", exc_info=True)
    st.error(f"❌ Error loading data: {e}")
    
    # Provide specific troubleshooting based on error type
    error_type = type(e).__name__
    if "FileNotFound" in error_type or "No such file" in str(e):
        st.error("**Dataset files not found**")
        st.info("""
        📁 **Missing Dataset Files**
        
        Please ensure MovieLens dataset files exist:
        - `data/ml-32m/ratings.csv`
        - `data/ml-32m/movies.csv`
        - `data/ml-32m/tags.csv`
        - `data/ml-32m/links.csv`
        """)
    elif "Memory" in str(e) or "MemoryError" in error_type:
        st.error("**Insufficient Memory**")
        st.info("""
        💾 **Memory Issue**
        
        Try these solutions:
        1. Use **Fast Demo** or **Balanced** mode (Performance Settings)
        2. Close other applications
        3. Ensure system has >4GB available RAM
        4. Restart Docker container if using Docker
        """)
    else:
        st.info("""
        🛠️ **Troubleshooting Steps**
        
        1. Dataset files in `data/ml-32m/` directory
        2. All dependencies installed (`pip install -r requirements.txt`)
        3. Sufficient memory available (>4GB recommended)
        4. Check file permissions
        5. Verify CSV files are not corrupted
        """)
    st.stop()

# Sidebar: Algorithm Selection & Information
with st.sidebar:
    st.markdown("## 🤖 Algorithm Selection")
    
    # Algorithm selector
    available_algorithms = manager.get_available_algorithms()
    algorithm_names = [algo.value for algo in available_algorithms]
    algorithm_icons = ["🔮", "👥", "🎬", "�", "�🚀"]  # SVD, User KNN, Item KNN, Content-Based, Hybrid
    
    # Create algorithm options with icons
    algorithm_options = [f"{icon} {name}" for icon, name in zip(algorithm_icons, algorithm_names)]
    
    selected_algorithm_display = st.selectbox(
        "Choose Recommendation Algorithm:",
        options=algorithm_options,
        index=3,  # Default to Hybrid
        help="Each algorithm uses different approaches to find movies you'll love"
    )
    
    # Map back to algorithm type
    selected_idx = algorithm_options.index(selected_algorithm_display)
    selected_algorithm = available_algorithms[selected_idx]
    
    # Algorithm Information Card
    algo_info = manager.get_algorithm_info(selected_algorithm)
    
    st.markdown(f"""
    <div class="algorithm-card">
        <h3>{algo_info['icon']} {algo_info['name']}</h3>
        <p><strong>Description:</strong> {algo_info['description']}</p>
        <p><strong>Best For:</strong> {', '.join(algo_info['ideal_for'][:2])}</p>
        <p><strong>Interpretability:</strong> {algo_info['interpretability']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show cached algorithms
    cached_algos = manager.get_cached_algorithms()
    if cached_algos:
        st.markdown("### ⚡ Cached Algorithms")
        for algo in cached_algos:
            st.markdown(f"✅ {algo.value}")
    
    # Performance comparison (if available)
    try:
        performance_df = manager.get_performance_comparison()
        if not performance_df.empty:
            st.markdown("### 📊 Algorithm Comparison")
            st.dataframe(
                performance_df[['Algorithm', 'RMSE', 'Interpretability']],
                width="stretch"
            )
    except AttributeError as e:
        st.warning(f"⚠️ Performance comparison temporarily unavailable: {e}")
        st.info("This will be available once algorithms are trained and have metrics.")

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 👤 User Input")
    
    input_col1, input_col2 = st.columns([3, 1])
    
    with input_col1:
        # Get actual user ID range from the dataset
        min_user_id = int(ratings_df['userId'].min())
        max_user_id = int(ratings_df['userId'].max())
        
        # Use a valid default user ID (either min_user_id or a safe fallback)
        default_user_id = max(min_user_id, 10)  # Prefer 10 if it's valid, otherwise use min
        
        user_id = st.number_input(
            "Enter User ID",
            min_value=min_user_id,
            max_value=max_user_id,
            value=default_user_id,
            step=1,
            help=f"Valid range: {min_user_id:,} - {max_user_id:,}. Try: {min_user_id}, {default_user_id}, or any number in this range."
        )
    
    with input_col2:
        get_recs_button = st.button(
            "🎯 Get Recommendations", 
            type="primary", 
            use_container_width=True
        )

with col2:
    # Algorithm Performance Metrics (live)
    st.markdown("### ⚡ Algorithm Performance")
    
    try:
        # Only show metrics if algorithm is already cached (don't load it just for metrics)
        if selected_algorithm in manager.get_cached_algorithms():
            current_algo = manager.get_algorithm(selected_algorithm)
        else:
            current_algo = None
    except Exception as e:
        logger.error(f"Error accessing cached algorithm: {e}")
        current_algo = None
    
    if current_algo is not None:
        try:
            # Get metrics from algorithm manager (uses cached pre-computed values)
            metrics_data = manager.get_algorithm_metrics(selected_algorithm)
            
            # Single column metrics display (no nesting)
            if metrics_data.get('status') == 'Trained':
                metrics = metrics_data.get('metrics', {})
                
                # Display RMSE
                rmse_value = metrics.get('rmse', current_algo.metrics.rmse)
                if rmse_value > 0:
                    st.metric(
                        "RMSE", 
                        f"{rmse_value:.4f}",
                        help="Lower is better (Root Mean Square Error)"
                    )
                else:
                    st.metric("RMSE", "N/A", help="Metrics not yet calculated")
                
                # Display MAE (Mean Absolute Error)
                mae_value = metrics.get('mae', current_algo.metrics.mae if current_algo.metrics.mae > 0 else 0.0)
                if mae_value > 0:
                    st.metric(
                        "MAE", 
                        f"{mae_value:.4f}",
                        help="Mean Absolute Error (lower is better)"
                    )
                else:
                    st.metric("MAE", "N/A", help="Not yet calculated")
                
                # Display Coverage
                coverage_value = metrics.get('coverage', current_algo.metrics.coverage)
                if coverage_value > 0:
                    st.metric(
                        "Coverage", 
                        f"{coverage_value:.1f}%",
                        help="Percentage of movies that can be recommended"
                    )
                else:
                    st.metric("Coverage", "N/A", help="Coverage not yet calculated")
                
                # Display Training Time
                training_time = metrics_data.get('training_time', current_algo.metrics.training_time)
                if training_time != 'Unknown' and (isinstance(training_time, (int, float)) and training_time > 0):
                    st.metric(
                        "Training Time", 
                        f"{training_time:.1f}s" if isinstance(training_time, (int, float)) else training_time,
                        help="Time taken to train the algorithm"
                    )
                else:
                    st.metric("Training Time", "N/A", help="Training time not recorded")
                
                # Display Memory Usage
                if current_algo.metrics.memory_usage_mb > 0:
                    st.metric(
                        "Memory Usage", 
                        f"{current_algo.metrics.memory_usage_mb:.1f}MB",
                        help="Memory footprint of the algorithm"
                    )
                else:
                    st.metric("Memory Usage", "N/A", help="Memory usage not recorded")
            else:
                st.info("📊 Metrics will be calculated on first load")
        except Exception as metrics_error:
            st.warning("⚠️ Metrics unavailable - algorithm is still loading")
    else:
        st.info("📊 Select an algorithm to view performance metrics")

st.markdown("---")

# Recommendation Generation
# Only generate NEW recommendations when button is clicked
if get_recs_button:
    
    logger.info(f"Recommendation request: user_id={user_id}, algorithm={selected_algorithm.value}")
    
    # Validate user ID (defensive check even though number_input has constraints)
    if user_id is None or user_id <= 0:
        logger.warning(f"Invalid user ID: {user_id}")
        st.error("❌ Invalid User ID. Please enter a positive number.")
        st.stop()
    
    # Validate user ID is within dataset range
    min_user_id = int(ratings_df['userId'].min())
    max_user_id = int(ratings_df['userId'].max())
    
    if user_id < min_user_id or user_id > max_user_id:
        st.error(f"❌ User ID {user_id} is out of range. Valid range: {min_user_id:,} - {max_user_id:,}")
        st.info(f"💡 **Suggestion**: Try User ID {min_user_id}, {max(min_user_id, 10)}, or {max_user_id}")
        st.stop()
    
    # Check if user exists in dataset
    user_exists = user_id in ratings_df['userId'].values
    
    if not user_exists:
        st.warning(f"⚠️ User {user_id} not found in dataset. Generating recommendations for new user profile.")
    
    try:
        # Get the selected algorithm (use cached if available)
        logger.info(f"Loading algorithm: {selected_algorithm.value}")
        
        # Check if algorithm is already cached to avoid reloading
        if selected_algorithm in manager.get_cached_algorithms():
            algorithm = manager.get_algorithm(selected_algorithm)
            logger.info(f"Using cached {algorithm.name}")
        else:
            with st.spinner(f"🤖 Loading {selected_algorithm.value} algorithm..."):
                algorithm = manager.switch_algorithm(selected_algorithm)
            logger.info(f"Algorithm loaded successfully: {algorithm.name}")
        
        # Generate recommendations
        logger.info(f"Generating recommendations for user {user_id}")
        with st.spinner("🎯 Generating personalized recommendations..."):
            recommendations = algorithm.get_recommendations(user_id, n=10, exclude_rated=True)
            logger.info(f"Generated {len(recommendations) if recommendations is not None else 0} recommendations")
            
            # Get user history if exists
            user_history = ratings_df[ratings_df['userId'] == user_id] if user_exists else pd.DataFrame()
            
            # Store in session state
            st.session_state.current_recommendations = recommendations
            st.session_state.current_user_id = user_id
            st.session_state.current_algorithm = selected_algorithm
            st.session_state.user_history = user_history
            
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}", exc_info=True)
        st.error(f"❌ Error generating recommendations: {e}")
        traceback.print_exc()
        st.stop()

# Display recommendations (if they exist in session state)
if 'current_recommendations' in st.session_state:
    try:
        # Get algorithm instance for display (from cache, don't reload)
        if st.session_state.current_algorithm in manager.get_cached_algorithms():
            algorithm = manager.get_algorithm(st.session_state.current_algorithm)
        else:
            # Fallback to current selected algorithm
            algorithm = manager.get_algorithm(selected_algorithm)
        
        # Get stored data with comprehensive validation
        recommendations = st.session_state.current_recommendations
        user_history = st.session_state.user_history
        displayed_user_id = st.session_state.current_user_id
        
        # DEFENSIVE VALIDATION: Check data integrity
        if recommendations is None:
            logger.error("Recommendations is None despite being in session_state")
            st.error("❌ Recommendation generation failed. Please try again.")
            st.stop()
        
        if not isinstance(recommendations, pd.DataFrame):
            logger.error(f"Recommendations is not a DataFrame: {type(recommendations)}")
            st.error("❌ Invalid recommendation format. Please refresh the page.")
            st.stop()
        
        if len(recommendations) == 0:
            logger.warning("Recommendations DataFrame is empty")
            st.warning("⚠️ No recommendations found for this user. This could happen if:")
            st.info("""
            - User is new (not in dataset)
            - Algorithm couldn't generate predictions
            - All candidate movies were filtered out
            
            **Try**: Different algorithm or different user ID
            """)
            st.stop()
        
        # Validate required columns exist
        required_columns = ['movieId', 'title', 'genres', 'predicted_rating']
        missing_columns = [col for col in required_columns if col not in recommendations.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            logger.error(f"Available columns: {recommendations.columns.tolist()}")
            st.error(f"❌ Invalid recommendations format - missing columns: {', '.join(missing_columns)}")
            st.info("This is likely a data processing issue. Please contact support.")
            st.stop()
        
        # Main recommendations display
        st.markdown(f"""
        <div class="recommendation-header">
            <h2>🎬 Recommendations for User {displayed_user_id}</h2>
            <p>Generated by {algorithm.name} • Based on {len(user_history)} ratings</p>
        </div>
        """, unsafe_allow_html=True)
        
        # DEBUG: Log recommendation data structure
        logger.info(f"Displaying {len(recommendations)} recommendations")
        logger.info(f"Columns: {recommendations.columns.tolist()}")
        logger.info(f"User history size: {len(user_history)}")
        
        # Layout: Recommendations + User Profile
        rec_col, profile_col = st.columns([2.5, 1])
        
        with rec_col:
            logger.info("Entering rec_col block")
            # Display recommendations with defensive coding
            for idx, row in recommendations.iterrows():
                logger.info(f"Rendering recommendation #{idx+1}")
                try:
                    movie_id = row['movieId']
                    title = row.get('title', 'Unknown Movie')
                    genres = row.get('genres', '')
                    predicted_rating = float(row.get('predicted_rating', 0.0))
                    
                    # Movie card with simplified layout
                    st.markdown(f"""
                    <div class="movie-card">
                        <h3 style="color: white; margin: 0.5rem 0;">#{idx+1}. {title}</h3>
                        <p style="color: #ccc;"><strong>Genres:</strong> {format_genres(genres)}</p>
                        <p style="color: #ddd;"><strong>Predicted Rating:</strong> {create_rating_stars(predicted_rating)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Feedback buttons with explanation
                    col_like, col_dislike, col_explain, col_info = st.columns([1, 1, 1, 2])
                    with col_like:
                        if st.button("👍 Like", key=f"like_{idx}_{movie_id}", help="Like this recommendation"):
                            st.toast("👍 Liked!", icon="✅")
                    with col_dislike:
                        if st.button("👎 Dislike", key=f"dislike_{idx}_{movie_id}", help="Not interested"):
                            st.toast("👎 Not interested", icon="ℹ️")
                    with col_explain:
                        if st.button("💡 Explain", key=f"explain_{idx}_{movie_id}", help="Why was this recommended?"):
                            # Toggle explanation visibility
                            explain_key = f"show_explain_{idx}_{movie_id}"
                            if explain_key not in st.session_state:
                                st.session_state[explain_key] = False
                            st.session_state[explain_key] = not st.session_state[explain_key]
                            st.rerun()
                    with col_info:
                        st.caption(f"Movie ID: {movie_id}")
                    
                    # Show explanation if toggled
                    explain_key = f"show_explain_{idx}_{movie_id}"
                    if explain_key in st.session_state and st.session_state[explain_key]:
                        with st.expander("🔍 Why this recommendation?", expanded=True):
                            try:
                                # Get explanation from algorithm
                                if hasattr(algorithm, 'get_explanation_context'):
                                    explanation_context = algorithm.get_explanation_context(user_id, movie_id)
                                    
                                    if explanation_context and explanation_context.get('method'):
                                        method = explanation_context.get('method', 'unknown')
                                        
                                        if method == 'latent_factors':
                                            n_components = getattr(algorithm, 'n_components', 100)
                                            st.info(f"""
                                            **SVD Matrix Factorization** discovered hidden patterns:
                                            - Predicted rating: **{explanation_context.get('prediction', 0):.2f}/5.0**
                                            - Your rating bias: **{explanation_context.get('user_bias', 0):+.2f}**
                                            - Movie quality score: **{explanation_context.get('movie_bias', 0):+.2f}**
                                            - Using {n_components} latent factors to match your taste
                                            """)
                                        
                                        elif method == 'similar_users':
                                            similar_count = explanation_context.get('similar_users_count', 0)
                                            st.info(f"""
                                            **User-Based Filtering** found {similar_count} users with similar taste who loved this movie.
                                            - Your ratings align with their preferences
                                            - Community-validated recommendation
                                            """)
                                        
                                        elif method == 'similar_items':
                                            similar_count = explanation_context.get('similar_movies_count', 0)
                                            st.info(f"""
                                            **Item-Based Filtering** found {similar_count} movies similar to ones you rated highly.
                                            - Based on rating patterns across all users
                                            - Reliable similarity matching
                                            """)
                                        
                                        elif method == 'content_based':
                                            st.info(f"""
                                            **Content-Based Filtering** matched movie features to your preferences:
                                            - Analyzed genres and themes from your top-rated movies
                                            - Found similar characteristics in this recommendation
                                            """)
                                        
                                        elif method == 'hybrid_ensemble':
                                            weights = explanation_context.get('algorithm_weights', {})
                                            primary = explanation_context.get('primary_algorithm', 'multiple')
                                            st.info(f"""
                                            **Hybrid Algorithm** combined multiple approaches:
                                            - Primary method: **{primary}**
                                            - SVD: {weights.get('svd', 0):.0%} | User KNN: {weights.get('user_knn', 0):.0%}
                                            - Item KNN: {weights.get('item_knn', 0):.0%} | Content: {weights.get('content_based', 0):.0%}
                                            """)
                                        else:
                                            st.info(f"**{selected_algorithm}** analyzed your viewing history and preferences.")
                                    else:
                                        st.info(f"**{selected_algorithm}** analyzed your viewing history and preferences.")
                                else:
                                    st.info(f"**{selected_algorithm}** analyzed your viewing history and preferences to generate this recommendation.")
                            except Exception as explain_error:
                                logger.error(f"Error getting explanation: {str(explain_error)}")
                                st.info(f"**{selected_algorithm}** recommended this based on your viewing patterns.")
                    
                    st.markdown("---")
                    
                except Exception as card_error:
                    # Log error but continue rendering other recommendations
                    logger.error(f"Error rendering card #{idx+1}: {str(card_error)}", exc_info=True)
                    st.error(f"❌ Error displaying movie #{idx+1}: {str(card_error)}")
                    st.markdown("---")
            
            logger.info("Completed rec_col block")
        
        with profile_col:
            logger.info("Entering profile_col block")
            # User taste profile (if user exists)
            if len(user_history) > 0:
                st.markdown("### 👤 Your Profile")
                
                # Basic stats
                avg_rating = user_history['rating'].mean()
                st.metric("Total Ratings", len(user_history))
                st.metric("Average Rating", f"{avg_rating:.1f} ⭐")
                
                # Top genres
                st.markdown("#### 🎭 Top Genres")
                # Get genre distribution with defensive coding
                genre_counts = {}
                try:
                    for _, movie_row in user_history.iterrows():
                        movie_info = movies_df[movies_df['movieId'] == movie_row['movieId']]
                        if len(movie_info) > 0:
                            genres_str = movie_info.iloc[0].get('genres', '')
                            if genres_str and pd.notna(genres_str):
                                genres = str(genres_str).split('|')
                                for genre in genres:
                                    genre = genre.strip()
                                    if genre:  # Only count non-empty genres
                                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
                    
                    # Show top 5 genres
                    if genre_counts:
                        top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                        total_ratings = len(user_history)
                        
                        for genre, count in top_genres:
                            percentage = (count / total_ratings) * 100
                            emoji = get_genre_emoji(genre)
                            st.markdown(f"{emoji} **{genre}**: {percentage:.1f}%")
                    else:
                        st.info("Genre preferences not available")
                except Exception as genre_error:
                    logger.warning(f"Error calculating genre distribution: {genre_error}")
                    st.info("Unable to calculate genre preferences")
                
            else:
                st.markdown("### 🆕 New User")
                st.info("As a new user, recommendations are based on popular movies and content-based filtering.")
            
            logger.info("Completed profile_col block")
        
        # Algorithm comparison suggestion
        if st.session_state.current_algorithm != AlgorithmType.HYBRID:
            st.markdown("---")
            st.info(f"💡 **Try the Hybrid algorithm** for potentially better recommendations that combine multiple approaches!")
        
        logger.info("Display section completed successfully")
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}", exc_info=True)
        st.error(f"❌ Error generating recommendations: {e}")
        
        # Enhanced error recovery options
        st.markdown("### 🛠️ Recovery Options")
        recovery_col1, recovery_col2, recovery_col3 = st.columns(3)
        
        with recovery_col1:
            if st.button("🔄 Clear Cache & Retry", help="Clear all cached data and restart"):
                logger.info("User clicked: Clear Cache & Retry")
                st.cache_data.clear()
                if 'last_dataset_size' in st.session_state:
                    del st.session_state.last_dataset_size
                st.success("Cache cleared! Please refresh the page.")
        
        with recovery_col2:
            if st.button("🎲 Try Different User", help="Test with a guaranteed valid user ID"):
                logger.info("User clicked: Try Different User")
                st.session_state.suggested_user_id = 10  # Valid in all datasets
                st.info("Try User ID 10 - it's valid in all dataset sizes.")
        
        with recovery_col3:
            if st.button("📊 Check System Status", help="View current system state"):
                logger.info("User clicked: Check System Status")
                cached_algos = manager.get_cached_algorithms()
                st.info(f"""
                **Current Status:**
                - Dataset Size: {len(ratings_df):,} ratings
                - Cached Algorithms: {len(cached_algos)} ({', '.join([a.value for a in cached_algos])})
                - Selected Algorithm: {selected_algorithm.value}
                - User ID: {user_id}
                """)
        
        st.markdown("""
        ### 💡 Common Solutions:
        - Try User IDs: **10, 15, 22, 33** (guaranteed to work)
        - Switch to **Fast Demo** dataset for better compatibility
        - Use **SVD algorithm** - it's most reliable for new users
        """)

else:
    # Instructions when no recommendations yet
    st.markdown("""
    ## 🎯 How to Get Started
    
    1. **Choose an algorithm** from the sidebar (Hybrid recommended for best results)
    2. **Enter a User ID** above (try: **10, 15, 22, 33** - guaranteed to work!)  
    3. **Click "Get Recommendations"** to see personalized movie suggestions
    4. **Click "Explain"** to understand why each movie was recommended
    
    ### 🤖 Algorithm Guide
    
    - **🔮 SVD**: Best overall accuracy, finds hidden patterns
    - **👥 User KNN**: Find people like you, highly interpretable  
    - **🎬 Item KNN**: Movies similar to your favorites, stable results
    - **🚀 Hybrid**: Combines all algorithms intelligently (recommended)
    
    ### 💡 Tips
    
    - Different algorithms may suggest different movies - try them all!
    - New users (not in dataset) get content-based recommendations
    """)

# Footer with algorithm information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>CineMatch V2.1.0 - Multi-Algorithm Recommendation Engine</strong></p>
    <p>Powered by SVD Matrix Factorization + KNN Collaborative Filtering + Hybrid Intelligence</p>
    <p>Trained on 32+ million ratings for maximum accuracy and diversity</p>
</div>
""", unsafe_allow_html=True)