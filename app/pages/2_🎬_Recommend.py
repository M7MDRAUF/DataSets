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
from pathlib import Path

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
    print("Loading MovieLens dataset...")
    ratings_df = load_ratings(sample_size=sample_size)
    movies_df = load_movies()
    return ratings_df, movies_df

try:
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
    
    # Initialize algorithm manager (singleton pattern)
    manager = get_algorithm_manager()
    
    # Initialize data ONLY if not already initialized or dataset size changed
    # This prevents creating 3.3GB copies on every Streamlit rerun
    if not manager._is_initialized or manager._training_data is None:
        manager.initialize_data(ratings_df, movies_df)
    elif 'last_dataset_size' in st.session_state:
        # Dataset size changed - reinitialize
        current_size = len(ratings_df)
        stored_size = len(manager._training_data[0]) if manager._training_data else 0
        if current_size != stored_size:
            manager.initialize_data(ratings_df, movies_df)
    
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.info("""
    **Please ensure:**
    1. Dataset files are in `data/ml-32m/` directory
    2. All dependencies are installed
    3. Sufficient memory available (>4GB recommended)
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
        # Load the algorithm if not cached (to get metrics)
        if selected_algorithm not in manager.get_cached_algorithms():
            with st.spinner(f"Loading {selected_algorithm.value}..."):
                try:
                    current_algo = manager.get_algorithm(selected_algorithm)
                except Exception as e:
                    st.warning(f"Algorithm not yet loaded. Metrics will appear after generating recommendations.")
                    current_algo = None
        else:
            current_algo = manager.get_algorithm(selected_algorithm)
    except Exception as e:
        traceback.print_exc()
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
                mae_value = metrics.get('mae', 0.0)
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
if get_recs_button or 'current_recommendations' in st.session_state:
    
    # Validate user ID
    user_exists = user_id in ratings_df['userId'].values
    
    if not user_exists:
        st.warning(f"⚠️ User {user_id} not found in dataset. Generating recommendations for new user profile.")
    
    try:
        # Get the selected algorithm
        with st.spinner(f"🤖 Loading {selected_algorithm.value} algorithm..."):
            algorithm = manager.switch_algorithm(selected_algorithm)
        
        # Generate recommendations
        with st.spinner("🎯 Generating personalized recommendations..."):
            recommendations = algorithm.get_recommendations(user_id, n=10, exclude_rated=True)
            
            # Get user history if exists
            user_history = ratings_df[ratings_df['userId'] == user_id] if user_exists else pd.DataFrame()
            
            # Store in session state
            st.session_state.current_recommendations = recommendations
            st.session_state.current_user_id = user_id
            st.session_state.current_algorithm = selected_algorithm
            st.session_state.user_history = user_history
        
        # Display results
        recommendations = st.session_state.current_recommendations
        user_history = st.session_state.user_history
        
        # Validate recommendations DataFrame structure
        if recommendations is None or len(recommendations) == 0:
            st.warning("⚠️ No recommendations found for this user. This could happen if:")
            st.info("""
            - User is new (not in dataset)
            - Algorithm couldn't generate predictions
            - All candidate movies were filtered out
            
            **Try**: Different algorithm or different user ID
            """)
        else:
            # Validate required columns exist
            required_columns = ['movieId', 'title', 'genres', 'predicted_rating']
            missing_columns = [col for col in required_columns if col not in recommendations.columns]
            
            if missing_columns:
                st.error(f"❌ Invalid recommendations format - missing columns: {', '.join(missing_columns)}")
                st.info("This is likely a data processing issue. Please contact support.")
            else:
                # Main recommendations display
                st.markdown(f"""
                <div class="recommendation-header">
                    <h2>🎬 Recommendations for User {user_id}</h2>
                    <p>Generated by {algorithm.name} • Based on {len(user_history)} ratings</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Layout: Recommendations + User Profile
                rec_col, profile_col = st.columns([2.5, 1])
                
                with rec_col:
                    # Display recommendations with defensive coding
                    for idx, row in recommendations.iterrows():
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
                        
                        # Simplified buttons layout (no nested columns)
                        col_left, col_right = st.columns(2)
                        
                        with col_left:
                            # Explanation button
                            if st.button(f"💡 Why this movie?", key=f"explain_{idx}_{movie_id}"):
                                try:
                                    explanation = manager.get_recommendation_explanation(
                                        selected_algorithm, user_id, movie_id
                                    )
                                    st.markdown(f"""
                                    <div class="explanation-box">
                                        <strong>Why this recommendation?</strong><br>
                                        {explanation}
                                    </div>
                                    """, unsafe_allow_html=True)
                                except Exception as exp_error:
                                    st.warning(f"⚠️ Could not generate explanation: {str(exp_error)}")
                        
                        with col_right:
                            # Feedback buttons in single row
                            if st.button("👍 Like", key=f"like_{idx}_{movie_id}"):
                                st.success("Thanks for the feedback! 👍")
                            if st.button("👎 Not interested", key=f"dislike_{idx}_{movie_id}"):
                                st.info("Feedback noted! 👎")
                        
                        st.markdown("---")
                        
                    except Exception as card_error:
                        # Log error but continue rendering other recommendations
                        st.error(f"❌ Error displaying movie #{idx+1}: {str(card_error)}")
                        st.markdown("---")
                
                with profile_col:
                    # User taste profile (if user exists)
                    if len(user_history) > 0:
                        st.markdown("### 👤 Your Profile")
                        
                        # Basic stats
                        avg_rating = user_history['rating'].mean()
                        st.metric("Total Ratings", len(user_history))
                        st.metric("Average Rating", f"{avg_rating:.1f} ⭐")
                        
                        # Top genres
                        st.markdown("#### 🎭 Top Genres")
                        # Get genre distribution
                        genre_counts = {}
                        for _, movie_row in user_history.iterrows():
                            movie_info = movies_df[movies_df['movieId'] == movie_row['movieId']]
                            if len(movie_info) > 0:
                                genres = movie_info.iloc[0]['genres'].split('|')
                                for genre in genres:
                                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
                        
                        # Show top 5 genres
                        top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                        total_ratings = len(user_history)
                        
                        for genre, count in top_genres:
                            percentage = (count / total_ratings) * 100
                            emoji = get_genre_emoji(genre)
                            st.markdown(f"{emoji} **{genre}**: {percentage:.1f}%")
                        
                    else:
                        st.markdown("### 🆕 New User")
                        st.info("As a new user, recommendations are based on popular movies and content-based filtering.")
                
                # Algorithm comparison suggestion
                if selected_algorithm != AlgorithmType.HYBRID:
                    st.markdown("---")
                    st.info(f"💡 **Try the Hybrid algorithm** for potentially better recommendations that combine multiple approaches!")
    
    except Exception as e:
        st.error(f"❌ Error generating recommendations: {e}")
        
        # Enhanced error recovery options
        st.markdown("### 🛠️ Recovery Options")
        recovery_col1, recovery_col2, recovery_col3 = st.columns(3)
        
        with recovery_col1:
            if st.button("🔄 Clear Cache & Retry", help="Clear all cached data and restart"):
                st.cache_data.clear()
                if 'last_dataset_size' in st.session_state:
                    del st.session_state.last_dataset_size
                st.success("Cache cleared! Please refresh the page.")
        
        with recovery_col2:
            if st.button("🎲 Try Different User", help="Test with a guaranteed valid user ID"):
                st.session_state.suggested_user_id = 10  # Valid in all datasets
                st.info("Try User ID 10 - it's valid in all dataset sizes.")
        
        with recovery_col3:
            if st.button("📊 Check System Status", help="View current system state"):
                st.info(f"""
                **Current Status:**
                - Dataset Size: {len(ratings_df):,} ratings
                - Available Algorithms: {len(manager.get_cached_algorithms())}
                - Selected Algorithm: {selected_algorithm.value}
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