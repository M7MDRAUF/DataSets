"""
CineMatch V2.1.6 - Main Application Entry Point

Streamlit multi-page application for movie recommendations.
Now featuring 5 algorithms: SVD, User-KNN, Item-KNN, Content-Based, and Hybrid.

Author: CineMatch Team
Date: November 13, 2025
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
from src.data_processing import check_data_integrity

# Model preloading for faster first request (Task 27)
try:
    from src.memory.model_preloader import preload_on_startup
    # Start background preload if enabled via MODEL_PRELOAD_ON_STARTUP env var
    preload_on_startup()
except ImportError:
    pass  # Preloader not available, skip


# Page configuration
st.set_page_config(
    page_title="CineMatch V2.1.6 - Multi-Algorithm Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "CineMatch V2.1.6 - Multi-Algorithm Movie Recommendation Engine\n\nFeaturing 5 advanced algorithms: SVD, User-KNN, Item-KNN, Content-Based, and Hybrid ensemble.\n\nBuilt on MovieLens 32M dataset with 87K movies and 32M ratings."
    }
)

# Custom CSS
st.markdown("""
<style>
    /* Skip Navigation Link for Accessibility */
    .skip-link {
        position: absolute;
        top: -40px;
        left: 0;
        background: #E50914;
        color: white;
        padding: 8px 16px;
        z-index: 9999;
        text-decoration: none;
        font-weight: bold;
        border-radius: 0 0 4px 0;
    }

    .skip-link:focus {
        top: 0;
    }

    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #E50914;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }

    .movie-card {
        background: linear-gradient(135deg, #000000 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #E50914;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
        color: white;
    }

    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #E50914;
        color: white;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #b20710;
    }
</style>
""", unsafe_allow_html=True)

# Check data integrity on startup
st.sidebar.markdown("### 🔍 System Status")

with st.sidebar:
    with st.spinner("Checking data integrity..."):
        success, missing, error_msg = check_data_integrity()

        if success:
            st.success("✅ All dataset files found")
        else:
            st.error("❌ Data integrity check failed")
            st.stop()

# Sidebar navigation info
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 Navigation

- **🏠 Home**: Project overview & statistics
- **🎬 Recommend**: Get personalized recommendations
- **📊 Analytics**: Multi-algorithm performance analysis

### ℹ️ About

CineMatch V2.1.6 uses **5 advanced algorithms** to provide personalized movie recommendations:

🎯 **SVD** - Matrix Factorization
👥 **User-KNN** - User-based Filtering
🎬 **Item-KNN** - Item-based Filtering
🔍 **Content-Based** - Feature Analysis
🚀 **Hybrid** - Best of All

**Version**: 2.1.6
**Dataset**: MovieLens 32M
**Movies**: 87,585
**Ratings**: 32M+
""")

# Main header with accessibility landmark
st.markdown('<div id="main-content" role="main" aria-label="Main Content">', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">🎬 CineMatch V2.1.6</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-Algorithm Movie Recommendation Engine with Intelligent Ensemble Learning</p>', unsafe_allow_html=True)

# Welcome message
st.markdown("""
---

Welcome to **CineMatch V2.1.6**, a production-grade movie recommendation system featuring **5 advanced algorithms** built for master's thesis demonstration.

Use the **sidebar** to navigate between pages:
- **🏠 Home** - Overview and dataset statistics
- **🎬 Recommend** - Get personalized recommendations with algorithm selection
- **📊 Analytics** - Multi-algorithm performance comparison and insights

---

### 🚀 Getting Started

1. Navigate to the **🎬 Recommend** page
2. **Select an algorithm** from the dropdown (SVD, User-KNN, Item-KNN, Content-Based, or Hybrid)
3. Enter a User ID (e.g., 1, 123, 1000)
4. Click "Get Recommendations" to see personalized movie suggestions
5. Click "Explain" to understand why each movie was recommended

### ✨ Key Features (V2.1.6)

- ✅ **5 Advanced Algorithms**: SVD, User-KNN, Item-KNN, Content-Based, and Hybrid
- ✅ **Algorithm Comparison**: Benchmark and compare all algorithms side-by-side
- ✅ **Intelligent Ensemble**: Hybrid algorithm combines strengths of all methods
- ✅ **Pre-trained Models**: Instant loading for fast recommendations
- ✅ **Personalized Recommendations**: Based on your unique taste profile
- ✅ **Explainable AI**: Understand *why* each movie is recommended
- ✅ **User Taste Profiling**: See your genre preferences and rating patterns
- ✅ **Content-Based Discovery**: Find movies even with no rating history
- ✅ **Movie Similarity Explorer**: Discover similar movies across multiple dimensions

---

### 📊 System Information

""")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Dataset", value="MovieLens 32M", delta="87K movies")

with col2:
    st.metric(label="Algorithms", value="5 Advanced", delta="V2.1.6")

with col3:
    st.metric(label="Best RMSE", value="0.6829", delta="SVD")

with col4:
    st.metric(label="Coverage", value="100%", delta="Content-Based")

# Algorithm showcase
st.markdown("""
---

### 🎯 Available Algorithms

""")

algo_col1, algo_col2, algo_col3 = st.columns(3)

with algo_col1:
    st.markdown("""
    <div class="movie-card">
        <h3>🎯 SVD Matrix Factorization</h3>
        <p><b>RMSE:</b> 0.6829 | <b>Speed:</b> Fast</p>
        <p>Discovers hidden patterns in user ratings for highly accurate predictions.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="movie-card">
        <h3>🔍 Content-Based Filtering</h3>
        <p><b>Coverage:</b> 100% | <b>Speed:</b> Instant</p>
        <p>Analyzes movie features (genres, tags, titles) for perfect cold-start handling.</p>
    </div>
    """, unsafe_allow_html=True)

with algo_col2:
    st.markdown("""
    <div class="movie-card">
        <h3>👥 User-KNN</h3>
        <p><b>RMSE:</b> 0.9089 | <b>Interpretability:</b> High</p>
        <p>Finds users with similar taste for community-based recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="movie-card">
        <h3>🚀 Hybrid Ensemble</h3>
        <p><b>Adaptive</b> | <b>Best Overall</b></p>
        <p>Intelligently combines all algorithms with dynamic weighting.</p>
    </div>
    """, unsafe_allow_html=True)

with algo_col3:
    st.markdown("""
    <div class="movie-card">
        <h3>🎬 Item-KNN</h3>
        <p><b>RMSE:</b> 0.9127 | <b>Stability:</b> High</p>
        <p>Recommends movies similar to what you've enjoyed before.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
---

### 🎓 Academic Context

This project demonstrates:
- **Multi-Algorithm Comparison**: 5 recommendation algorithms with benchmarking
- **Ensemble Learning**: Hybrid algorithm with adaptive weighting strategies
- **Advanced Collaborative Filtering**: SVD, User-KNN, and Item-KNN implementations
- **Content-Based Filtering**: TF-IDF feature extraction with cosine similarity
- **Explainable AI (XAI)**: Clear explanations for each algorithm's recommendations
- **Professional Engineering**: Docker deployment, comprehensive testing, Git LFS for models
- **Performance Optimization**: Pre-trained models with intelligent caching (< 2s load time)
- **User-Centric Design**: Interactive visualizations and algorithm selection

**Built for**: Master's Thesis Defense
**Objective**: Showcase comprehensive ML application with production-ready implementation
**Status**: ✅ 100% Complete - All 5 algorithms trained and deployed

---

*Navigate to the **🎬 Recommend** page to start exploring personalized recommendations with algorithm comparison!*
""")

# Close main content landmark
st.markdown('</div>', unsafe_allow_html=True)
