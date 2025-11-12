"""
CineMatch V2.0 - Main Application Entry Point

Streamlit multi-page application for movie recommendations with multi-algorithm support.

Author: CineMatch Team
Date: November 11, 2025
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing import check_data_integrity


# Page configuration
st.set_page_config(
    page_title="CineMatch V2.0 - Multi-Algorithm Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "CineMatch V2.0 - Multi-Algorithm Movie Recommendation Engine\n\nBuilt with 5 advanced algorithms: SVD, User-KNN, Item-KNN, Content-Based, and Hybrid on MovieLens 32M dataset."
    }
)

# Custom CSS
st.markdown("""
<style>
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
- **🎬 Recommend**: Multi-algorithm recommendations  
- **📊 Analytics**: Explore dataset insights

### ℹ️ About

CineMatch V2.0 offers **5 advanced algorithms** to provide personalized movie recommendations with explainable AI.

**Version**: 2.0  
**Dataset**: MovieLens 32M  
**Algorithms**: 
- 🔮 SVD Matrix Factorization
- 👥 User-Based KNN
- 🎬 Item-Based KNN
- 📝 Content-Based Filtering
- 🚀 Hybrid (Best of All)
""")

# Main header
st.markdown('<h1 class="main-header">🎬 CineMatch V2.0</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-Algorithm Movie Recommendation Engine with AI-Powered Intelligence</p>', unsafe_allow_html=True)

# Welcome message
st.markdown("""
---

Welcome to **CineMatch V2.0**, a production-grade multi-algorithm movie recommendation system built for a master's thesis demonstration.

Use the **sidebar** to navigate between pages:
- **🏠 Home** - Overview and dataset statistics
- **🎬 Recommend** - Choose from 5 advanced recommendation algorithms
- **📊 Analytics** - Explore data insights and visualizations

---

### 🚀 Getting Started

1. Navigate to the **🎬 Recommend** page
2. **Choose your algorithm** from the sidebar (SVD, User-KNN, Item-KNN, Content-Based, or Hybrid)
3. Enter a User ID (e.g., 1, 123, 1000)
4. Click "Get Recommendations" to see personalized movie suggestions
5. Click "Explain" to understand why each movie was recommended

### ✨ Key Features

- 🤖 **5 Advanced Algorithms**: Choose the best algorithm for your needs
- ✅ **Personalized Recommendations**: Based on your unique taste profile
- ✅ **Explainable AI**: Understand *why* each movie is recommended
- ✅ **Algorithm Comparison**: Compare performance metrics across algorithms
- ✅ **User Taste Profiling**: See your genre preferences and rating patterns
- ✅ **Configurable Dataset Size**: From 100K to full 32M ratings
- ✅ **Hybrid Intelligence**: Combines all algorithms for maximum accuracy

---

### 📊 System Information

""")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Dataset", value="MovieLens 32M", delta="32M ratings")

with col2:
    st.metric(label="Algorithms", value="5 Advanced", delta="Multi-algorithm")

with col3:
    st.metric(label="Version", value="V2.0", delta="Latest")

with col4:
    st.metric(label="Target RMSE", value="< 0.87", delta="Production-ready")

st.markdown("""
---

### 🎓 Academic Context

This project demonstrates:
- **Multi-Algorithm Approach**: SVD, KNN (User & Item), Content-Based, and Hybrid methods
- **Explainable AI (XAI)**: Transparent recommendation reasoning
- **Professional Engineering**: Docker containerization, modular architecture, caching
- **Algorithm Comparison**: Performance metrics and benchmarking
- **User-centric Design**: Interactive visualizations and intuitive interface
- **Scalability**: Configurable dataset sizes (100K to 32M ratings)

**Built for**: Master's Thesis Defense  
**Objective**: Showcase comprehensive ML recommendation system with production-grade quality

---

### 🤖 Available Algorithms

| Algorithm | Type | Best For |
|-----------|------|----------|
| 🔮 **SVD** | Matrix Factorization | General recommendations, cold start |
| 👥 **User-KNN** | Collaborative Filtering | Finding similar users |
| 🎬 **Item-KNN** | Collaborative Filtering | "If you liked X" recommendations |
| 📝 **Content-Based** | Feature Analysis | Genre/tag-based matching |
| 🚀 **Hybrid** | Ensemble | Maximum accuracy (combines all) |

---

*Navigate to the **🎬 Recommend** page to start exploring with multiple algorithms!*
""")
