"""
CineMatch V1.0.0 - Main Application Entry Point

Streamlit multi-page application for movie recommendations.

Author: CineMatch Team
Date: October 24, 2025
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing import check_data_integrity


# Page configuration
st.set_page_config(
    page_title="CineMatch - Intelligent Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "CineMatch V1.0.0 - Intelligent Movie Recommendation Engine\n\nBuilt with collaborative filtering (SVD) on MovieLens 32M dataset."
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
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
        transition: transform 0.2s;
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
- **📊 Analytics**: Explore dataset insights

### ℹ️ About

CineMatch uses collaborative filtering (SVD matrix factorization) to provide personalized movie recommendations with explainable AI.

**Version**: 1.0.0  
**Dataset**: MovieLens 32M  
**Algorithm**: SVD (Surprise)
""")

# Main header
st.markdown('<h1 class="main-header">🎬 CineMatch</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent Movie Recommendation Engine powered by Collaborative Filtering</p>', unsafe_allow_html=True)

# Welcome message
st.markdown("""
---

Welcome to **CineMatch**, a production-grade movie recommendation system built for a master's thesis demonstration.

Use the **sidebar** to navigate between pages:
- **🏠 Home** - Overview and dataset statistics
- **🎬 Recommend** - Get personalized movie recommendations
- **📊 Analytics** - Explore data insights and visualizations

---

### 🚀 Getting Started

1. Navigate to the **🎬 Recommend** page
2. Enter a User ID (e.g., 1, 123, 1000)
3. Click "Get Recommendations" to see personalized movie suggestions
4. Click "Explain" to understand why each movie was recommended

### ✨ Key Features

- ✅ **Personalized Recommendations**: Based on your unique taste profile
- ✅ **Explainable AI**: Understand *why* each movie is recommended
- ✅ **User Taste Profiling**: See your genre preferences and rating patterns
- ✅ **Surprise Me Mode**: Discover movies outside your usual taste
- ✅ **Movie Similarity**: Find movies similar to ones you love

---

### 📊 System Information

""")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Dataset", value="MovieLens 32M", delta="32M ratings")

with col2:
    st.metric(label="Algorithm", value="SVD", delta="Matrix Factorization")

with col3:
    st.metric(label="Target RMSE", value="< 0.87", delta="Production-ready")

st.markdown("""
---

### 🎓 Academic Context

This project demonstrates:
- Advanced collaborative filtering techniques
- Explainable AI (XAI) for recommendation systems
- Professional software engineering practices (Docker, testing, documentation)
- User-centric design with interactive visualizations

**Built for**: Master's Thesis Defense  
**Objective**: Showcase practical ML application with professional polish

---

*Navigate to the **🎬 Recommend** page to start exploring personalized recommendations!*
""")
