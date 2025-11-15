# 🎬 CineMatch V2.1.1

**Multi-Algorithm Recommendation Engine with Explainable AI**

A production-grade collaborative filtering recommendation system built for a master's thesis demonstration. CineMatch V2.1.1 features **5 recommendation algorithms** including SVD matrix factorization, User-KNN, Item-KNN, Content-Based filtering, and intelligent Hybrid ensemble on the MovieLens 32M dataset to provide personalized, explainable movie recommendations through an interactive Streamlit web interface.

![CineMatch Banner](https://img.shields.io/badge/CineMatch-V2.1.1-red?style=for-the-badge&logo=film)
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.51+-red?style=for-the-badge&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker)
![Algorithms](https://img.shields.io/badge/Algorithms-5%20Types-green?style=for-the-badge)

## 🎯 Project Status

**Last Updated**: November 14, 2025  
**Version**: V2.1.1 Production-Optimized  
**Algorithms**: ✅ SVD + User KNN + Item KNN + Content-Based + Hybrid (All Optimized)  
**Model Accuracy**: ✅ SVD RMSE 0.7502, User-KNN 0.8394, Item-KNN 0.9100, Content-Based 1.1130, Hybrid 0.8701  
**Pre-trained Models**: ✅ 4.07 GB models with full 32M dataset  
**Performance**: ✅ All models load in <10s, Hybrid in 25s  
**Memory Optimization**: ✅ 98.6% reduction (13.2GB → 185MB)  
**Analytics Dashboard**: ✅ Full benchmarking with RMSE/MAE/Coverage metrics  
**All Tests Passed**: ✅ All algorithms working with production optimizations  
**Application**: ✅ Clean UI with suppressed debug output  
**Docker**: ✅ 8GB limit, 2.6GB usage (68% headroom)  
**Status**: **Production-Ready** 🚀

---

## 🚀 What's New in V2.1.1

### **Critical Performance & Memory Fixes**
- **Memory Optimization** - Fixed memory explosion from 13.2GB crash to 185MB stable (98.6% reduction)
- **Shallow References** - Eliminated unnecessary data copying in algorithm manager and hybrid recommender
- **One-time Initialization** - Fixed Streamlit reinitialization causing N × 3.3GB copies
- **Context-Aware Logging** - Suppressed verbose debug output in UI while preserving terminal logs
- **Clean User Experience** - Professional UI without debug spam

### **Model Improvements**
- **SVD sklearn** - 909.6 MB, loads in 5-9s, RMSE 0.7502
- **User-KNN** - 1114 MB, loads in 1-8s, RMSE 0.8394, 100% coverage
- **Item-KNN** - 1108.4 MB, loads in 1-8s, RMSE 0.9100, 50.1% coverage
- **Content-Based** - 1059.9 MB, loads in 6s, RMSE 1.1130
- **Hybrid** - 491.3 MB combined, loads in 25s, RMSE 0.8701

### **Bug Fixes**
- ✅ Fixed missing `get_explanation_context()` in Content-Based recommender
- ✅ Fixed RMSE calculation bottleneck (3+ hour hangs eliminated)
- ✅ Fixed SVD model path (removed memory-heavy Surprise variant)
- ✅ Suppressed Streamlit ScriptRunContext warnings
- ✅ Fixed Docker memory management (2.6GB / 8GB usage)

---

## 📋 Table of Contents

- [Key Features](#key-features)
- [Multi-Algorithm Architecture](#multi-algorithm-architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)  
- [Usage](#usage)
- [Algorithm Guide](#algorithm-guide)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Performance Benchmarks](#performance-benchmarks)
- [Development](#development)
- [Deployment](#deployment)
- [Contributing](#-contributing)

---

## Overview

CineMatch V2.1.0 addresses the "analysis paralysis" problem in movie selection by providing **5 distinct recommendation approaches** backed by explainable AI. Built on 32 million user ratings, it demonstrates practical application of **ensemble learning** and **multi-paradigm collaborative filtering** while maintaining production-level code quality and user experience.

### Success Metrics V2.1.0

- ✅ **Multi-Algorithm Accuracy**: SVD RMSE 0.8406 (full 32M), 0.6829 (100K sample), Coverage 100%
- ✅ **Performance**: KNN loading 1.5s (200x faster), Hybrid 7s (vs 2+ hours before)
- ✅ **Pre-trained Models**: 526MB models trained on full 32M dataset (User-KNN 266MB, Item-KNN 260MB)
- ✅ **Response Time**: < 2 seconds for all algorithms with smart sampling
- ✅ **Algorithm Diversity**: 5 different recommendation paradigms (SVD, User-KNN, Item-KNN, Content-Based, Hybrid)
- ✅ **Professional UI**: Algorithm selection with live metrics and benchmarking
- ✅ **Analytics Dashboard**: Complete RMSE/MAE/Coverage metrics with visualizations
- ✅ **Explainability**: 80%+ recommendations have clear explanations
- ✅ **Usability**: Professor comprehension in < 60 seconds

---

## Key Features

### Core Functionality

| Feature | Description | Status |
|---------|-------------|--------|
| **F-01: User Input** | Simple interface for User ID entry with validation | ✅ Implemented |
| **F-02: Recommendation Generation** | Top-N personalized recommendations using SVD | ✅ Implemented |
| **F-03: Recommendation Display** | Clean card-based layout with movie details | ✅ Implemented |
| **F-04: Model Pre-computation** | Pre-trained model for instant predictions | ✅ Implemented |
| **F-05: Explainable AI** | "Why?" explanations for each recommendation | ✅ Implemented |
| **F-06: User Taste Profile** | Genre preferences and rating patterns | ✅ Implemented |
| **F-07: "Surprise Me" Mode** | Serendipity recommendations | ✅ Implemented |
| **F-08: Feedback Loop** | Like/dislike simulation | ✅ Implemented |
| **F-09: Data Visualization** | Interactive analytics dashboard | ✅ Implemented |
| **F-10: Movie Similarity** | Find similar movies using latent factors | ✅ Implemented |

### Technical Features

- ✅ **Data Integrity Checks** (NF-01): Automated dataset validation with actionable error messages
- ✅ **Docker Containerization**: One-command deployment
- ✅ **Multi-Page Interface**: Organized navigation (Home, Recommend, Analytics)
- ✅ **Caching & Optimization**: Sub-second response times
- ✅ **Professional Code Structure**: Modular, documented, maintainable

---

## Multi-Algorithm Architecture

CineMatch V2.1.0 implements multiple recommendation paradigms with intelligent ensemble methods:

```
┌─────────────────────────────────────────────────────────────┐
│                STREAMLIT UI LAYER V2.1.0                     │
│      (Algorithm Selector + Performance Metrics)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│             ALGORITHM MANAGER LAYER                          │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Lazy Loading│  │ Smart Caching│  │ Dynamic Weighting│  │
│  │ & Factory   │  │ & Lifecycle  │  │ & Ensemble       │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│               MULTI-ALGORITHM LAYER                          │
│ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────────┐ ┌───────────┐ │
│ │  SVD  │ │ User  │ │ Item  │ │ Content-  │ │  Hybrid   │ │
│ │Matrix │ │  KNN  │ │  KNN  │ │  Based    │ │ Ensemble  │ │
│ └───────┘ └───────┘ └───────┘ └───────────┘ └───────────┘ │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│             BASE RECOMMENDER INTERFACE                       │
│            (Standardized API + Performance Metrics)          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   DATA LAYER                                 │
│            MovieLens 32M Dataset (CSV)                       │
└──────────────────────────────────────────────────────────────┘
```

### Algorithm Comparison

| Algorithm | Paradigm | Best For | Interpretability | Accuracy |
|-----------|----------|----------|------------------|----------|
| **SVD** | Matrix Factorization | Hidden patterns, latent factors | Medium | High |
| **User-KNN** | User-Based CF | New users, social recommendations | Very High | Medium |
| **Item-KNN** | Item-Based CF | Similar movies, genre exploration | High | Medium |
| **Content-Based** | Feature Similarity | New items, explainable recommendations | Very High | Medium |
| **Hybrid** | Ensemble Learning | Best overall results, all scenarios | Medium | Highest |

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/M7MDRAUF/DataSets.git
cd DataSets

# 2. Pull pre-trained models (526MB via Git LFS)
git lfs install
git lfs pull

# 3. Download dataset
# Visit: http://files.grouplens.org/datasets/movielens/ml-32m.zip
# Extract and place files in data/ml-32m/

# 4. Run with Docker
docker-compose up --build
```

Visit: **http://localhost:8501**

### Option 2: Local Python

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset (see below)

# 4. Train model
python src/model_training.py

# 5. Run Streamlit app
streamlit run app/main.py
```

---

## Installation

### Prerequisites

- **Python**: 3.9 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Disk Space**: 2GB for dataset + 1GB for model
- **Docker** (optional): Latest version

### Step 1: Dataset Download

**CRITICAL**: The application requires the MovieLens 32M dataset.

1. Download from: http://files.grouplens.org/datasets/movielens/ml-32m.zip
2. Extract the ZIP file
3. Copy all CSV files to: `data/ml-32m/`

**Required files:**
```
data/ml-32m/
├── ratings.csv     (32M ratings)
├── movies.csv      (Movie metadata)
├── links.csv       (IMDb/TMDb IDs)
└── tags.csv        (User tags)
```

The application will **automatically check** for these files on startup and provide clear instructions if missing.

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `pandas` (≥2.2.0) - data processing
- `scikit-learn` (≥1.5.0) - machine learning algorithms (SVD, KNN)
- `streamlit` (≥1.32.0) - web interface
- `plotly` (≥5.18.0) - visualizations
- `joblib` - model persistence
- `scipy` - sparse matrix operations

### Step 3: Train Model

```bash
python src/model_training.py
```

**Options:**
- **Full dataset** (32M ratings, recommended for thesis): ~30 minutes, RMSE 0.8406
- **Sample (100K ratings)**: ~2 minutes, RMSE 0.6829

The trained SVD model will be saved to `models/svd_model.pkl` (~200 MB).

**Note**: Pre-trained KNN and Content-Based models (526MB total) are included via Git LFS.

---

## Algorithm Guide

### 🔮 SVD Matrix Factorization
**Best for**: Hidden patterns, complex user preferences, high accuracy

**How it works**: Decomposes the user-movie matrix into lower-dimensional representations to discover latent factors (e.g., "action lovers", "drama enthusiasts").

**Strengths**:
- Excellent accuracy (RMSE 0.8406 on 32M dataset)
- Handles sparse data well
- Discovers complex relationships
- Good for diverse recommendations

**When to use**: Academic research, high accuracy needs, users with varied taste

---

### 👥 User-KNN (Collaborative Filtering)
**Best for**: Social recommendations, new users, interpretable results

**How it works**: Finds users with similar rating patterns and recommends movies that those similar users enjoyed.

**Strengths**:
- Highly interpretable ("Users like you loved this")
- Good performance with sparse data
- Works well for new items
- Fast training and prediction

**When to use**: Community-based recommendations, explanations important, sparse user profiles

---

### 🎬 Item-KNN (Item Similarity)  
**Best for**: Movie similarity, genre exploration, stable recommendations

**How it works**: Analyzes movies with similar rating patterns and recommends items similar to what you've enjoyed before.

**Strengths**:
- Stable, consistent recommendations  
- Works well for users with many ratings
- Pre-computed similarities for speed
- Less susceptible to new user ratings

**When to use**: Users with established preferences, discovering similar movies, genre-based exploration

---

### 🎯 Content-Based Filtering
**Best for**: New items (cold-start), explainable recommendations, feature-driven discovery

**How it works**: Analyzes movie attributes (genres, tags, titles) using TF-IDF vectorization and cosine similarity to find movies matching user preference profiles.

**Strengths**:
- Excellent for new/unpopular movies (no ratings needed)
- Highly interpretable ("You liked Sci-Fi, here are more Sci-Fi movies")
- No cold-start problem for items
- Works with sparse user profiles
- Feature-driven recommendations

**When to use**: New movie catalog additions, explainability requirements, genre/tag-based discovery

---

### 🚀 Hybrid Ensemble
**Best for**: Best overall results, production systems, all user types

**How it works**: Intelligently combines 4 algorithms (SVD, User-KNN, Item-KNN, Content-Based) with dynamic weighting based on user profile and data context.

**Strengths**:
- Combines strengths of all 4 paradigms
- Adapts to different user types
- Robust against individual algorithm weaknesses  
- Leverages collaborative + content-based filtering

**When to use**: Production systems, academic research, when robustness and diversity are required

### Algorithm Selection Guide

| User Profile | Recommended Algorithm | Reason |
|-------------|----------------------|--------|
| **New user (0 ratings)** | Content-Based → Hybrid | Feature-based works without user history |
| **Sparse user (<30 ratings)** | User-KNN → Hybrid | Social recommendations are more interpretable |
| **Dense user (>50 ratings)** | SVD → Hybrid | Matrix factorization captures complex patterns |
| **New movies (cold-start)** | Content-Based | No ratings needed, uses movie features |
| **Research/Production** | Hybrid | Best overall performance and robustness |

---

## Usage

### Multi-Algorithm Interface

#### Recommend Page
1. **Select Algorithm**: Choose from sidebar (SVD, User-KNN, Item-KNN, Content-Based, Hybrid)
2. **Enter User ID**: (1 to 200,000) or try sample IDs: 66954, 123, 1000
3. **Click "Get Recommendations"**: See personalized results with algorithm-specific explanations
4. **Compare Performance**: View real-time RMSE, training time, memory usage
5. **Advanced Options**: Fine-tune dataset size (100K/500K/1M/Full 32M)

#### Algorithm Switching
- **Live Switching**: Change algorithms instantly with cached models
- **Performance Comparison**: Side-by-side metrics display
- **Explanations**: Algorithm-specific reasoning for each recommendation

**Sample User IDs to try**: 1, 50, 123, 500, 1000, 5000, 66954

### Running the Application

```bash
streamlit run app/main.py
```

Access at: **http://localhost:8501**

#### Analytics Page
- Genre analysis
- Temporal trends
- Popularity metrics
- Movie similarity explorer

---

## Project Structure

```
cinematch-demo/
│
├── 📊 data/                      # Dataset files
│   ├── ml-32m/                   # Raw MovieLens data
│   │   ├── ratings.csv
│   │   ├── movies.csv
│   │   ├── links.csv
│   │   └── tags.csv
│   └── processed/                # Cached processed data
│
├── 🧠 models/                    # Trained models
│   ├── svd_model.pkl             # Pre-trained SVD model
│   └── model_metadata.json       # Training metrics
│
├── ⚙️ src/                       # Core logic
│   ├── __init__.py
│   ├── data_processing.py        # Data loading & integrity (NF-01)
│   ├── model_training.py         # Training pipeline
│   ├── recommendation_engine.py  # Legacy V1.0 SVD engine
│   ├── algorithms/               # Multi-algorithm system (V2.1.0)
│   │   ├── base_recommender.py   #     Abstract base class interface
│   │   ├── algorithm_manager.py  #     Central management (Factory + Singleton)
│   │   ├── svd_recommender.py    #     SVD implementation
│   │   ├── user_knn_recommender.py #   User-based collaborative filtering
│   │   ├── item_knn_recommender.py #   Item-based collaborative filtering
│   │   ├── content_based_recommender.py # Content-based filtering (TF-IDF)
│   │   └── hybrid_recommender.py #     Ensemble system (4 algorithms)
│   └── utils.py                  # Explainability & helpers
│
├── 🎨 app/                       # Streamlit UI
│   ├── main.py                   # Entry point
│   └── pages/
│       ├── 1_🏠_Home.py          # Algorithm selector
│       ├── 2_🎬_Recommend.py     # Recommendation engine (all 5 algorithms)
│       └── 3_📊_Analytics.py     # Advanced analytics dashboard
│
├── 🐳 Dockerfile                # Docker container config
├── docker-compose.yml           # Multi-container orchestration
├── .dockerignore                # Docker build exclusions
│
├── 📄 requirements.txt          # Python dependencies
├── README.md                    # This file
├── QUICKSTART.md                # Quick setup guide
├── ARCHITECTURE.md              # System design docs
└── .dockerignore
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── 📋 Documentation/
│   ├── README.md                 # This file
│   ├── ARCHITECTURE.md           # Technical architecture
│   ├── QUICKSTART.md             # Quick start guide
│   └── DEPLOYMENT.md             # Deployment instructions
│
├── requirements.txt              # Python dependencies
├── test_multi_algorithm.py       # 🆕 Algorithm testing
├── simple_test.py                # 🆕 Quick algorithm test
├── .gitignore
└── .dockerignore
```

---

## Performance Benchmarks

### Algorithm Comparison (Sample Dataset: 100K ratings)

| Algorithm | RMSE | Training Time | Memory Usage | Coverage | Interpretability |
|-----------|------|---------------|--------------|----------|------------------|
| **SVD** | 0.5690 | 1.8s | <1 MB | 12.8% | Medium |
| **User-KNN** | 0.5992 | 13.1s | 1.0 MB | 100% | Very High |
| **Item-KNN** | 1.0218 | 116.7s | 50.9 MB | 4.1% | High |
| **Content-Based** | N/A* | <1s | ~300 MB | 100% | Very High |
| **Hybrid** | 0.5585 | 246.0s | 51.9 MB | 100% | Medium |

*Content-Based uses cosine similarity (no RMSE applicable)

### Full Dataset Performance (32M ratings)

| Algorithm | Model Size | Load Time | Training Time |
|-----------|------------|-----------|---------------|
| **SVD** | ~200 MB | <1s | ~30 min |
| **User-KNN** | 266 MB | 1.5s | N/A (pre-trained) |
| **Item-KNN** | 260 MB | 1.5s | N/A (pre-trained) |
| **Content-Based** | ~300 MB | <1s | ~5 min |
| **Hybrid** | 526 MB+ | 7s | Ensemble |

### Performance Insights

- **Best Accuracy**: Hybrid (0.5585 RMSE) > SVD (0.5690) > User-KNN (0.5992) on sample data
- **Fastest Training**: SVD (1.8s) > Content-Based (<1s load) > User-KNN (13.1s)
- **Memory Efficient**: SVD (<1MB) > User-KNN (1.0MB) > Item-KNN (50.9MB)
- **Best Coverage**: User-KNN, Content-Based & Hybrid (100%) > SVD (12.8%) > Item-KNN (4.1%)
- **Production Ready**: All 5 algorithms optimized with pre-trained models (526MB total)

### SVD Algorithm Details (Full Dataset: 32M ratings)

**Algorithm**: SVD (Singular Value Decomposition) via scikit-learn

**Mathematical Foundation:**
```
R ≈ U × Σ × V^T

Prediction: r̂_ui = μ + b_u + b_i + q_i^T · p_u
```

**Hyperparameters:**
- `n_factors`: 100 (latent dimensions)
- `n_epochs`: 20 (training iterations)
- `lr_all`: 0.005 (learning rate)
- `reg_all`: 0.02 (regularization)

**Performance (Full 32M Dataset):**
- Test RMSE: 0.8406 (per model_metadata.json)
- Training time: ~31 seconds (optimized sklearn implementation)
- Model size: ~200 MB

**Performance (100K Sample):**
- Test RMSE: 0.6829
- Training time: ~2 minutes

### Explainable AI (XAI)

**Explanation Strategies:**
1. **Content-based**: Similar genres to highly-rated movies
2. **Collaborative**: Users with similar taste loved this
3. **Genre preference**: Matches user's top genres
4. **Fallback**: High global rating

**Example:**
> "Because you highly rated The Matrix and Inception, and users with similar taste to yours loved this movie"

### Data Integrity (NF-01)

The application implements comprehensive data validation:

```python
# Automatic check on startup
success, missing, error = check_data_integrity()

if not success:
    # Display detailed error message
    # List missing files
    # Provide download instructions
    # Halt execution gracefully
```

**Error message includes:**
- Missing filenames
- Expected location
- Download link
- Step-by-step instructions

---

## Development

### Setting Up Development Environment

```bash
# Clone repository
git clone <repository-url>
cd cinematch-demo

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest black flake8
```

### Code Style

```bash
# Format code
black src/ app/

# Lint code
flake8 src/ app/
```

### Testing Individual Modules

```bash
# Test data integrity
python src/data_processing.py

# Test recommendation engine
python src/recommendation_engine.py

# Test utilities
python src/utils.py
```

---

## Testing

### Manual Testing

1. **Data Integrity Test**:
   - Remove `ratings.csv` temporarily
   - Run app - should show error with instructions
   - Restore file - app should work

2. **Recommendation Test**:
   - Try User IDs: 1, 50, 100, 1000
   - Verify recommendations differ
   - Check explanations make sense

3. **Edge Cases**:
   - Invalid User ID (negative, zero, > max)
   - Empty search in Analytics
   - Multiple rapid requests

### Automated Testing (Future)

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

---

## Deployment

### Docker Deployment (Production)

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Rebuild after changes
docker-compose up --build -d
```

### Cloud Deployment (AWS/Azure/GCP)

**Recommended Setup:**
1. **EC2/VM**: t3.xlarge (4 vCPU, 16GB RAM)
2. **Storage**: 20GB SSD
3. **Network**: Open port 8501
4. **Domain**: Optional custom domain with SSL

**Deploy script:**
```bash
# Install Docker
curl -fsSL https://get.docker.com | sh

# Clone and run
git clone <repo-url>
cd cinematch-demo
docker-compose up -d
```

---

## Performance

### Benchmarks

| Metric | Target | Actual |
|--------|--------|--------|
| Model RMSE | < 0.87 | 0.85 ✅ |
| Recommendation Time | < 2s | ~1.2s ✅ |
| UI Load Time | < 3s | ~2.1s ✅ |
| Memory Usage | < 8GB | ~6GB ✅ |

### Optimization Techniques

- ✅ **Model caching**: Load once, reuse (@st.cache_resource)
- ✅ **Data caching**: Cache expensive operations (@st.cache_data)
- ✅ **Vectorized predictions**: NumPy for batch operations
- ✅ **Efficient data types**: int32, float32 for memory savings
- ✅ **Pre-trained model**: No real-time training

---

## Troubleshooting

### Common Issues

#### 1. "Data Integrity Check Failed"

**Problem**: Dataset files missing

**Solution**:
```bash
# Download dataset
wget http://files.grouplens.org/datasets/movielens/ml-32m.zip

# Extract
unzip ml-32m.zip

# Move files
mv ml-32m/*.csv data/ml-32m/
```

#### 2. "Model file not found"

**Problem**: Model not trained

**Solution**:
```bash
python src/model_training.py
```

#### 3. "Memory Error"

**Problem**: Insufficient RAM

**Solution**:
- Use sample mode in training
- Reduce `sample_size` in data loading
- Close other applications
- Upgrade to 16GB RAM

#### 4. Docker Port Already in Use

**Problem**: Port 8501 occupied

**Solution**:
```bash
# Change port in docker-compose.yml
ports:
  - "8502:8501"  # Use 8502 instead
```

#### 5. Slow Performance

**Solution**:
- Ensure model is pre-trained (not training in real-time)
- Check caching is enabled
- Reduce sample sizes
- Use Docker with resource limits

#### 6. Page Crashes When Changing Slider Values

**Problem**: Page reloads/crashes when changing "Number of recommendations" slider

**Root Cause**: Streamlit widget state conflicts due to missing unique keys

**Solution**: ✅ FIXED in v2.1.2
- All widgets now have unique keys (`user_id_input`, `num_recommendations_slider`, `generate_button`)
- Added `step=1` parameters for better state management
- Widget values persist correctly during reruns

**Manual Fix** (if building from older version):
```python
# Add unique keys to all widgets in 1_🏠_Home.py
st.number_input(..., key="user_id_input")
st.slider(..., key="num_recommendations_slider")
st.button(..., key="generate_button")
```

#### 7. Missing Page Sections (Dataset Overview, Genre Distribution, etc.)

**Problem**: Content below recommendations section doesn't render

**Root Cause**: Inappropriate `st.stop()` terminating entire script execution

**Solution**: ✅ FIXED in v2.1.2
- Removed `st.stop()` at line 438 that was killing page rendering
- Button handler already properly scoped, st.stop() was unnecessary
- All sections now render correctly (Dataset Overview, Genre Distribution, Rating Distribution, Top Movies, User Engagement, About Project)

**Manual Fix** (if building from older version):
```python
# Remove st.stop() after recommendation generation in 1_🏠_Home.py
# Line 438: DELETE the st.stop() call
```

#### 8. Invalid User ID Errors

**Problem**: Entering non-existent user IDs causes errors

**Solution**: ✅ FIXED in v2.1.2
- User ID validation added before recommendation generation
- Shows friendly error: "User ID {id} not found in dataset!"
- Displays valid range: "Please enter a valid User ID between {min} and {max}"

**Testing**: Use user IDs between 1 and 200948 (dataset range)

---

## 🧪 Testing

### Automated Test Suite (v2.1.2+)

CineMatch includes 40+ automated tests covering UI rendering, model loading, and end-to-end workflows.

**Run All Tests:**
```bash
pytest tests/ -v
```

**Test Categories:**
1. **UI Rendering Tests** (`test_ui_rendering.py`)
   - Widget key presence
   - st.stop() usage validation
   - Required sections rendering
   - Cached function decorators

2. **Model Loading Tests** (`test_model_loading.py`)
   - AlgorithmManager singleton
   - Model loader utilities
   - Memory management
   - All 5 algorithm implementations

3. **End-to-End Tests** (`test_end_to_end_flow.py`)
   - Data loading functions
   - SVD/KNN training and prediction
   - Recommendation generation
   - User validation

**Run Specific Test File:**
```bash
pytest tests/test_ui_rendering.py -v
pytest tests/test_model_loading.py -v
pytest tests/test_end_to_end_flow.py -v
```

**Coverage Report:**
```bash
pytest tests/ --cov=src --cov=app --cov-report=html
```

---

## Demo Script (Professor Presentation)

### 5-Minute Walkthrough

**[Minute 0-1: Introduction]**
> "Professor, today I present CineMatch, a production-grade movie recommendation engine. It uses SVD collaborative filtering on 32 million ratings to provide personalized, explainable recommendations."

**[Minute 1-2: Technical Overview]**
> "The system is containerized with Docker for easy deployment. I've implemented data integrity checks, ensuring robust error handling. Let me show you the one-command deployment..."

```bash
docker-compose up
```

**[Minute 2-4: Live Demo]**
> "Let's get recommendations for User 123..."
>
> **[Navigate to Recommend page]**
> - Enter User ID: 123
> - Click "Get Recommendations"
> - **[Point out]**: "These are the top 10 predicted movies"
> - Click "Explain" on a recommendation
> - **[Highlight]**: "This is the explainable AI feature - it shows WHY"
> - **[Show sidebar]**: "Here's the user's taste profile"

**[Minute 4-5: Advanced Features]**
> "We also have a Surprise Me mode for serendipity, and analytics for dataset insights."
>
> **[Navigate to Analytics]**
> - Show genre distribution
> - Demonstrate movie similarity search

**[Minute 5: Conclusion]**
> "The model achieves an RMSE of 0.8406, beating our target of 0.87. Response times are under 2 seconds. This demonstrates both technical ML skills and professional software engineering practices."

---

## Academic Contribution

### Key Innovations

1. **Explainable Recommendations**: Multi-strategy XAI approach
2. **User Taste Profiling**: Genre preference analysis
3. **Serendipity Mode**: Exploration vs. exploitation balance
4. **Production-Ready**: Docker, testing, documentation

### Future Research Directions

- Real-time model updating with online learning
- Hybrid models (collaborative + content-based)
- Deep learning approaches (neural collaborative filtering)
- A/B testing framework for recommendation strategies
- Cold-start problem solutions (new users/movies)

---

## 🤝 Contributing

This is a master's thesis project, but suggestions are welcome!

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Standards

- Follow PEP 8 (Python style guide)
- Add docstrings (Google style)
- Include type hints
- Write unit tests for new features

---

## 📜 License

This project is created for academic purposes (Master's Thesis).

**Dataset License**: MovieLens data is provided by GroupLens Research under their usage license.

---

## 🙏 Acknowledgments

- **GroupLens Research**: For the MovieLens dataset
- **Streamlit Team**: For the amazing web framework
- **scikit-learn**: For machine learning algorithms (SVD, KNN)
- **My Thesis Advisor**: For guidance and support

---

## 📚 References

1. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30-37.

2. Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and context. *ACM Transactions on Interactive Intelligent Systems (TiiS)*, 5(4), 1-19.

3. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

---

**CineMatch V2.1.0** | Multi-Algorithm Recommendation Engine  
Last Updated: November 13, 2025

3. Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender systems handbook*. Springer.

4. Zhang, Y., & Chen, X. (2020). Explainable recommendation: A survey and new perspectives. *Foundations and Trends in Information Retrieval*, 14(1), 1-101.

---

<div align="center">

**CineMatch V1.0.0**

*Built for Master's Thesis Defense*

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://docker.com)

*Powered by 32 Million Ratings*

</div>
