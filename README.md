# 🎬 CineMatch V2.0

**Multi-Algorithm Recommendation Engine with Explainable AI**

A production-grade collaborative filtering recommendation system built for a master's thesis demonstration. CineMatch V2.0 now features **multiple algorithms** including SVD matrix factorization, User KNN, Item KNN, and intelligent Hybrid systems on the MovieLens 32M dataset to provide personalized, explainable movie recommendations through an interactive Streamlit web interface.

![CineMatch Banner](https://img.shields.io/badge/CineMatch-V2.0-red?style=for-the-badge&logo=film)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker)
![Algorithms](https://img.shields.io/badge/Algorithms-4%20Types-green?style=for-the-badge)

## 🎯 Project Status

**Last Updated**: November 7, 2025  
**Version**: V2.0 Multi-Algorithm System
**Algorithms**: ✅ SVD + User KNN + Item KNN + Hybrid
**Model Accuracy**: ✅ SVD RMSE 0.5690, Hybrid RMSE 0.5585
**All Tests Passed**: ✅ All algorithms working correctly
**Application**: ✅ Enhanced UI with algorithm selection
**Cloud Deployment**: ✅ https://m7md007.streamlit.app
**Status**: Production-ready with multi-algorithm support

---

## 🚀 What's New in V2.0

### **Multi-Algorithm Support**
- **🔮 SVD Matrix Factorization** - Latent factor modeling for high accuracy
- **👥 User KNN** - Find users with similar taste for social recommendations  
- **🎬 Item KNN** - Discover movies similar to your favorites
- **🚀 Hybrid System** - Intelligent ensemble combining all algorithms

### **Professional UI Enhancement**
- **Algorithm Selector** - Choose your preferred recommendation approach
- **Live Performance Metrics** - Real-time RMSE, training time, memory usage
- **Advanced Options** - Fine-tune algorithm parameters
- **Enhanced Explanations** - Algorithm-specific reasoning for recommendations

### **Academic Features**
- **Algorithm Comparison** - Side-by-side performance analysis
- **Explainable AI** - Transparent recommendation reasoning
- **Professional Architecture** - Abstract classes, factory patterns, intelligent caching
- **Research-Grade Implementation** - Multiple recommendation paradigms

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

CineMatch V2.0 addresses the "analysis paralysis" problem in movie selection by providing **multiple recommendation approaches** backed by explainable AI. Built on 32 million user ratings, it demonstrates practical application of **ensemble learning** and **multi-paradigm collaborative filtering** while maintaining production-level code quality and user experience.

### Success Metrics V2.0

- ✅ **Multi-Algorithm Accuracy**: SVD RMSE 0.5690, Hybrid RMSE 0.5585  
- ✅ **Response Time**: < 2 seconds for all algorithms
- ✅ **Algorithm Diversity**: 4 different recommendation paradigms
- ✅ **Professional UI**: Algorithm selection with live metrics
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

CineMatch V2.0 implements multiple recommendation paradigms with intelligent ensemble methods:

```
┌─────────────────────────────────────────────────────────────┐
│                STREAMLIT UI LAYER V2.0                       │
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
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐│
│ │ SVD Matrix  │ │ User KNN    │ │ Item KNN    │ │ Hybrid  ││
│ │ Factorization│ │ Collaborative│ │ Collaborative│ │ Ensemble││
│ └─────────────┘ └─────────────┘ └─────────────┘ └─────────┘│
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
| **User KNN** | User-Based CF | New users, social recommendations | Very High | Medium |
| **Item KNN** | Item-Based CF | Similar movies, genre exploration | High | Medium |
| **Hybrid** | Ensemble Learning | Best overall results, all scenarios | Medium | Highest |

---

## Quick Start

### Option 1: Multi-Algorithm V2.0

```bash
# 1. Clone repository
git clone <repository-url>
cd cinematch-demo

# 2. Download dataset (see below)
# Place files in data/ml-32m/

# 3. Test multi-algorithm system
python simple_test.py

# 4. Launch enhanced UI
streamlit run app/pages/2_🎬_Recommend_V2.py
```

### Option 2: Classic V1.0 (Original)

# 4. Run with Docker
docker-compose up --build
```

Visit: http://localhost:8501

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
- `pandas` (data processing)
- `scikit-surprise` (collaborative filtering)
- `streamlit` (web interface)
- `plotly` (visualizations)
- `joblib` (model persistence)

### Step 3: Train Model

```bash
python src/model_training.py
```

**Options:**
- **Full dataset** (recommended for thesis): ~30 minutes, RMSE < 0.87
- **Sample (1M ratings)**: ~2 minutes, RMSE ~0.90

The trained model will be saved to `models/svd_model.pkl` (~200 MB).

---

## Algorithm Guide

### 🔮 SVD Matrix Factorization
**Best for**: Hidden patterns, complex user preferences, high accuracy

**How it works**: Decomposes the user-movie matrix into lower-dimensional representations to discover latent factors (e.g., "action lovers", "drama enthusiasts").

**Strengths**:
- Excellent accuracy (RMSE ~0.57)
- Handles sparse data well
- Discovers complex relationships
- Good for diverse recommendations

**When to use**: Academic research, high accuracy needs, users with varied taste

---

### 👥 User KNN (Collaborative Filtering)
**Best for**: Social recommendations, new users, interpretable results

**How it works**: Finds users with similar rating patterns and recommends movies that those similar users enjoyed.

**Strengths**:
- Highly interpretable ("Users like you loved this")
- Good performance with sparse data
- Works well for new items
- Fast training and prediction

**When to use**: Community-based recommendations, explanations important, sparse user profiles

---

### 🎬 Item KNN (Content-Based)  
**Best for**: Movie similarity, genre exploration, stable recommendations

**How it works**: Analyzes movies with similar rating patterns and recommends items similar to what you've enjoyed before.

**Strengths**:
- Stable, consistent recommendations  
- Works well for users with many ratings
- Pre-computed similarities for speed
- Less susceptible to new user ratings

**When to use**: Users with established preferences, discovering similar movies, genre-based exploration

---

### 🚀 Hybrid Ensemble
**Best for**: Best overall results, production systems, all user types

**How it works**: Intelligently combines SVD, User KNN, and Item KNN with dynamic weighting based on user profile and data context.

**Strengths**:
- Highest accuracy (RMSE ~0.56)
- Adapts to different user types
- Robust against individual algorithm weaknesses  
- Combines multiple recommendation paradigms

**When to use**: Production systems, academic research, when highest accuracy is required

### Algorithm Selection Guide

| User Profile | Recommended Algorithm | Reason |
|-------------|----------------------|--------|
| **New user (0 ratings)** | Hybrid → Item KNN | Content-based fallbacks work better |
| **Sparse user (<30 ratings)** | User KNN → Hybrid | Social recommendations are more interpretable |
| **Dense user (>50 ratings)** | SVD → Hybrid | Matrix factorization captures complex patterns |
| **Research/Production** | Hybrid | Best overall performance and robustness |

---

## Usage

### Multi-Algorithm V2.0 Interface

#### Enhanced Recommend Page
1. **Select Algorithm**: Choose from sidebar (SVD, User KNN, Item KNN, Hybrid)
2. **Enter User ID**: (1 to 200,000) or try sample IDs: 66954, 123, 1000
3. **Click "Get Recommendations"**: See personalized results with algorithm-specific explanations
4. **Compare Performance**: View real-time RMSE, training time, memory usage
5. **Advanced Options**: Fine-tune algorithm parameters

#### Algorithm Switching
- **Live Switching**: Change algorithms instantly with cached models
- **Performance Comparison**: Side-by-side metrics display
- **Explanations**: Algorithm-specific reasoning for each recommendation

### Classic V1.0 Interface

#### Recommend Page (Original)
1. Enter a User ID (1 to 200,000)
2. Click "Get Recommendations"
3. View top 10 SVD-based recommendations
4. Click "Explain" to see why each movie is recommended
5. Try "Surprise Me" for serendipity mode

**Sample User IDs to try**: 1, 50, 123, 500, 1000, 5000

### Running the Application

#### V2.0 Multi-Algorithm
```bash
streamlit run app/pages/2_🎬_Recommend_V2.py
```

#### V1.0 Classic
```bash
streamlit run app/main.py
```

Access at: http://localhost:8501

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
│   ├── recommendation_engine.py  # Original SVD engine
│   ├── algorithms/               # 🆕 Multi-algorithm system
│   │   ├── base_recommender.py   #     Abstract base class
│   │   ├── algorithm_manager.py  #     Central management
│   │   ├── svd_recommender.py    #     SVD implementation
│   │   ├── user_knn_recommender.py #   User-based KNN
│   │   ├── item_knn_recommender.py #   Item-based KNN
│   │   └── hybrid_recommender.py #     Ensemble system
│   └── utils.py                  # Explainability & helpers
│
├── 🎨 app/                       # Streamlit UI
│   ├── main.py                   # Entry point
│   └── pages/
│       ├── 1_🏠_Home.py
│       ├── 2_🎬_Recommend.py     # Classic V1.0
│       ├── 2_🎬_Recommend_V2.py  # 🆕 Multi-algorithm V2.0
│       └── 3_📊_Analytics.py
│
├── 🐳 Docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── 📋 Documentation/
│   ├── README.md                 # This file
│   ├── PRD.md                    # Product requirements
│   ├── ARCHITECTURE.md           # Technical architecture
│   └── ALGORITHM_ARCHITECTURE_PLAN.md # 🆕 Multi-algorithm design
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
| **User KNN** | 0.5992 | 13.1s | 1.0 MB | 100% | Very High |
| **Item KNN** | 1.0218 | 116.7s | 50.9 MB | 4.1% | High |
| **Hybrid** | 0.5585 | 246.0s | 51.9 MB | 100% | Medium |

### Performance Insights

- **Best Accuracy**: Hybrid (0.5585 RMSE) > SVD (0.5690) > User KNN (0.5992)
- **Fastest Training**: SVD (1.8s) > User KNN (13.1s) > Item KNN (116.7s)
- **Memory Efficient**: SVD (<1MB) > User KNN (1.0MB) > Item KNN (50.9MB)
- **Best Coverage**: User KNN & Hybrid (100%) > SVD (12.8%) > Item KNN (4.1%)

### Original V1.0 Performance (Full Dataset: 32M ratings)

**Algorithm**: SVD (Singular Value Decomposition) via scikit-surprise

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

**Performance:**
- Training RMSE: ~0.83
- Test RMSE: < 0.87 ✅
- Training time: 15-30 minutes (full dataset)

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
- **scikit-surprise**: For collaborative filtering implementation
- **My Thesis Advisor**: For guidance and support

---

## Contact

**Author**: [Your Name]  
**Email**: [Your Email]  
**University**: [Your University]  
**Program**: Master's in [Your Program]  
**Year**: 2025

---

## References

1. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30-37.

2. Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and context. *ACM Transactions on Interactive Intelligent Systems (TiiS)*, 5(4), 1-19.

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
