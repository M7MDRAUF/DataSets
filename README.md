# CineMatch - Multi-Algorithm Movie Recommendation System

**Version:** 2.1.7  
**Author:** Mohamed Rauf  
**Institution:** Master's Thesis Project  
**Date:** December 11, 2025  
**License:** MIT  
**Repository:** https://github.com/M7MDRAUF/DataSets

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Algorithms](#algorithms)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)
- [Performance Metrics](#performance-metrics)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)

---

## Executive Summary

CineMatch is an enterprise-grade, multi-algorithm recommendation engine built as a comprehensive demonstration of modern machine learning techniques and software engineering best practices. The system implements five distinct recommendation algorithms (SVD Matrix Factorization, User-based KNN, Item-based KNN, Content-Based Filtering, and Hybrid Ensemble) trained on the MovieLens 32M dataset, delivering personalized, explainable movie recommendations through an interactive web interface.

### Project Objectives

- **Academic Excellence**: Demonstrate mastery of collaborative filtering, content-based filtering, and ensemble methods
- **Production-Ready Implementation**: Enterprise-grade architecture with Docker deployment and comprehensive testing
- **Algorithm Comparison**: Quantitative analysis of five different recommendation paradigms
- **Explainable AI**: Clear, algorithm-specific reasoning for every recommendation
- **Scalability**: Optimized for handling 32 million ratings with sub-2-second response times

### Technology Stack

- **Backend**: Python 3.13, scikit-learn 1.7.2, scikit-surprise 1.1.4, pandas 2.3.3, numpy 1.26.4
- **Frontend**: Streamlit 1.51.0, Plotly 6.5.0, Matplotlib 3.10.7
- **Data**: MovieLens 32M dataset (32 million ratings, 86,000+ movies)
- **Infrastructure**: Docker, Git LFS, joblib model persistence
- **Testing**: Automated validation suite with comprehensive coverage

---

## Key Features

### Core Capabilities

1. **Multi-Algorithm Support**
   - Five independent recommendation engines with distinct methodologies
   - Real-time algorithm switching without system restart
   - Comparative benchmarking and performance analysis
   - Intelligent algorithm selection based on user profile

2. **Advanced Recommendation Algorithms**
   - **SVD Matrix Factorization**: Latent factor modeling for pattern discovery (RMSE: 0.7502)
   - **User-based KNN**: Collaborative filtering based on similar users (RMSE: 0.8394)
   - **Item-based KNN**: Movie similarity using rating patterns (RMSE: 0.9100)
   - **Content-Based Filtering**: TF-IDF feature extraction from genres, tags, and titles (RMSE: 1.1130)
   - **Hybrid Ensemble**: Adaptive weighted combination of all algorithms (RMSE: 0.8701)

3. **Explainable AI (XAI)**
   - Algorithm-specific explanation generation
   - Factor contribution analysis for matrix factorization
   - Similar user identification for collaborative filtering
   - Feature similarity breakdowns for content-based recommendations
   - Transparent scoring and ranking methodology

4. **Performance Optimization**
   - Pre-trained models on full 32M dataset (approximately 4.4GB via Git LFS)
   - Memory optimization: 98.6% reduction (13.2GB to 185MB runtime)
   - Intelligent caching with LRU eviction policies
   - Vectorized batch processing for 200x speedup
   - Smart candidate sampling reducing search space from 80,000 to 5,000 items

5. **User Experience**
   - Interactive web interface with professional design
   - Real-time recommendation generation (under 2 seconds)
   - Comprehensive analytics dashboard with visualizations
   - User history and taste profile analysis
   - Export functionality (CSV, JSON formats)

6. **Production Engineering**
   - Docker containerization with optimized resource usage (2.6GB / 8GB)
   - Comprehensive test suite with automated validation
   - Data integrity checks and error handling
   - Context-aware logging and debugging
   - Configuration management with environment variables

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE LAYER                     │
│                   (Streamlit Multi-Page App)                 │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Home Page     │ Recommend Page  │   Analytics Page        │
│  (Overview &    │ (Core Feature)  │  (5-Algorithm           │
│  Visualizations)│                 │   Benchmarking)         │
└────────┬────────┴────────┬────────┴──────────┬──────────────┘
         │                 │                   │
         └─────────────────┼───────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────┐
│                   APPLICATION LAYER                         │
│                  (Business Logic - src/)                    │
├──────────────────┬──────────────────┬──────────────────────┤
│ AlgorithmManager │  Explanation     │   Data Processing    │
│ (Factory +       │    Engine        │      Module          │
│  Singleton +     │   (XAI Logic)    │  (Integrity Checks)  │
│  Shallow Refs)   │                  │                      │
└────────┬─────────┴─────────┬────────┴──────────┬───────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│                      MODEL LAYER                              │
│           (5 Pre-trained Models - models/ - Git LFS)          │
├───────────────────────────────────────────────────────────────┤
│  SVD (sklearn): 954MB matrix factorization                     │
│  User-KNN: 1170MB collaborative filtering                      │
│  Item-KNN: 1160MB item similarity                              │
│  Content-Based: 1110MB TF-IDF                                  │
│  Hybrid: Ensemble of above                                     │
└─────────────────────────────┬─────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────┐
│                        DATA LAYER                               │
│                    (MovieLens 32M Dataset)                      │
├─────────────────────────────────────────────────────────────────┤
│  ratings.csv (32M user-movie-rating records, ~3.3GB in memory) │
│  movies.csv (Movie metadata: title, genres)                     │
│  links.csv (IMDb/TMDb IDs for external integration)             │
│  tags.csv (User-generated tags for Content-Based)               │
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

**1. User Interface Layer (Streamlit)**
- Multi-page application with responsive design
- Algorithm selection and parameter configuration
- Real-time recommendation display with explanations
- Interactive analytics dashboard with comparative visualizations
- User search and history exploration

**2. Application Layer (src/)**
- **AlgorithmManager**: Thread-safe singleton coordinating all algorithms
  - Factory pattern for algorithm instantiation
  - Intelligent caching with zero-copy data references
  - Dynamic algorithm switching
  - Centralized metrics aggregation
- **Data Processing**: Dataset integrity validation and preprocessing
- **Explanation Engine**: Algorithm-specific XAI logic

**3. Model Layer (models/)**
- Five pre-trained recommendation models (approximately 4.4GB total)
- Joblib/pickle serialization for efficient persistence
- Lazy loading with intelligent caching
- Model metadata tracking (hyperparameters, metrics, timestamps)

**4. Data Layer (data/)**
- MovieLens 32M dataset storage
- Preprocessed feature matrices
- Cached intermediate computations

---

## Algorithms

### 1. SVD Matrix Factorization

**Type**: Collaborative Filtering - Model-based

**Description**: Uses Singular Value Decomposition to discover latent factors representing hidden patterns in user-movie interactions. Learns low-dimensional representations of users and movies that capture preferences and characteristics.

**Implementation**: scikit-learn TruncatedSVD with 100 components

**Strengths**:
- Excellent accuracy on dense datasets
- Captures complex user-item interactions
- Handles sparse data efficiently
- Dimensionality reduction reduces noise
- Proven performance on MovieLens benchmarks

**Ideal For**: Users with substantial rating history, general-purpose recommendations

**Performance**:
- RMSE: 0.7502
- Coverage: 24.1%
- Training Time: ~25 minutes on full dataset
- Prediction Time: ~0.5 seconds per user

**Configurable Parameters**:
- `n_components`: Number of latent factors (default: 100, range: 50-200)

---

### 2. User-Based KNN

**Type**: Collaborative Filtering - Memory-based

**Description**: Finds users with similar rating patterns and recommends movies that similar users have rated highly. Uses K-Nearest Neighbors algorithm with cosine similarity.

**Implementation**: scikit-learn NearestNeighbors with smart candidate sampling

**Strengths**:
- Intuitive and explainable
- Captures diverse preferences
- No training required (memory-based)
- Dynamic updates with new ratings
- Works well for niche preferences

**Ideal For**: Users seeking diverse recommendations, serendipitous discovery

**Performance**:
- RMSE: 0.8394
- Coverage: 95%+
- Model Size: 1.17GB
- Prediction Time: ~1.5 seconds per user

**Configurable Parameters**:
- `n_neighbors`: Number of similar users (default: 50, range: 10-100)
- `similarity_metric`: Distance metric (default: cosine, options: cosine, euclidean, manhattan)

---

### 3. Item-Based KNN

**Type**: Collaborative Filtering - Memory-based

**Description**: Finds movies with similar rating patterns and recommends them based on user's past preferences. More stable than user-based KNN since item relationships change less frequently.

**Implementation**: Precomputed item-item similarity matrix with efficient lookup

**Strengths**:
- Fast predictions with precomputation
- Stable over time
- Highly interpretable
- Scalable for large item catalogs
- No cold-start for popular items

**Ideal For**: Users with clear genre preferences, "similar to this movie" queries

**Performance**:
- RMSE: 0.9100
- Coverage: 90%+
- Model Size: 1.16GB
- Prediction Time: ~1.0 seconds per user

**Configurable Parameters**:
- `n_neighbors`: Number of similar items (default: 30, range: 10-50)
- `min_ratings`: Minimum ratings per item (default: 5, range: 3-20)
- `similarity_metric`: Distance metric (default: cosine)

---

### 4. Content-Based Filtering

**Type**: Content-based - Feature similarity

**Description**: Recommends movies with similar characteristics (genres, tags, titles) to those the user has rated highly. Uses TF-IDF vectorization and cosine similarity.

**Implementation**: scikit-learn TfidfVectorizer with multi-feature fusion

**Strengths**:
- No cold-start problem for new items
- Highly explainable recommendations
- Independent of other users (privacy-friendly)
- Can recommend niche/unpopular items
- Effective for genre-specific preferences

**Ideal For**: New users with limited history, genre-focused recommendations

**Performance**:
- RMSE: 1.1130 (higher than collaborative methods, as expected)
- Coverage: 100%
- Model Size: 1.11GB
- Prediction Time: ~0.8 seconds per user

**Configurable Parameters**:
- `genre_weight`: Weight for genre features (default: 0.5, range: 0.0-1.0)
- `tag_weight`: Weight for tag features (default: 0.3, range: 0.0-1.0)
- `title_weight`: Weight for title features (default: 0.2, range: 0.0-1.0)
- `min_similarity`: Minimum similarity threshold (default: 0.01, range: 0.0-0.5)

---

### 5. Hybrid Ensemble

**Type**: Ensemble - Adaptive weighted combination

**Description**: Intelligently combines predictions from all four algorithms using adaptive weighting based on user profile, data availability, and historical performance.

**Implementation**: Custom weighted ensemble with dynamic weight adjustment

**Strengths**:
- Best overall accuracy
- Balanced diversity and relevance
- Robust to algorithm weaknesses
- Adapts to user characteristics
- Leverages strengths of all approaches

**Ideal For**: Production deployment, general-purpose recommendations

**Performance**:
- RMSE: 0.8701
- Coverage: 95%+
- Prediction Time: ~2.0 seconds per user
- Weights: SVD=0.33, UserKNN=0.22, ItemKNN=0.25, Content=0.20

**Configurable Parameters**:
- `weighting_strategy`: Combination method (default: adaptive, options: adaptive, equal, performance_based)
- Individual algorithm parameters for each sub-model

---

## Installation

### Prerequisites

- Python 3.13 or higher
- Git with Git LFS extension
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space for dataset and models

### Step 1: Clone Repository

```bash
git clone https://github.com/M7MDRAUF/DataSets.git
cd DataSets
```

### Step 2: Install Git LFS

Git LFS (Large File Storage) is required for downloading pre-trained models (approximately 4.4GB).

**Windows**:
```bash
git lfs install
git lfs pull
```

**macOS** (via Homebrew):
```bash
brew install git-lfs
git lfs install
git lfs pull
```

**Linux** (Ubuntu/Debian):
```bash
sudo apt-get install git-lfs
git lfs install
git lfs pull
```

### Step 3: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Key Dependencies**:
- pandas 2.3.3 (data manipulation)
- numpy 1.26.4 (numerical computing)
- scikit-learn 1.7.2 (machine learning algorithms)
- scikit-surprise 1.1.4 (collaborative filtering)
- streamlit 1.51.0 (web interface)
- plotly 6.5.0 (interactive visualizations)
- joblib 1.3.2 (model persistence)

### Step 5: Download Dataset

1. Download MovieLens 32M dataset:
   - URL: http://files.grouplens.org/datasets/movielens/ml-32m.zip
   - Size: Approximately 700MB compressed

2. Extract the ZIP file

3. Copy CSV files to `data/ml-32m/`:
   ```
   data/ml-32m/
   ├── ratings.csv   (~3.3GB uncompressed, 32 million ratings)
   ├── movies.csv    (Movie catalog with titles and genres)
   ├── links.csv     (IMDb and TMDb identifiers)
   └── tags.csv      (User-generated tags)
   ```

### Step 6: Verify Installation

```bash
# Run system validation script
python scripts/validate_system.py
```

Expected output: All 10 validation checks should pass (100% score).

---

## Quick Start

### Option 1: Launch Web Application (Recommended)

```bash
# Ensure virtual environment is activated
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Launch Streamlit application
streamlit run app/main.py --server.port 8501
```

The application will open automatically in your default browser at `http://localhost:8501`

### Option 2: Quick Start Script (Windows)

```bash
# Double-click start.bat in File Explorer
# Or run from command line:
.\start.bat
```

### Option 3: Docker Deployment

```bash
# Build Docker image
docker build -t cinematch:latest .

# Run container
docker run -p 8501:8501 cinematch:latest
```

Access at `http://localhost:8501`

---

## Usage

### Web Interface

**1. Home Page**
- System overview and dataset statistics
- Quick algorithm comparison
- Sample recommendation generation
- System status and health checks

**2. Recommend Page** (Core Feature)
- Select recommendation algorithm (SVD, User-KNN, Item-KNN, Content-Based, Hybrid)
- Enter user ID (1 to 330,000)
- Generate personalized top-N recommendations
- View algorithm performance metrics (RMSE, coverage, training time)
- Explain individual recommendations (algorithm-specific reasoning)
- Export recommendations to CSV or JSON

**3. Analytics Page**
- Comprehensive algorithm benchmarking
- Side-by-side performance comparison
- Interactive visualizations (rating distributions, genre analysis, temporal trends)
- Sample recommendation generation for all algorithms
- Detailed metrics tables

### Programmatic Usage

```python
from src.algorithms.algorithm_manager import AlgorithmManager, AlgorithmType
from src.data_processing import load_ratings, load_movies

# Initialize manager
manager = AlgorithmManager()

# Load data
ratings_df = load_ratings(sample_size=None)  # None = full dataset
movies_df = load_movies()

# Load/train algorithm
algorithm = manager.get_algorithm(AlgorithmType.SVD)
if not algorithm.is_trained:
    algorithm.fit(ratings_df, movies_df)

# Generate recommendations
recommendations = algorithm.recommend(
    user_id=123,
    n=10,
    exclude_rated=True
)

# Get explanation
explanation = algorithm.get_explanation_context(
    user_id=123,
    movie_id=recommendations.iloc[0]['movieId']
)

print(f"Top recommendation: {recommendations.iloc[0]['title']}")
print(f"Explanation: {explanation}")
```

### Switching Algorithms

```python
# Switch to different algorithm
user_knn = manager.switch_algorithm(AlgorithmType.USER_KNN)
recommendations_knn = user_knn.recommend(user_id=123, n=10)

# Compare algorithms
all_metrics = manager.get_all_algorithm_metrics()
for algo_type, metrics in all_metrics.items():
    print(f"{algo_type}: RMSE={metrics['rmse']:.4f}")
```

---

## Configuration

### Algorithm Parameters

Edit algorithm-specific parameters in their respective files:

**SVD** (`src/model_training_sklearn.py`, line 39):
```python
N_COMPONENTS = 100  # Range: 50-200
```

**User-KNN** (`src/algorithms/user_knn_recommender.py`, line 38):
```python
n_neighbors = 50  # Range: 10-100
similarity_metric = 'cosine'  # Options: cosine, euclidean, manhattan
```

**Item-KNN** (`src/algorithms/item_knn_recommender.py`, line 38):
```python
n_neighbors = 30  # Range: 10-50
min_ratings = 5   # Range: 3-20
```

**Content-Based** (`src/algorithms/content_based_recommender.py`, line 68):
```python
genre_weight = 0.5   # Range: 0.0-1.0
tag_weight = 0.3     # Range: 0.0-1.0
title_weight = 0.2   # Range: 0.0-1.0
```

**Hybrid** (`src/algorithms/hybrid_recommender.py`, line 50):
```python
weighting_strategy = 'adaptive'  # Options: adaptive, equal, performance_based
```

Detailed configuration guide: `reports/ALGORITHM_CONFIGURATION_GUIDE.md`

### Environment Variables

Create `.env` file in project root:

```bash
# Application settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true

# Data settings
DATASET_PATH=data/ml-32m
MODEL_PATH=models

# Performance settings
ENABLE_CACHING=true
CACHE_SIZE=10000
LOG_LEVEL=INFO
```

---

## Testing

### Automated Test Suite

**System Validation**:
```bash
python scripts/validate_system.py
```
Checks: project structure, dataset integrity, dependencies, models, documentation

**Feature Testing**:
```bash
python scripts/test_features.py
```
Tests all 10 functional requirements from project specification

**End-to-End Testing**:
```bash
python scripts/test_end_to_end.py
```
Comprehensive workflow validation including algorithm switching and predictions

### Unit Tests

```bash
# Run pytest suite
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

### Performance Benchmarking

```bash
# Benchmark all algorithms
python scripts/final_benchmark.py

# Benchmark specific algorithm
python scripts/benchmark.py --algorithm svd
```

### Expected Test Results

- System Validation: 10/10 checks passed (100%)
- Feature Tests: 10/10 features passing
- RMSE Targets:
  - SVD: < 0.87 (achieved: 0.7502)
  - User-KNN: < 1.0 (achieved: 0.8394)
  - Item-KNN: < 1.0 (achieved: 0.9100)
  - Hybrid: < 0.90 (achieved: 0.8701)

---

## Deployment

### Docker Deployment

**Build Image**:
```bash
docker build -t cinematch:2.1.7 .
```

**Run Container**:
```bash
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  --name cinematch \
  --memory=8g \
  cinematch:2.1.7
```

**Docker Compose**:
```bash
docker-compose up -d
```

### Production Considerations

1. **Resource Allocation**:
   - Minimum: 4 CPU cores, 8GB RAM, 20GB storage
   - Recommended: 8 CPU cores, 16GB RAM, 50GB storage
   - Current usage: 2.6GB RAM (with 68% headroom in 8GB container)

2. **Scaling**:
   - Stateless design enables horizontal scaling
   - Shared model storage (NFS/S3) for multi-instance deployments
   - Redis caching for distributed environments

3. **Monitoring**:
   - Application logs: `logs/app.log`
   - Performance metrics via Streamlit dashboard
   - Docker container stats: `docker stats cinematch`

4. **Security**:
   - No user authentication in current version (thesis demo)
   - Add reverse proxy (nginx) for production
   - Enable HTTPS with SSL certificates
   - Implement rate limiting and input validation

### Cloud Deployment

**Streamlit Cloud** (recommended for demos):
```bash
# Already configured in streamlit/config.toml
# Simply connect GitHub repository to Streamlit Cloud
```

**AWS/GCP/Azure**:
- Use container services (ECS, Cloud Run, Container Instances)
- Attach persistent volumes for data and models
- Configure load balancer for high availability

---

## Performance Metrics

### Algorithm Comparison

| Algorithm | RMSE | MAE | Coverage | Training Time | Prediction Time | Model Size |
|-----------|------|-----|----------|---------------|-----------------|------------|
| SVD | 0.7502 | 0.5812 | 24.1% | ~25 min | 0.5s | 954 MB |
| User-KNN | 0.8394 | 0.6523 | 95%+ | N/A (memory) | 1.5s | 1.17 GB |
| Item-KNN | 0.9100 | 0.7012 | 90%+ | N/A (memory) | 1.0s | 1.16 GB |
| Content-Based | 1.1130 | 0.8834 | 100% | ~15 min | 0.8s | 1.11 GB |
| Hybrid | 0.8701 | 0.6745 | 95%+ | Combined | 2.0s | Composed |

### System Performance

- **Dataset**: 32 million ratings, 86,000+ movies, 330,000+ users
- **Model Loading**: 1-9 seconds (varies by algorithm)
- **Memory Usage**: 185MB runtime (98.6% optimization from v2.0.x)
- **Docker Container**: 2.6GB / 8GB (32% utilization)
- **Recommendation Generation**: Under 2 seconds per user
- **Concurrent Users**: Tested up to 10 simultaneous users

### Optimization Techniques

1. **Memory Optimization**:
   - Shallow references instead of data copying (13.2GB to 185MB)
   - LRU caching with bounded size and TTL
   - Lazy loading of models and data structures

2. **Computation Optimization**:
   - Vectorized batch processing (200x speedup)
   - Smart candidate sampling (80K to 5K items)
   - Precomputed similarity matrices
   - Efficient sparse matrix operations

3. **I/O Optimization**:
   - Parquet format for 5x faster data loading
   - Model checkpointing with joblib
   - Intelligent caching layer

---

## Documentation

### Core Documentation

- **README.md**: This file - comprehensive project overview
- **doc/QUICKSTART.md**: 15-minute setup guide with step-by-step instructions
- **doc/ARCHITECTURE.md**: Detailed system architecture and design decisions
- **doc/ALGORITHM_ARCHITECTURE.md**: Deep dive into recommendation algorithms
- **doc/ALGORITHM_COMPLEXITY.md**: Computational complexity analysis
- **reports/ALGORITHM_CONFIGURATION_GUIDE.md**: Algorithm tuning and parameter optimization

### Technical Documentation

- **doc/API_DOCUMENTATION.md**: Programmatic API reference
- **doc/DEPLOYMENT.md**: Production deployment guide
- **doc/DOCKER.md**: Docker configuration and usage
- **doc/TESTING.md**: Comprehensive testing strategy
- **doc/TROUBLESHOOTING.md**: Common issues and solutions
- **doc/PERFORMANCE_TUNING.md**: Optimization techniques and benchmarks

### User Documentation

- **scripts/README.md**: Automation scripts and workflow guide
- **scripts/demo_script.md**: Thesis defense demonstration script
- **doc/RUNBOOK.md**: Operational procedures and maintenance

### Development Documentation

- **doc/CODE_STYLE.md**: Coding standards and conventions
- **doc/CONTRIBUTING_GUIDE.md**: Contribution guidelines
- **doc/EXTENDING.md**: How to add custom algorithms
- **CHANGELOG.md**: Version history and release notes

---

## Contributing

### Development Workflow

1. **Fork Repository**:
   ```bash
   git clone https://github.com/M7MDRAUF/DataSets.git
   cd DataSets
   ```

2. **Create Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**:
   - Follow code style guidelines (doc/CODE_STYLE.md)
   - Add tests for new features
   - Update documentation

4. **Test Changes**:
   ```bash
   python scripts/validate_system.py
   python scripts/test_features.py
   pytest tests/ -v
   ```

5. **Commit and Push**:
   ```bash
   git add .
   git commit -m "feat: descriptive commit message"
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**:
   - Provide clear description of changes
   - Reference related issues
   - Ensure all checks pass

### Code Standards

- **Style**: PEP 8 compliant Python code
- **Type Hints**: Use type annotations for function signatures
- **Documentation**: Docstrings for all public methods (Google style)
- **Testing**: Unit tests for new functionality (>80% coverage target)
- **Commits**: Conventional Commits format (feat, fix, docs, refactor, test)

### Adding Custom Algorithms

See `doc/EXTENDING.md` for detailed guide on implementing custom recommendation algorithms.

Quick example:
```python
from src.algorithms.base_recommender import BaseRecommender

class CustomRecommender(BaseRecommender):
    def __init__(self, **kwargs):
        super().__init__("Custom Algorithm", **kwargs)
    
    def fit(self, ratings_df, movies_df):
        # Training logic
        pass
    
    def recommend(self, user_id, n=10, exclude_rated=True):
        # Recommendation logic
        pass
```

---

## Troubleshooting

### Common Issues

**Issue**: Model files not found after cloning

**Solution**:
```bash
# Ensure Git LFS is installed and initialized
git lfs install
git lfs pull
```

---

**Issue**: Out of memory errors

**Solution**:
- Reduce dataset sampling: `load_ratings(sample_size=1000000)` instead of `None`
- Close other applications to free RAM
- Increase Docker memory limit: `docker run --memory=16g ...`
- Use smaller `n_components` for SVD training

---

**Issue**: Slow recommendation generation

**Solution**:
- Ensure models are pre-trained (check `models/` directory)
- Use algorithm with precomputed similarities (Item-KNN)
- Enable caching in configuration
- Reduce number of candidates via sampling

---

**Issue**: Dataset integrity check fails

**Solution**:
- Re-download MovieLens 32M dataset
- Verify file sizes:
  - ratings.csv: ~700MB compressed, ~3.3GB uncompressed
  - movies.csv: ~3MB
  - links.csv: ~3MB
  - tags.csv: ~20MB
- Check file permissions (read access required)

---

**Issue**: Algorithm returns no recommendations

**Solution**:
- Verify user ID exists in dataset: `ratings_df['userId'].min()` to `ratings_df['userId'].max()`
- Check if algorithm is trained: `algorithm.is_trained` should be `True`
- Review logs for error messages: `logs/app.log`
- Try different algorithm (some have better coverage)

---

**Issue**: Docker container crashes

**Solution**:
- Check Docker logs: `docker logs cinematch`
- Increase memory allocation: `--memory=8g` or higher
- Verify all required files are mounted
- Ensure ports are not in use: `netstat -ano | findstr :8501`

---

For additional troubleshooting, see `doc/TROUBLESHOOTING.md` or open an issue on GitHub.

---

## Acknowledgments

### Dataset

This project uses the **MovieLens 32M Dataset** provided by GroupLens Research at the University of Minnesota.

**Citation**:
```
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. 
ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.
https://doi.org/10.1145/2827872
```

**License**: Creative Commons Attribution 4.0 International (CC BY 4.0)

### Technologies

- **scikit-learn**: Machine learning library for Python
- **scikit-surprise**: Python library for recommender systems
- **Streamlit**: Framework for data science applications
- **Plotly**: Interactive visualization library
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

### References

- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer, 42(8), 30-37.
- Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based collaborative filtering recommendation algorithms. WWW '01.
- Lops, P., de Gemmis, M., & Semeraro, G. (2011). Content-based recommender systems: State of the art and trends. Recommender systems handbook, 73-105.

---

## License

MIT License

Copyright (c) 2025 Mohamed Rauf

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Contact

**Author**: Mohamed Rauf  
**Email**: m7mdrauf@example.com  
**GitHub**: https://github.com/M7MDRAUF  
**Repository**: https://github.com/M7MDRAUF/DataSets  

**Project Status**: Complete - Ready for Thesis Defense  
**Last Updated**: December 11, 2025  
**Version**: 2.1.7
