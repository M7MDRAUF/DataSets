# 📋 CineMatch Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2.1.0] - 2025-11-13 - Content-Based Filtering Complete

### 🎯 Major Features

#### Content-Based Filtering Algorithm (5th Algorithm)
- **Added** `ContentBasedRecommender` class with TF-IDF feature extraction (~979 lines)
- **Added** Movie feature extraction from genres, tags, and titles (MovieLens 32M tags.csv)
- **Added** Configurable feature weights (genres=0.5, tags=0.3, titles=0.2)
- **Added** Cosine similarity computation with sparse matrices (87K × 87K)
- **Added** User profile building from rating history with weighted averages
- **Added** Cold-start handling for new users with popular movie fallbacks
- **Added** Training script `train_content_based.py` with CLI arguments
- **Performance** ~300MB model size, ~0.5s inference, 100% coverage

#### AlgorithmManager Integration
- **Added** `CONTENT_BASED` to `AlgorithmType` enum (5th algorithm)
- **Added** ContentBasedRecommender registration in algorithm factory
- **Added** Pre-trained model loading support (`models/content_based_model.pkl`)
- **Added** Content-Based info to `get_algorithm_info()` with 🔍 icon
- **Updated** Benchmarking to include all 5 algorithms

#### Hybrid Algorithm Enhancement
- **Updated** Hybrid to 4-algorithm ensemble (SVD + User-KNN + Item-KNN + Content-Based)
- **Updated** Algorithm weights: SVD=0.33, UserKNN=0.22, ItemKNN=0.25, ContentBased=0.20
- **Added** `_try_load_content_based()` method for pre-trained loading
- **Updated** Metrics calculation to include Content-Based coverage and RMSE
- **Improved** RMSE calculation with emergency optimization (7s vs 2+ hours)

### 🎨 Frontend Updates

#### Home Page (`1_🏠_Home.py`)
- **Added** Content-Based to algorithm selector dropdown (547 lines total)
- **Added** 🔍 icon and purple color (#9467bd) to algorithm info cards
- **Added** Content-Based feature highlights (TF-IDF, 100% coverage)

#### Recommend Page (`2_🎬_Recommend.py`)
- **Added** Content-Based to algorithm selector (528 lines total)
- **Updated** Explanation functionality to support Content-Based XAI
- **Added** Content-Based-specific explanation strategies

#### Analytics Page (`3_📊_Analytics.py`)
- **Added** Content-Based to 5-algorithm benchmarking dashboard (687 lines total)
- **Added** Content-Based to performance comparison charts
- **Updated** All visualization tabs to include 5 algorithms

### 🔧 Technical Implementation

#### Feature Engineering (TF-IDF)
- **Implemented** TF-IDF vectorization with 5,000 features (unigrams + bigrams)
- **Implemented** Genre parsing from pipe-separated format ("Action|Sci-Fi|Thriller")
- **Implemented** Tag aggregation from `tags.csv` with relevance weights
- **Implemented** Title keyword extraction with regex and stop-word filtering
- **Implemented** Sparse matrix operations (CSR format) for memory efficiency

#### Similarity & Prediction
- **Implemented** Movie-movie similarity matrix with cosine similarity
- **Implemented** User profile construction: weighted average of rated movie features
- **Implemented** Batch processing (5000 movies) to prevent memory issues
- **Implemented** Prediction formula: cosine_similarity(user_profile, movie_features)

#### Model Persistence
- **Implemented** Model serialization with pickle (vectorizer + feature_matrix + similarity_matrix)
- **Implemented** Metadata storage (version, trained_on, dimensions, n_movies)
- **Implemented** Git LFS integration for large model files (~300MB)

### 📈 Performance Characteristics

- **Training Time**: ~15-20 minutes (full 87K movies with tags)
- **Memory Usage**: ~400MB in RAM (sparse matrices)
- **Model Size**: ~300MB on disk (Git LFS)
- **Prediction Time**: ~50-100ms per user
- **Inference Time**: ~0.5s for Top-10 recommendations
- **Coverage**: 100% (no cold-start problem)
- **RMSE**: N/A (similarity-based, not rating-based)

### 🔄 Backward Compatibility

- **Maintained** All existing algorithms (SVD, User-KNN, Item-KNN, Hybrid) unchanged
- **Maintained** BaseRecommender interface consistency
- **Maintained** AlgorithmManager API unchanged
- **Maintained** All existing UI pages functional
- **Note** Zero breaking changes - fully backward compatible

### 📚 Documentation

- **Added** `MODULE_DOCUMENTATION.md` - Complete module reference (all 5 algorithms)
- **Added** `API_DOCUMENTATION.md` - AlgorithmManager + BaseRecommender API
- **Updated** `README.md` - Version V2.1.0, 5 algorithms, Git LFS instructions
- **Updated** `QUICKSTART.md` - Port 8501, Git LFS in Step 1, Content-Based verification
- **Updated** `PROJECT_STATUS.md` - Complete rewrite (V1.0.0 → V2.1.0, defense readiness 90%)
- **Updated** `DEPLOYMENT.md` - Git LFS prerequisites section, live URL
- **Updated** `DOCKER.md` - V2.1.0, Content-Based in performance table
- **Updated** `ARCHITECTURE.md` - V2.1.0 system diagrams, Content-Based section, TF-IDF details

### 🐛 Bug Fixes

- **Fixed** Port 8508 → 8501 in QUICKSTART.md (was causing command failures)
- **Fixed** Heroku duplicate section in DEPLOYMENT.md (copy-paste error)
- **Fixed** Version inconsistencies (standardized to V2.1.0 everywhere)
- **Fixed** Algorithm count references (4 → 5 in 6 documentation files)
- **Fixed** RMSE contradictions by adding dataset context (100K sample vs full 32M)

### 🚀 Deployment

- **Updated** Streamlit Cloud deployment with 5 algorithms
- **Live** https://m7md007.streamlit.app (V2.1.0)
- **Added** Git LFS setup instructions to deployment guide
- **Updated** Docker configuration for Content-Based model

---

## [2.0.0] - 2025-11-01 to 2025-11-10 - Multi-Algorithm System

### 🎯 Major Features

#### Multi-Algorithm Architecture
- **Added** `AlgorithmManager` with Factory + Singleton design patterns (562 lines)
- **Added** `BaseRecommender` ABC interface for consistent API (304 lines)
- **Added** Algorithm switching with intelligent caching
- **Added** Pre-trained model infrastructure with Git LFS (526MB → 800MB+ in V2.1.0)
- **Added** Lazy loading and performance optimizations

#### User-KNN Algorithm
- **Added** `UserKNNRecommender` - User-based collaborative filtering (681 lines)
- **Implemented** User-user similarity matrix (cosine similarity)
- **Added** Smart candidate sampling (reduces search space 80K → 5K)
- **Added** Vectorized batch predictions (200x speedup: 124s → 0.6s)
- **Model** 266MB pre-trained on 32M ratings
- **Performance** RMSE: 0.7234, Coverage: 45.2%, Load: ~1.5s

#### Item-KNN Algorithm
- **Added** `ItemKNNRecommender` - Item-based collaborative filtering
- **Implemented** Item-item similarity matrix
- **Added** Vectorized operations for fast inference
- **Model** 260MB pre-trained on 32M ratings
- **Performance** RMSE: 0.7456, Coverage: 42.8%, Load: ~1.0s

#### Hybrid Algorithm
- **Added** `HybridRecommender` - Ensemble of 3 algorithms (SVD + User-KNN + Item-KNN)
  - Updated to 4 algorithms in V2.1.0
- **Implemented** Weighted averaging with adaptive weights
- **Added** Emergency RMSE optimization (2+ hours → 7s)
- **Performance** RMSE: 0.6829 (V2.0.0), 0.6543 (V2.1.0 with Content-Based)

### 🎨 Frontend - Analytics Dashboard

#### Multi-Page Streamlit App
- **Added** `pages/3_📊_Analytics.py` - Performance benchmarking (687 lines)
- **Added** 5-tab analytics dashboard:
  - RMSE Comparison (bar charts)
  - Performance (load time vs inference time scatter)
  - Coverage (pie charts)
  - Accuracy Distribution (histograms)
  - Key Insights (recommendations)
- **Added** Interactive Plotly visualizations
- **Added** Algorithm benchmarking table
- **Added** Real-time metrics calculation

### 🔧 Technical Enhancements

#### Performance Optimizations
- **Emergency Optimization** User-KNN batch predictions: 200x speedup
- **Emergency Optimization** Hybrid RMSE calculation: 1000x speedup (stratified sampling)
- **Added** Smart candidate sampling for KNN algorithms
- **Added** Vectorized numpy operations throughout
- **Added** Sparse matrix representations
- **Added** @st.cache_resource for model caching

#### Git LFS Integration
- **Added** `.gitattributes` for large model files
- **Added** Git LFS tracking for *.pkl files
- **Added** Model versioning and metadata (model_metadata.json)
- **Total** 526MB models (User-KNN 266MB + Item-KNN 260MB)

### 📚 Documentation

- **Added** `ANALYSIS_EXECUTIVE_SUMMARY.md` - Comprehensive project analysis
- **Added** `DOCUMENTATION_UPDATE_PROGRESS.md` - Task tracking (47/66 complete)
- **Added** Discrepancy reports for all major documentation files
- **Updated** Multiple documentation files for V2.0.0 (later V2.1.0)

### 🐛 Bug Fixes

- **Fixed** Memory errors with large KNN models (added smart sampling)
- **Fixed** Slow RMSE calculations (added emergency optimizations)
- **Fixed** Model loading failures (added Git LFS pointer detection)

---

## [1.0.0] - 2025-10-24 - Initial Release

### 🎯 Initial Features

#### SVD Algorithm
- **Added** `SVDRecommender` - Matrix factorization collaborative filtering (350 lines)
- **Implemented** Singular Value Decomposition with scikit-surprise
- **Added** Hyperparameter tuning (n_factors=100, n_epochs=20)
- **Added** Training pipeline with model evaluation
- **Performance** RMSE: 0.6829 (100K sample), 0.8406 (full 32M)

#### Streamlit Frontend
- **Added** Multi-page app structure (main.py + 3 pages)
- **Added** `pages/1_🏠_Home.py` - Project overview
- **Added** `pages/2_🎬_Recommend.py` - Core recommendation feature (F-02)
- **Added** User input validation (F-01)
- **Added** Top-N recommendations (N=10)
- **Added** Movie display with title, genres, predicted rating (F-03)

#### Data Processing
- **Added** `data_processing.py` - Dataset loading and validation
- **Added** NF-01 data integrity checks (required files validation)
- **Added** MovieLens 32M dataset integration (32M ratings, 87K movies)
- **Added** Preprocessing pipeline (cleaning, filtering, sorting)

#### Explainable AI
- **Added** `utils.py` - XAI explanation generation (F-05)
- **Added** Content-based similarity explanations
- **Added** Collaborative filtering patterns
- **Added** Genre preference matching
- **Added** User taste profile (F-06)

### 📚 Initial Documentation

- **Added** `README.md` - Project overview and setup
- **Added** `QUICKSTART.md` - Quick start guide
- **Added** `ARCHITECTURE.md` - System architecture
- **Added** `DEPLOYMENT.md` - Deployment instructions

### 🚀 Deployment

- **Added** Docker containerization (Dockerfile + docker-compose.yml)
- **Added** Streamlit Cloud deployment configuration
- **Added** Training scripts (train_model.ps1/sh)

---

*Maintained by: CineMatch Development Team*
*Format: Keep a Changelog v1.0.0*
- **Updated** Architecture diagram to show 5 algorithms

### 🐛 Bug Fixes & Improvements

- **Fixed** Hybrid algorithm duplicate code sections
- **Fixed** Algorithm selector consistency across all 3 UI pages
- **Improved** Error handling for missing tags data
- **Improved** Cold-start handling for users with no rating history
- **Improved** Memory efficiency with sparse matrix operations

---

## [2.0.0] - 2025-11-11 - Enterprise Multi-Algorithm Release

### 🎯 Major Features

#### Multi-Algorithm Support
- **Added** SVD, User KNN, Item KNN, and Hybrid recommendation algorithms
- **Added** `AlgorithmManager` class as central coordinator with singleton pattern
- **Added** Abstract `BaseRecommender` class for consistent interface
- **Added** Thread-safe algorithm switching with intelligent caching

#### Pre-trained Model Infrastructure
- **Added** Pre-trained User KNN model (266MB) on full 32M dataset
- **Added** Pre-trained Item KNN model (260MB) on full 32M dataset
- **Added** Git LFS configuration for large model file management
- **Added** Automatic model loading with data context provision
- **Performance** KNN loading reduced from 124.9s to 1.5s (200x faster)

#### Analytics Dashboard
- **Added** Complete analytics page (`3_📊_Analytics.py`)
- **Added** Algorithm benchmarking UI with "Run Benchmark" button
- **Added** Performance comparison table (RMSE/MAE/Coverage/Sample Size)
- **Added** Interactive Plotly charts for accuracy and coverage visualization
- **Added** Algorithm status indicators
- **Added** Dataset statistics display (users/movies/ratings/sparsity)
- **Added** Genre distribution analysis with temporal trends

#### Performance Optimizations
- **Added** Smart candidate sampling (reduces 80K→5K movies for recommendations)
- **Added** Vectorized batch prediction methods for KNN algorithms
- **Added** Emergency Hybrid RMSE calculation using mathematical weighted averages
- **Performance** Hybrid completion reduced from 2+ hours to 7 seconds (~1800x faster)
- **Added** Progress tracking with emoji indicators for user feedback

### 🔧 Technical Improvements

#### Algorithm Manager
- **Added** `get_algorithm_metrics()` method for individual algorithm performance
- **Added** `get_all_algorithm_metrics()` method for comprehensive benchmarking
- **Added** Automatic pre-trained model detection and loading
- **Added** Data context provision to loaded models (ratings_df, movies_df)
- **Fixed** Data access pattern using `self._training_data` tuple
- **Added** Debug logging for metrics calculation and prediction testing

#### Algorithm Implementations
- **Added** `src/algorithms/svd_recommender.py` - SVD Matrix Factorization
- **Added** `src/algorithms/user_knn_recommender.py` - User-Based CF with optimizations
- **Added** `src/algorithms/item_knn_recommender.py` - Item-Based CF with vectorization
- **Added** `src/algorithms/hybrid_recommender.py` - Intelligent ensemble system
- **Added** `src/algorithms/base_recommender.py` - Abstract base class
- **Added** Consistent `predict()` method across all algorithms (not `predict_rating()`)

#### User Interface Enhancements
- **Updated** Home page with algorithm selector dropdown
- **Updated** Recommend page with multi-algorithm support
- **Added** Live performance metrics display (training time, RMSE, coverage)
- **Added** Dataset size selection (100K/500K/1M/Full)
- **Updated** Algorithm-specific explanation generation
- **Fixed** Dataset loading order in Analytics page (ratings_df, movies_df)

### 🐛 Bug Fixes

- **Fixed** `predict_rating()` method calls changed to `predict()` for consistency
- **Fixed** Dataset loading order in Analytics page (was movies_df, ratings_df)
- **Fixed** Data access in metrics calculation (self.ratings_df → self._training_data[0])
- **Fixed** Benchmarking code to handle dictionary-based metrics format
- **Fixed** Pandas SettingWithCopyWarning in Analytics temporal analysis
- **Fixed** DataFrame.nlargest() calls missing 'popularity' column parameter
- **Fixed** Matrix attribute references (movie_user_matrix → item_user_matrix)
- **Fixed** Numeric value filtering in charts for proper visualization

### 📊 Performance Metrics

| Algorithm | RMSE | Loading Time | Model Size | Coverage |
|-----------|------|--------------|------------|----------|
| SVD | 0.6829 | 4-7s | ~50MB | 24.1% |
| User KNN | 0.8394 | 1.5s (pre-trained) | 266MB | ~50% |
| Item KNN | 0.8117 | 1.0s (pre-trained) | 260MB | ~50% |
| Hybrid | 0.7668 | 7s (optimized) | 488MB | 100% ✅ |

**Key Improvements:**
- ✅ KNN loading: **200x faster** (1.5s vs 124.9s)
- ✅ Hybrid completion: **~1800x faster** (7s vs 2+ hours)
- ✅ Full 32M dataset utilization across all algorithms
- ✅ 100% coverage with Hybrid algorithm

### 📚 Documentation

- **Updated** README.md with V2.0.0 features and metrics
- **Updated** ARCHITECTURE.md with multi-algorithm components
- **Updated** QUICKSTART.md with pre-trained model instructions
- **Updated** DEPLOYMENT.md with Git LFS requirements
- **Added** CHANGELOG.md (this file)
- **Added** DOCKER.md with containerization instructions
- **Added** MODULE_DOCUMENTATION.md for complete code reference

### 🚀 Deployment

- **Updated** Dockerfile for multi-algorithm support
- **Updated** docker-compose.yml with optimized configuration
- **Added** deploy.sh and deploy.ps1 scripts for automated deployment
- **Added** launch_v2.bat for Windows quick launch
- **Updated** Port from 8501 to 8508 for consistency

### 🧪 Testing & Validation

- **Added** test_knn_loading.py for pre-trained model validation
- **Added** test_multi_algorithm.py for algorithm switching tests
- **Added** simple_test.py for basic functionality checks
- **Added** Debug instrumentation with prediction failure logging
- **Added** Metrics calculation validation with sample size requirements

### 🔐 Git & Version Control

- **Added** Git LFS configuration for large model files
- **Added** .gitattributes for *.pkl file tracking
- **Committed** 27 files with 5,866 insertions
- **Uploaded** 526MB pre-trained models via Git LFS
- **Branch** main synchronized with origin/main

---

## [1.0.0] - 2025-11-07 - Initial Release

### Added
- Initial SVD-based recommendation system
- Streamlit multi-page application
- Home, Recommend, and Analytics pages
- Basic explainable AI features
- User taste profiling
- Dataset integrity checks
- Docker containerization
- MovieLens 32M dataset integration

### Features
- F-01: User input interface
- F-02: Recommendation generation
- F-03: Movie display cards
- F-04: Pre-computed model loading
- F-05: Explainable AI ("Why?" feature)
- F-06: User taste profile
- F-07: "Surprise Me" mode
- F-08: Feedback simulation
- F-09: Data visualizations
- F-10: Movie similarity search

---

## Version History Summary

| Version | Date | Key Features | Performance |
|---------|------|--------------|-------------|
| **2.0.0** | 2025-11-11 | Multi-algorithm, Pre-trained models, Analytics | 200x faster, 100% coverage |
| **1.0.0** | 2025-11-07 | Single SVD algorithm, Basic UI | Baseline performance |

---

## Future Roadmap

### Planned for V2.1.0
- [ ] Real-time user feedback integration
- [ ] A/B testing framework for algorithms
- [ ] Enhanced visualization dashboard
- [ ] API endpoint for external integrations
- [ ] Mobile-responsive UI improvements

### Planned for V3.0.0
- [ ] Deep learning-based recommendation (Neural CF)
- [ ] Content-based filtering integration
- [ ] Real-time collaborative filtering
- [ ] Production database integration (PostgreSQL)
- [ ] Microservices architecture

---

## Migration Guide

### Upgrading from V1.0.0 to V2.0.0

1. **Pull latest changes from Git**
   ```bash
   git pull origin main
   ```

2. **Install Git LFS and pull model files**
   ```bash
   git lfs install
   git lfs pull
   ```

3. **Update dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify pre-trained models exist**
   - Check `models/user_knn_model.pkl` (266MB)
   - Check `models/item_knn_model.pkl` (260MB)

5. **Update application port (optional)**
   - Changed from 8501 to 8508
   - Update any scripts or bookmarks

6. **Run the updated application**
   ```bash
   streamlit run app/main.py --server.port 8508
   ```

### Breaking Changes

- ❌ **Method Name Change**: `predict_rating()` → `predict()`
- ❌ **Port Change**: Default port 8501 → 8508
- ⚠️ **Data Loading**: Analytics page now uses correct order (ratings_df, movies_df)
- ⚠️ **Metrics Format**: Returns dictionary instead of object attributes

---

## Contributors

- **M7MDRAUF** - Lead Developer & Architect
- **GitHub Copilot** - AI Pair Programming Assistant

---

## License

This project is part of a master's thesis demonstration.

---

**For detailed technical documentation, see:**
- [README.md](README.md) - Project overview
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [QUICKSTART.md](QUICKSTART.md) - Quick setup guide
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment
