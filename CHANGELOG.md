# 📋 CineMatch Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2.1.0] - 2025-11-11 - Content-Based Filtering Release

### 🎯 Major Features

#### Content-Based Filtering Algorithm
- **Added** ContentBasedRecommender class with TF-IDF feature extraction
- **Added** Movie feature extraction from genres, tags, and titles
- **Added** Configurable feature weights (genres=0.5, tags=0.3, titles=0.2)
- **Added** Cosine similarity computation with sparse matrices
- **Added** User profile building from rating history with weighted averages
- **Added** Cold-start handling for new users with popular movie fallbacks
- **Added** Recommendation explanation generation with feature matching
- **Added** Training script `train_content_based.py` with CLI arguments
- **Performance** ~50-100ms prediction time, ~500-700MB memory usage

#### AlgorithmManager Integration
- **Added** `CONTENT_BASED` to `AlgorithmType` enum (5th algorithm)
- **Added** ContentBasedRecommender registration in algorithm factory
- **Added** Default parameters for Content-Based (genre/tag/title weights)
- **Added** Pre-trained model loading support (`models/content_based_model.pkl`)
- **Added** Content-Based info to `get_algorithm_info()` with 🔍 icon

#### Hybrid Algorithm Enhancement
- **Updated** Hybrid to 4-algorithm ensemble (was 3 algorithms)
- **Updated** Algorithm weights: SVD=0.30, UserKNN=0.25, ItemKNN=0.25, ContentBased=0.20
- **Added** `_try_load_content_based()` method for pre-trained loading
- **Updated** `_calculate_weights()` for 4-algorithm adaptive scoring
- **Updated** `_predict_hybrid_rating()` to include Content-Based predictions
- **Updated** Metrics calculation to include Content-Based coverage and RMSE

### 🎨 Frontend Updates

#### Home Page (`1_🏠_Home.py`)
- **Added** Content-Based to algorithm selector dropdown
- **Added** 🔍 icon and purple color (#9467bd) to algorithm info cards
- **Added** `AlgorithmType.CONTENT_BASED` to algorithm mapping

#### Recommend Page (`2_🎬_Recommend.py`)
- **Added** 🔍 icon to algorithm icons array
- **Updated** Algorithm selector to include Content-Based
- **Note** Explanation functionality already supports Content-Based via AlgorithmManager

#### Analytics Page (`3_📊_Analytics.py`)
- **Added** Content-Based to algorithm benchmarking loop
- **Added** Content-Based to similarity finder algorithm map
- **Updated** Performance comparison charts to include 5 algorithms

### 🔧 Technical Implementation

#### Feature Engineering
- **Implemented** TF-IDF vectorization for genres with pipe-separated parsing
- **Implemented** Tag aggregation from `tags.csv` with relevance weights
- **Implemented** Title keyword extraction with regex pattern matching
- **Implemented** Sparse matrix operations (CSR format) for memory efficiency
- **Implemented** Feature matrix construction with weighted concatenation

#### Similarity & Prediction
- **Implemented** Movie-movie similarity matrix with cosine similarity
- **Implemented** Min similarity threshold (0.01) for sparsification
- **Implemented** Batch processing (5000 movies) to prevent memory issues
- **Implemented** User profile normalization to unit vectors
- **Implemented** Prediction formula: user_profile × movie_features

#### Model Persistence
- **Implemented** Model serialization with pickle format
- **Implemented** Metadata storage (version, trained_on, dimensions, n_movies)
- **Implemented** `load_model()` class method for pre-trained loading
- **Implemented** `save_model()` instance method with validation metrics

### 📈 Performance Characteristics

- **Training Time**: ~2-3 minutes (10K sample), ~15-20 minutes (full 87K movies)
- **Memory Usage**: ~500-700 MB (feature + similarity matrices)
- **Prediction Time**: ~50-100ms per user
- **Expected RMSE**: ~0.85-0.95
- **Expected Coverage**: ~85-95%
- **Diversity**: High (feature-based, not rating-based)

### 🔄 Backward Compatibility

- **Maintained** All existing algorithms (SVD, UserKNN, ItemKNN, Hybrid) unchanged
- **Maintained** BaseRecommender interface consistency
- **Maintained** AlgorithmManager API unchanged
- **Maintained** All existing UI pages functional
- **Note** Zero breaking changes - fully backward compatible

### 📚 Documentation

- **Added** `CONTENT_BASED_COMPLETE.md` - Comprehensive implementation guide
- **Added** `CONTENT_BASED_IMPLEMENTATION_STATUS.md` - Phase-by-phase status
- **Updated** `README.md` - Version to 2.1.0, added Content-Based to algorithm list
- **Updated** Algorithm comparison table to include Content-Based
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
- **Added** ALGORITHM_ARCHITECTURE_PLAN.md for development planning
- **Added** debug.md with comprehensive debugging methodology
- **Added** DOCKER.md with containerization instructions

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
