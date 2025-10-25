# CineMatch V1.0.0 - Professional Architecture Documentation

## 🎯 Executive Summary

**CineMatch** is a production-grade movie recommendation engine built for a master's thesis demonstration. It leverages collaborative filtering (SVD matrix factorization) on the MovieLens 32M dataset to provide personalized, explainable movie recommendations through an interactive Streamlit web interface.

**Key Differentiators:**
- ✅ Explainable AI: Every recommendation comes with a "why"
- ✅ User Taste Profiling: Deep insight into user preferences
- ✅ Professional Engineering: Docker containerization, data integrity checks
- ✅ Sub-second Response Time: Pre-trained model with optimized inference

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE LAYER                     │
│                   (Streamlit Multi-Page App)                 │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Home Page     │ Recommend Page  │   Analytics Page        │
│  (Overview &    │ (Core Feature)  │  (Data Insights)        │
│  Visualizations)│                 │                         │
└────────┬────────┴────────┬────────┴──────────┬──────────────┘
         │                 │                   │
         └─────────────────┼───────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────┐
│                   APPLICATION LAYER                         │
│                  (Business Logic - src/)                    │
├──────────────────┬──────────────────┬──────────────────────┤
│ Recommendation   │  Explanation     │   Data Processing    │
│    Engine        │    Engine        │      Module          │
│                  │   (XAI Logic)    │  (Integrity Checks)  │
└────────┬─────────┴─────────┬────────┴──────────┬───────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│                      MODEL LAYER                              │
│              (Pre-trained SVD Model - models/)                │
├───────────────────────────────────────────────────────────────┤
│  • Trained on 32M ratings                                     │
│  • Optimized for RMSE < 0.87                                  │
│  • Serialized with joblib/pickle                              │
│  • Loaded once and cached                                     │
└─────────────────────────────┬─────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────────┐
│                        DATA LAYER                               │
│                    (MovieLens 32M Dataset)                      │
├─────────────────────────────────────────────────────────────────┤
│  • ratings.csv (32M user-movie-rating records)                  │
│  • movies.csv (Movie metadata: title, genres)                   │
│  • links.csv (IMDb/TMDb IDs for external integration)           │
│  • tags.csv (User-generated tags)                               │
│  • Integrity checked on startup (NF-01)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure & Component Responsibilities

```
cinematch-demo/
│
├── 📊 data/                              # DATA LAYER
│   ├── ml-32m/                           # Raw MovieLens 32M dataset
│   │   ├── ratings.csv                   # 32M ratings (userId, movieId, rating, timestamp)
│   │   ├── movies.csv                    # Movie catalog (movieId, title, genres)
│   │   ├── links.csv                     # External IDs (movieId, imdbId, tmdbId)
│   │   └── tags.csv                      # User tags (userId, movieId, tag, timestamp)
│   └── processed/                        # Preprocessed/cached data
│       ├── user_genre_matrix.pkl         # User-genre preference matrix
│       └── movie_features.pkl            # Extracted movie features
│
├── 🧠 models/                            # MODEL LAYER
│   ├── svd_model.pkl                     # Trained SVD model (primary)
│   └── model_metadata.json               # Training metrics, hyperparameters
│
├── ⚙️ src/                               # APPLICATION LAYER (Core Logic)
│   ├── __init__.py
│   ├── data_processing.py                # 🔍 Data integrity checker (NF-01)
│   │   ├── check_data_integrity()        #    Validates dataset presence
│   │   ├── load_ratings()                #    Loads ratings.csv
│   │   ├── load_movies()                 #    Loads movies.csv
│   │   ├── preprocess_data()             #    Cleans and transforms data
│   │   └── create_user_genre_matrix()    #    Generates user taste profiles
│   │
│   ├── model_training.py                 # 🎓 Model training pipeline
│   │   ├── train_svd_model()             #    Trains SVD on full dataset
│   │   ├── evaluate_model()              #    Calculates RMSE, MAE
│   │   ├── save_model()                  #    Serializes trained model
│   │   └── hyperparameter_tuning()       #    Grid search for optimization
│   │
│   ├── recommendation_engine.py          # 🎬 Core recommendation logic
│   │   ├── load_model()                  #    Loads pre-trained model (cached)
│   │   ├── get_recommendations()         #    F-02: Top-N predictions
│   │   ├── get_user_history()            #    Retrieves user's rated movies
│   │   ├── filter_unseen_movies()        #    Excludes already-rated movies
│   │   └── surprise_recommendations()    #    F-07: Serendipity mode
│   │
│   └── utils.py                          # 🧩 Explainability & helpers
│       ├── explain_recommendation()      #    F-05: XAI logic
│       ├── get_user_taste_profile()      #    F-06: Genre preferences
│       ├── find_similar_users()          #    Collaborative filtering insights
│       ├── get_similar_movies()          #    F-10: Item-item similarity
│       └── format_genres()               #    UI formatting utilities
│
├── 🎨 app/                               # USER INTERFACE LAYER (Streamlit)
│   ├── main.py                           # App entry point & configuration
│   ├── 1_🏠_Home.py                      # Landing page & overview
│   │   ├── Display project intro
│   │   ├── Show dataset statistics
│   │   └── Visualize top genres
│   │
│   ├── 2_🎬_Recommend.py                 # ⭐ CORE FEATURE PAGE
│   │   ├── F-01: User ID input
│   │   ├── F-02: Generate recommendations
│   │   ├── F-03: Display movie cards
│   │   ├── F-05: Explanation popups
│   │   ├── F-06: User taste sidebar
│   │   ├── F-07: "Surprise Me" button
│   │   └── F-08: Like/dislike feedback
│   │
│   └── 3_📊_Analytics.py                 # Data visualization page
│       ├── F-09: Genre distribution
│       ├── Ratings timeline
│       ├── User activity heatmap
│       └── F-10: Movie similarity explorer
│
├── 🐳 Docker/                            # DEPLOYMENT LAYER
│   ├── Dockerfile                        # Container definition
│   ├── docker-compose.yml                # One-command deployment
│   └── .dockerignore                     # Optimized build context
│
├── 🛠️ scripts/                          # AUTOMATION SCRIPTS
│   ├── train_model.sh                    # Training execution wrapper
│   ├── download_dataset.sh               # Dataset download helper
│   └── test_integrity.py                 # Standalone integrity test
│
├── 📋 requirements.txt                   # Python dependencies
├── 📖 README.md                          # Main documentation
├── 📐 ARCHITECTURE.md                    # This file
├── 📝 todo.md                            # Development checklist
├── 📄 PRD.md                             # Product requirements
├── .gitignore                            # Git exclusions
└── .env.example                          # Configuration template
```

---

## 🔄 Data Flow & Processing Pipeline

### 1. **Initialization Flow (App Startup)**

```
┌─────────────────────────────────────────────────────────────┐
│ 1. STREAMLIT APP STARTS (app/main.py)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. DATA INTEGRITY CHECK (src/data_processing.py)            │
│    ├─ check_data_integrity()                                │
│    ├─ ✅ SUCCESS: Log "[INFO] All files found"              │
│    └─ ❌ FAILURE: Display error + download instructions     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. LOAD PRE-TRAINED MODEL (@st.cache_resource)              │
│    ├─ load_model() from recommendation_engine.py            │
│    └─ Model cached in memory for instant inference          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. LOAD DATASET (@st.cache_data)                            │
│    ├─ load_movies() → movies.csv in DataFrame               │
│    └─ load_ratings() → ratings.csv (sampled for UI)         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. APP READY - Display Home Page                            │
└─────────────────────────────────────────────────────────────┘
```

### 2. **Recommendation Generation Flow (F-02)**

```
USER INPUT (User ID: 123)
    │
    ▼
┌─────────────────────────────────────────────────┐
│ VALIDATE INPUT                                  │
│ ├─ Check if user exists in dataset             │
│ └─ Handle invalid IDs gracefully                │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ GET USER HISTORY                                │
│ ├─ Query ratings.csv for user's rated movies   │
│ └─ Store rated_movie_ids                        │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ GENERATE PREDICTIONS                            │
│ ├─ For each movie in catalog:                  │
│ │   ├─ Skip if already rated                   │
│ │   └─ model.predict(user_id, movie_id)        │
│ ├─ Sort by predicted rating (descending)       │
│ └─ Return top N=10 movies                       │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ ENRICH RECOMMENDATIONS                          │
│ ├─ Join with movies.csv (title, genres)        │
│ ├─ Generate explanations (F-05)                │
│ └─ Format for display (F-03)                    │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ DISPLAY IN STREAMLIT                            │
│ ├─ Render movie cards                          │
│ ├─ Show taste profile sidebar (F-06)           │
│ └─ Add interaction buttons (F-07, F-08)        │
└─────────────────────────────────────────────────┘
```

### 3. **Explanation Generation Flow (F-05 - XAI)**

```
FOR EACH RECOMMENDED MOVIE:
    │
    ▼
┌──────────────────────────────────────────────────┐
│ STRATEGY 1: Content-Based Similarity            │
│ ├─ Extract genres of recommended movie          │
│ ├─ Find user's top-rated movies in same genres  │
│ └─ "Because you rated 'Movie X' highly..."      │
└─────────────────┬────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────┐
│ STRATEGY 2: Collaborative Filtering              │
│ ├─ Find users similar to current user           │
│ ├─ Check if similar users rated this movie high │
│ └─ "Users like you loved this movie..."         │
└─────────────────┬────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────┐
│ STRATEGY 3: Genre Preference                     │
│ ├─ User's top genres from taste profile         │
│ ├─ Match with recommended movie's genres        │
│ └─ "Matches your love for Sci-Fi and Action"    │
└─────────────────┬────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────┐
│ FALLBACK: High Global Rating                     │
│ └─ "This critically acclaimed film..."           │
└──────────────────────────────────────────────────┘
```

---

## 🧠 Machine Learning Model Details

### SVD (Singular Value Decomposition) - Matrix Factorization

**Algorithm Choice Rationale:**
- ✅ State-of-the-art for collaborative filtering
- ✅ Handles sparse matrices efficiently (32M ratings across millions of user-movie pairs)
- ✅ Captures latent factors (hidden patterns in user preferences)
- ✅ Proven performance on MovieLens datasets

**Mathematical Foundation:**
```
Rating Matrix R ≈ U × Σ × V^T

Where:
- R: User-Movie rating matrix (sparse)
- U: User latent factor matrix (users × k factors)
- Σ: Diagonal matrix of singular values
- V: Movie latent factor matrix (movies × k factors)
- k: Number of latent factors (hyperparameter)

Prediction for user u and movie i:
r̂_ui = μ + b_u + b_i + q_i^T · p_u

Where:
- μ: Global mean rating
- b_u: User bias (tendency to rate high/low)
- b_i: Movie bias (generally well-rated or not)
- q_i: Movie latent factor vector
- p_u: User latent factor vector
```

**Hyperparameter Configuration:**
```python
{
    "n_factors": 100,        # Number of latent factors (k)
    "n_epochs": 20,          # Training iterations
    "lr_all": 0.005,         # Learning rate (SGD)
    "reg_all": 0.02,         # Regularization (prevent overfitting)
    "random_state": 42       # Reproducibility
}
```

**Training Process:**
1. Load 32M ratings from `ratings.csv`
2. Split: 80% training, 20% test
3. Train SVD model using Stochastic Gradient Descent (SGD)
4. Evaluate on test set: RMSE < 0.87 (success criteria)
5. Serialize model with joblib: `models/svd_model.pkl`

**Inference Optimization:**
- Pre-compute user and item latent vectors
- Cache model in memory (@st.cache_resource)
- Vectorized prediction (batch predict for all movies)
- Target: < 2 seconds for Top-10 recommendations

---

## 🔒 Data Integrity & Error Handling (NF-01)

### Implementation Strategy

**File: `src/data_processing.py`**

```python
def check_data_integrity() -> Tuple[bool, List[str]]:
    """
    Validates presence of all required dataset files.
    
    Returns:
        (success: bool, missing_files: List[str])
    """
    required_files = [
        "data/ml-32m/ratings.csv",
        "data/ml-32m/movies.csv",
        "data/ml-32m/links.csv",
        "data/ml-32m/tags.csv"
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        error_msg = f"""
        ❌ DATA INTEGRITY CHECK FAILED
        
        Missing files: {', '.join(missing)}
        Expected location: {os.path.abspath('data/ml-32m/')}
        
        🔧 ACTION REQUIRED:
        1. Download MovieLens 32M dataset from:
           http://grouplens.org/datasets/movielens/latest/
        2. Extract the archive
        3. Place all files in: data/ml-32m/
        4. Restart the application
        """
        return False, missing, error_msg
    
    return True, [], None
```

**Integration in Streamlit:**
```python
# app/main.py
st.set_page_config(page_title="CineMatch", page_icon="🎬")

success, missing, error = check_data_integrity()
if not success:
    st.error(error)
    st.stop()  # Halt execution gracefully
else:
    st.success("✅ All dataset files found")
```

---

## 🎨 User Interface Design Principles

### Streamlit Multi-Page Architecture

**Navigation Structure:**
```
Sidebar:
├── 🏠 Home (Overview)
├── 🎬 Recommend (Core Feature)
└── 📊 Analytics (Insights)
```

**Design Philosophy:**
1. **Simplicity First**: Clean, uncluttered interface
2. **Progressive Disclosure**: Show complexity only when needed
3. **Immediate Feedback**: Loading spinners, success messages
4. **Error Resilience**: Graceful degradation, clear error messages

**Component Hierarchy (Recommend Page):**
```
┌─────────────────────────────────────────────────┐
│ PAGE HEADER: "Get Your Personalized Picks"     │
└─────────────────────────────────────────────────┘
┌─────────────────────┬───────────────────────────┐
│ MAIN CONTENT        │ SIDEBAR                   │
│                     │                           │
│ ┌─────────────────┐ │ ┌───────────────────────┐│
│ │ User ID Input   │ │ │ 👤 Your Taste Profile ││
│ │ [123         ]  │ │ ├───────────────────────┤│
│ │ [Get Recs] [🎲]│ │ │ Top Genres:           ││
│ └─────────────────┘ │ │ • Drama (35%)         ││
│                     │ │ • Action (28%)        ││
│ ┌─────────────────┐ │ │                       ││
│ │ MOVIE CARD #1   │ │ │ Avg Rating: 4.2⭐    ││
│ │ ┌─────────────┐ │ │ │                       ││
│ │ │ [Poster]    │ │ │ │ Top Rated:            ││
│ │ └─────────────┘ │ │ │ 1. The Shawshank...   ││
│ │ Title: Movie X  │ │ │ 2. Pulp Fiction      ││
│ │ Genres: Action  │ │ └───────────────────────┘│
│ │ Predicted: 4.5⭐│ │                           │
│ │ [Explain] [👍] │ │                           │
│ └─────────────────┘ │                           │
│ ...                 │                           │
└─────────────────────┴───────────────────────────┘
```

---

## 🐳 Docker Containerization Strategy

### Dockerfile Best Practices

```dockerfile
# Multi-stage build for optimization
FROM python:3.9-slim as builder

# Install dependencies in builder stage
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Final lightweight image
FROM python:3.9-slim
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run app
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  cinematch:
    build: .
    container_name: cinematch-demo
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data          # Persist dataset
      - ./models:/app/models      # Persist models
    environment:
      - STREAMLIT_THEME_BASE=light
      - STREAMLIT_SERVER_HEADLESS=true
    restart: unless-stopped
```

**Deployment Command:**
```bash
docker-compose up --build
```

---

## 📊 Performance & Scalability Considerations

### Current Optimizations (V1.0.0)
- ✅ Pre-trained model (no real-time training)
- ✅ Streamlit caching (@st.cache_data, @st.cache_resource)
- ✅ Vectorized NumPy operations
- ✅ Efficient data loading (chunked reading for large CSVs)

### Future Scalability (Post-V1.0.0)
- 🔮 Redis caching for multi-user scenarios
- 🔮 Model versioning (MLflow integration)
- 🔮 Distributed training (Spark MLlib)
- 🔮 API layer (FastAPI) for production deployment
- 🔮 Real-time retraining pipeline (Kafka + Airflow)

---

## 🧪 Testing Strategy

### Test Pyramid
```
        ┌──────────────────┐
        │   E2E Tests      │  ← Streamlit UI flow
        │   (Manual Demo)  │
        └──────────────────┘
       ┌────────────────────┐
       │ Integration Tests  │   ← Module interactions
       └────────────────────┘
     ┌──────────────────────────┐
     │     Unit Tests           │ ← Function-level testing
     └──────────────────────────┘
```

**Test Coverage Goals:**
- Unit Tests: 80% coverage (core logic)
- Integration Tests: Critical user flows
- E2E Tests: Demo script walkthrough

---

## 📈 Success Metrics & Monitoring

### Technical Metrics
| Metric | Target | Measurement |
|--------|--------|-------------|
| Model RMSE | < 0.87 | Test set evaluation |
| Response Time | < 2s | From input to display |
| UI Load Time | < 3s | Cold start to interactive |
| Explanation Coverage | 80% | % of recs with explanations |

### Demo Success Criteria
- ✅ Professor comprehension < 60 seconds
- ✅ Zero crashes during 5-minute demo
- ✅ All features work on first try
- ✅ "Wow" moments trigger (XAI features)

---

## 🔐 Security & Privacy Notes

**Data Privacy:**
- ✅ No real user PII (MovieLens is anonymized)
- ✅ User IDs are research identifiers, not personal data
- ✅ No data collection or external transmission

**Security Considerations (Production Future):**
- 🔮 Input sanitization for user IDs
- 🔮 Rate limiting for API endpoints
- 🔮 HTTPS/TLS for production deployment
- 🔮 Environment variable management (.env)

---

## 📚 Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit 1.28+ | Rapid prototyping, interactive UI |
| **Backend** | Python 3.9+ | Core application logic |
| **ML Library** | scikit-surprise | SVD collaborative filtering |
| **Data Processing** | Pandas, NumPy | Data manipulation |
| **Visualization** | Plotly, Matplotlib | Interactive charts |
| **Serialization** | Joblib | Model persistence |
| **Containerization** | Docker, Docker Compose | Deployment |
| **Version Control** | Git | Source control |

---

## 🎓 Academic Contribution

**Key Innovations for Master's Thesis:**
1. **Explainable Recommendations**: Bridging the "black box" gap
2. **User Taste Profiling**: Moving beyond simple ratings
3. **Production-Ready Demo**: Real-world software engineering practices
4. **Serendipity Feature**: Balancing exploitation vs. exploration

**Potential Research Questions:**
- How does explainability affect user trust in recommendations?
- What is the optimal balance between accuracy and diversity?
- Can taste profiles improve cold-start problem solutions?

---

## 📞 Support & Maintenance

**Development Contact:**
- Project Lead: [Your Name]
- Repository: [GitHub URL]
- Documentation: This file + README.md

**Known Limitations (V1.0.0):**
- Dataset must be manually downloaded (32M ratings = 600MB+)
- Model training takes 15-30 minutes on standard hardware
- No real-time feedback incorporation (simulated only)

---

*Document Version: 1.0.0*
*Last Updated: October 24, 2025*
*Maintained By: CineMatch Development Team*
