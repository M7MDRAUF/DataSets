# CineMatch V2.1.0 - Module Documentation

## 📋 Overview

Complete documentation for all code modules in the CineMatch recommendation system. This document provides detailed information about each component's purpose, key functions, dependencies, and usage patterns.

---

## 🗂️ Table of Contents

1. [Core Modules (`src/`)](#core-modules-src)
   - [data_processing.py](#data_processingpy)
   - [utils.py](#utilspy)
   - [model_training.py](#model_trainingpy)
   - [recommendation_engine.py](#recommendation_enginepy)

2. [Algorithm Modules (`src/algorithms/`)](#algorithm-modules-srcalgorithms)
   - [algorithm_manager.py](#algorithm_managerpy)
   - [base_recommender.py](#base_recommenderpy)
   - [svd_recommender.py](#svd_recommenderpy)
   - [user_knn_recommender.py](#user_knn_recommenderpy)
   - [item_knn_recommender.py](#item_knn_recommenderpy)
   - [content_based_recommender.py](#content_based_recommenderpy)
   - [hybrid_recommender.py](#hybrid_recommenderpy)

3. [Application Modules (`app/`)](#application-modules-app)
   - [main.py](#mainpy)
   - [pages/1_🏠_Home.py](#pages1_homepy)
   - [pages/2_🎬_Recommend.py](#pages2_recommendpy)
   - [pages/3_📊_Analytics.py](#pages3_analyticspy)

4. [Utility Scripts (`scripts/`)](#utility-scripts-scripts)

---

## Core Modules (`src/`)

### `data_processing.py`

**Purpose**: Data integrity validation (NF-01), dataset loading, and preprocessing.

**Key Functions**:

```python
def check_data_integrity() -> Tuple[bool, List[str], Optional[str]]:
    """
    Validates presence of all required MovieLens 32M dataset files.
    
    Returns:
        (success: bool, missing_files: List[str], error_message: Optional[str])
    
    Files checked:
        - data/ml-32m/ratings.csv (32M ratings)
        - data/ml-32m/movies.csv (movie metadata)
        - data/ml-32m/links.csv (external IDs)
        - data/ml-32m/tags.csv (user tags)
    
    Usage:
        success, missing, error = check_data_integrity()
        if not success:
            st.error(error)
            st.stop()
    """
```

```python
def load_ratings(sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Loads ratings.csv with optional sampling for UI performance.
    
    Args:
        sample_size: Number of ratings to sample (None = all 32M)
    
    Returns:
        DataFrame with columns: userId, movieId, rating, timestamp
    
    Performance:
        - Full 32M: ~15GB RAM, ~30s load time
        - 100K sample: ~200MB RAM, <1s load time
    """
```

```python
def load_movies() -> pd.DataFrame:
    """
    Loads movies.csv with metadata.
    
    Returns:
        DataFrame with columns: movieId, title, genres
        
    Notes:
        - Genres pipe-separated: "Action|Sci-Fi|Thriller"
        - ~87,000 movies total
    """
```

```python
def preprocess_data(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and transforms raw ratings data.
    
    Transformations:
        - Convert timestamps to datetime
        - Handle missing values
        - Filter out invalid ratings
        - Sort by userId and timestamp
    """
```

```python
def create_user_genre_matrix(ratings_df: pd.DataFrame, 
                             movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates user taste profiles based on genre preferences.
    
    Returns:
        Matrix shape: (n_users, n_genres)
        Values: Normalized preference scores (0-1)
    
    Used for:
        - F-06: Taste Profile sidebar
        - XAI explanations
    """
```

**Dependencies**:
- pandas
- numpy
- pathlib

**Integration Points**:
- Called by `app/main.py` at startup (NF-01 check)
- Used by all algorithm modules for data loading
- Cached with `@st.cache_data` for performance

---

### `utils.py`

**Purpose**: Explainable AI (XAI) logic and shared utility functions.

**Key Functions**:

```python
def generate_explanation(user_id: int, 
                        movie_id: int, 
                        predicted_rating: float,
                        algorithm: str,
                        ratings_df: pd.DataFrame,
                        movies_df: pd.DataFrame) -> str:
    """
    Generates algorithm-specific explanations for recommendations (F-05).
    
    Explanation Strategies:
        1. Content-Based Similarity (if applicable)
        2. Collaborative Filtering patterns
        3. Genre preference matching
        4. Fallback: High global rating
    
    Returns:
        Human-readable explanation string
        Example: "Because you rated 'Inception' (Sci-Fi) highly (4.5★), 
                 you might enjoy this similar Sci-Fi thriller."
    """
```

```python
def get_user_top_genres(user_id: int, 
                       ratings_df: pd.DataFrame,
                       movies_df: pd.DataFrame,
                       top_n: int = 3) -> List[Tuple[str, float]]:
    """
    Identifies user's favorite genres for taste profile.
    
    Returns:
        List of (genre, preference_score) tuples
        Example: [("Drama", 0.35), ("Action", 0.28), ("Comedy", 0.18)]
    """
```

```python
def calculate_diversity_score(recommendations: List[int],
                              movies_df: pd.DataFrame) -> float:
    """
    Measures recommendation diversity (F-08: Serendipity).
    
    Returns:
        Score 0-1 (1 = maximum genre diversity)
    """
```

**Dependencies**:
- pandas
- numpy
- collections

---

### `model_training.py`

**Purpose**: Legacy SVD model training pipeline.

**Key Functions**:

```python
def train_svd_model(ratings_df: pd.DataFrame,
                   n_factors: int = 100,
                   n_epochs: int = 20) -> SVD:
    """
    Trains SVD model on full dataset.
    
    Hyperparameters:
        - n_factors: 100 (latent dimensions)
        - n_epochs: 20 (training iterations)
        - lr_all: 0.005 (learning rate)
        - reg_all: 0.02 (regularization)
    
    Training Time:
        - Full 32M: ~15-30 minutes on standard hardware
    
    Returns:
        Trained SVD model
    """
```

```python
def evaluate_model(model, test_data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculates performance metrics.
    
    Returns:
        {
            "RMSE": 0.6829,
            "MAE": 0.5234,
            "Coverage": 0.241
        }
    """
```

```python
def save_model(model, path: str = "models/svd_model.pkl"):
    """
    Serializes trained model with joblib.
    
    File Size: ~10-50MB (depending on n_factors)
    """
```

**Note**: For V2.1.0, User-KNN, Item-KNN, and Content-Based models are trained using dedicated scripts in `scripts/`.

---

### `recommendation_engine.py`

**Purpose**: Legacy recommendation interface (pre-AlgorithmManager architecture).

**Status**: Partially deprecated in V2.1.0, replaced by `AlgorithmManager` for multi-algorithm support.

**Key Functions**:

```python
@st.cache_resource
def load_model() -> SVD:
    """
    Loads pre-trained SVD model with caching.
    
    Cached in Streamlit session for instant inference.
    """
```

```python
def get_recommendations(user_id: int, 
                       model,
                       ratings_df: pd.DataFrame,
                       movies_df: pd.DataFrame,
                       n: int = 10) -> List[Dict]:
    """
    Generates Top-N recommendations (F-02).
    
    Returns:
        List of recommendation dictionaries:
        [
            {
                "movieId": 123,
                "title": "Inception",
                "genres": "Sci-Fi|Thriller",
                "predicted_rating": 4.5
            },
            ...
        ]
    """
```

---

## Algorithm Modules (`src/algorithms/`)

### `algorithm_manager.py`

**Purpose**: Central algorithm coordinator using Factory + Singleton design patterns.

**Design Patterns**:
1. **Singleton**: Single instance manages all algorithms
2. **Factory**: Creates algorithm instances based on string names
3. **Lazy Loading**: Models loaded on-demand, then cached

**Key Class**:

```python
class AlgorithmManager:
    """
    Thread-safe singleton for managing 5 recommendation algorithms.
    
    Responsibilities:
        - Algorithm lifecycle management
        - Pre-trained model loading
        - Caching and performance optimization
        - Metrics calculation and benchmarking
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton implementation with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_algorithm(self, 
                     algorithm_name: str,
                     ratings_df: pd.DataFrame = None,
                     movies_df: pd.DataFrame = None) -> BaseRecommender:
        """
        Factory method: Creates/retrieves algorithm instance.
        
        Args:
            algorithm_name: "SVD", "User-KNN", "Item-KNN", "Content-Based", "Hybrid"
            ratings_df: Required for first-time initialization
            movies_df: Required for Content-Based
        
        Returns:
            BaseRecommender implementation
        
        Caching:
            - First call: Loads pre-trained model (~1-2s)
            - Subsequent calls: Returns cached instance (<0.01s)
        """
    
    def switch_algorithm(self, 
                        new_algorithm: str,
                        ratings_df: pd.DataFrame,
                        movies_df: pd.DataFrame):
        """
        Intelligent algorithm switching with state management.
        """
    
    def get_algorithm_metrics(self, 
                             algorithm_name: str) -> Dict[str, float]:
        """
        Retrieves performance metrics for single algorithm.
        
        Returns:
            {
                "RMSE": 0.6829,
                "MAE": 0.5234,
                "Coverage": 0.241,
                "Load_Time": 1.5
            }
        """
    
    def get_all_algorithm_metrics(self) -> pd.DataFrame:
        """
        Benchmarks all 5 algorithms for Analytics dashboard.
        
        Returns:
            DataFrame with comparison table (used in pages/3_Analytics.py)
        """
    
    def _try_load_pretrained_model(self, algorithm_name: str):
        """
        Attempts to load pre-trained model from models/ directory.
        
        Model Paths:
            - SVD: models/svd_model.pkl
            - User-KNN: models/user_knn_model.pkl (266MB)
            - Item-KNN: models/item_knn_model.pkl (260MB)
            - Content-Based: models/content_based_model.pkl (~300MB)
        
        Error Handling:
            - If model not found: Returns None (will train on-demand)
            - If Git LFS pointers: Logs error, suggests git lfs pull
        """
```

**Performance**:
- Singleton creation: <0.001s
- First algorithm load: 0.8-2s (depends on model size)
- Algorithm switch: <0.01s (cached)
- Metrics calculation: ~7s (all 5 algorithms, emergency optimized)

---

### `base_recommender.py`

**Purpose**: Abstract Base Class (ABC) defining common interface for all algorithms.

**Key Class**:

```python
from abc import ABC, abstractmethod

class BaseRecommender(ABC):
    """
    Contract for all recommendation algorithms.
    
    Ensures consistent API across SVD, KNN, Content-Based, and Hybrid.
    """
    
    @abstractmethod
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame = None):
        """
        Trains the recommendation model.
        
        Args:
            ratings_df: Training data (userId, movieId, rating)
            movies_df: Movie metadata (required for Content-Based)
        """
        pass
    
    @abstractmethod
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predicts rating for user-movie pair.
        
        Returns:
            Predicted rating (1.0 - 5.0 scale)
        """
        pass
    
    @abstractmethod
    def recommend(self, 
                 user_id: int, 
                 ratings_df: pd.DataFrame,
                 movies_df: pd.DataFrame,
                 n: int = 10) -> List[Dict]:
        """
        Generates Top-N recommendations for user.
        
        Returns:
            List of dicts: [{"movieId": int, "predicted_rating": float}, ...]
        """
        pass
    
    def get_name(self) -> str:
        """Returns algorithm name for display."""
        return self.__class__.__name__
```

**Usage Pattern**:

```python
# All algorithms follow same interface
recommender: BaseRecommender = manager.get_algorithm("Content-Based")
recommender.fit(ratings_df, movies_df)
prediction = recommender.predict(user_id=123, movie_id=456)
recommendations = recommender.recommend(user_id=123, n=10)
```

---

### `svd_recommender.py`

**Purpose**: Singular Value Decomposition matrix factorization algorithm.

**Implementation**: 350 lines

**Key Methods**:

```python
class SVDRecommender(BaseRecommender):
    """
    Matrix factorization collaborative filtering.
    
    Algorithm: Stochastic Gradient Descent (SGD)
    Model Size: ~10-50MB
    RMSE: 0.6829 (100K sample), 0.8406 (full 32M)
    """
    
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame = None):
        """
        Hyperparameters:
            - n_factors: 100
            - n_epochs: 20
            - lr_all: 0.005
            - reg_all: 0.02
        
        Training Time: 15-30 minutes on full 32M dataset
        """
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Prediction formula:
            r̂_ui = μ + b_u + b_i + q_i^T · p_u
        
        Speed: <0.001s per prediction
        """
    
    def recommend(self, user_id: int, ratings_df: pd.DataFrame, 
                 movies_df: pd.DataFrame, n: int = 10) -> List[Dict]:
        """
        Vectorized batch prediction for all unrated movies.
        
        Performance: ~2s for Top-10 from 87K movies
        """
```

**Metrics**:
- **RMSE**: 0.6829 (sample), 0.8406 (full)
- **Coverage**: 24.1%
- **Load Time**: <0.1s
- **Inference**: ~2s

---

### `user_knn_recommender.py`

**Purpose**: User-based collaborative filtering with KNN.

**Implementation**: 681 lines (most complex due to optimizations)

**Key Methods**:

```python
class UserKNNRecommender(BaseRecommender):
    """
    Collaborative filtering based on user similarity.
    
    Pre-trained Model: 266MB (similarity matrix for all users)
    RMSE: 0.7234
    Load Time: ~1.5s
    """
    
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame = None):
        """
        Builds user-user similarity matrix using cosine similarity.
        
        Matrix Size: (n_users, n_users) - sparse representation
        Training Time: 2-4 hours on full 32M dataset
        """
    
    def _batch_predict_ratings(self, 
                              user_id: int,
                              movie_ids: List[int]) -> np.ndarray:
        """
        Emergency optimization: Vectorized batch predictions.
        
        Performance: 200x faster than loop-based approach
        Previous: 124s for 5K movies
        Optimized: 0.6s for 5K movies
        """
    
    def recommend(self, user_id: int, ratings_df: pd.DataFrame,
                 movies_df: pd.DataFrame, n: int = 10) -> List[Dict]:
        """
        Smart candidate sampling: Reduces search space 80K → 5K.
        
        Strategy:
            1. Find K=50 nearest neighbors
            2. Get movies they rated highly (rating ≥ 4.0)
            3. Sample top 5K candidates
            4. Batch predict ratings
            5. Return Top-10
        
        Result: 200x speedup with minimal accuracy loss
        """
```

**Optimizations**:
- Smart candidate sampling (5K/80K)
- Vectorized numpy operations
- Sparse matrix representations
- Pre-trained model caching

---

### `item_knn_recommender.py`

**Purpose**: Item-based collaborative filtering with KNN.

**Pre-trained Model**: 260MB (item similarity matrix)

**Key Methods**:

```python
class ItemKNNRecommender(BaseRecommender):
    """
    Collaborative filtering based on item similarity.
    
    Advantage: More stable than User-KNN (items change less than users)
    RMSE: 0.7456
    Load Time: ~1.0s
    """
    
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame = None):
        """
        Builds item-item similarity matrix.
        
        Training Time: 2-4 hours on full dataset
        """
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Weighted average of similar items' ratings.
        
        Formula:
            r̂_ui = Σ(sim(i,j) × r_uj) / Σ|sim(i,j)|
            for all items j rated by user u
        """
    
    def recommend(self, user_id: int, ratings_df: pd.DataFrame,
                 movies_df: pd.DataFrame, n: int = 10) -> List[Dict]:
        """
        Similar vectorized optimizations as User-KNN.
        
        Performance: ~1s for Top-10
        """
```

---

### `content_based_recommender.py`

**Purpose**: Content-based filtering using TF-IDF and cosine similarity.

**Implementation**: 979 lines (most feature engineering)

**Key Methods**:

```python
class ContentBasedRecommender(BaseRecommender):
    """
    Recommendation based on movie content features (genres, tags, titles).
    
    Model: TF-IDF vectorizer + precomputed similarity matrix
    Size: ~300MB
    Coverage: 100% (no cold-start problem)
    """
    
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        """
        Feature Engineering Pipeline:
            1. Load movies.csv (genres) + tags.csv (user tags)
            2. Extract keywords from titles
            3. Combine: "Action Sci-Fi space adventure inception dream"
            4. TF-IDF vectorization (5,000 features)
            5. Compute cosine similarity matrix (87K × 87K, sparse)
        
        Training Time: 10-20 minutes
        Model Size: ~300MB (vectorizer + feature matrix + similarity)
        """
    
    def _build_user_profile(self, 
                           user_id: int,
                           ratings_df: pd.DataFrame) -> np.ndarray:
        """
        Aggregates user's rated movies into feature vector.
        
        Formula:
            u_profile = Σ(rating_i × movie_features_i) / Σ(rating_i)
        
        Result: Weighted average of liked movies' features
        """
    
    def _compute_similarities(self, 
                             user_profile: np.ndarray,
                             candidate_movies: List[int]) -> np.ndarray:
        """
        Cosine similarity between user profile and movie features.
        
        Formula:
            sim(u, m) = (u · m) / (||u|| × ||m||)
        
        Vectorized: ~0.1s for 87K movies
        """
    
    def recommend(self, user_id: int, ratings_df: pd.DataFrame,
                 movies_df: pd.DataFrame, n: int = 10) -> List[Dict]:
        """
        Steps:
            1. Build user profile from ratings
            2. Compute similarities to all unrated movies
            3. Rank by similarity score
            4. Return Top-10
        
        Performance: <1s for recommendations
        """
```

**Features Used**:
- **Genres**: Action, Sci-Fi, Drama, etc. (pipe-separated)
- **Tags**: User-generated tags from tags.csv
- **Title Keywords**: Extracted meaningful words

**TF-IDF Configuration**:
```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=2,
    max_df=0.8
)
```

---

### `hybrid_recommender.py`

**Purpose**: Ensemble of 4 sub-algorithms (SVD, User-KNN, Item-KNN, Content-Based).

**Implementation**: 781 lines

**Key Methods**:

```python
class HybridRecommender(BaseRecommender):
    """
    Weighted ensemble combining 4 recommendation algorithms.
    
    Weights (adaptive):
        - SVD: 0.33
        - User-KNN: 0.22
        - Item-KNN: 0.25
        - Content-Based: 0.20
    
    RMSE: 0.6543 (best performance)
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Args:
            weights: Custom algorithm weights (default: adaptive)
        """
        self.weights = weights or {
            "SVD": 0.33,
            "User-KNN": 0.22,
            "Item-KNN": 0.25,
            "Content-Based": 0.20
        }
        self.algorithms = {}
    
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        """
        Trains all 4 sub-algorithms.
        
        Training Time: Sum of all algorithms (~4-8 hours total)
        Note: In production, use pre-trained models
        """
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Weighted average of all algorithm predictions.
        
        Formula:
            r̂_hybrid = Σ(w_i × r̂_i) where w_i = algorithm weight
        
        Fallback: If algorithm fails, use remaining algorithms
        """
    
    def _calculate_hybrid_rmse(self, 
                              test_data: pd.DataFrame) -> float:
        """
        Emergency optimized RMSE calculation.
        
        Previous: 2+ hours (full 32M test set)
        Optimized: ~7s (stratified sampling + vectorization)
        
        Strategy:
            1. Sample 10K test ratings (stratified by rating distribution)
            2. Batch predictions from all algorithms
            3. Weighted ensemble
            4. Calculate RMSE on sample
        """
    
    def recommend(self, user_id: int, ratings_df: pd.DataFrame,
                 movies_df: pd.DataFrame, n: int = 10) -> List[Dict]:
        """
        Combines recommendations from all algorithms.
        
        Strategy:
            1. Get Top-20 from each algorithm
            2. Merge and deduplicate
            3. Re-rank by weighted ensemble scores
            4. Return Top-10
        
        Benefits:
            - Diversity: Multiple algorithm perspectives
            - Robustness: Fallback if one algorithm fails
            - Accuracy: Best overall RMSE
        """
```

**Performance**:
- **RMSE**: 0.6543 (best of all 5 algorithms)
- **Coverage**: Highest (combines all algorithms)
- **Load Time**: ~3s (loads all 4 models)
- **Inference**: ~3s (4 algorithm predictions + merging)

---

## Application Modules (`app/`)

### `main.py`

**Purpose**: Streamlit application entry point.

**Implementation**: 200 lines

**Key Components**:

```python
import streamlit as st
from src.data_processing import check_data_integrity

# Page configuration
st.set_page_config(
    page_title="CineMatch V2.1.0",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# NF-01: Data Integrity Check (CRITICAL - must run first)
success, missing_files, error_msg = check_data_integrity()
if not success:
    st.error(error_msg)
    st.markdown("""
    ### 🔧 Quick Fix:
    1. Download MovieLens 32M from: http://grouplens.org/datasets/movielens/
    2. Extract to: `data/ml-32m/`
    3. Run: `git lfs pull` (for pre-trained models)
    4. Restart application
    """)
    st.stop()  # Halt execution gracefully

st.success("✅ All dataset files found!")

# Home page content
st.title("🎬 CineMatch V2.1.0")
st.markdown("""
### Multi-Algorithm Movie Recommendation Engine

Choose an option from the sidebar:
- 🏠 **Home**: Project overview
- 🎬 **Recommend**: Get personalized movie recommendations
- 📊 **Analytics**: Performance benchmarks and insights
""")

# Algorithm overview
st.header("🧠 5 Recommendation Algorithms")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("SVD", "RMSE: 0.68")
with col2:
    st.metric("User-KNN", "RMSE: 0.72")
with col3:
    st.metric("Item-KNN", "RMSE: 0.75")
with col4:
    st.metric("Content-Based", "100% Coverage")
with col5:
    st.metric("Hybrid", "RMSE: 0.65")
```

**Flow**:
1. Page configuration
2. **NF-01 check** (blocks if dataset missing)
3. Display welcome message
4. Algorithm overview metrics

---

### `pages/1_🏠_Home.py`

**Purpose**: Project overview and visualizations.

**Implementation**: 547 lines

**Key Sections**:

```python
# Algorithm selector
algorithm = st.selectbox(
    "Select Algorithm:",
    ["SVD", "User-KNN", "Item-KNN", "Content-Based", "Hybrid"]
)

# Display algorithm-specific details
if algorithm == "Content-Based":
    st.markdown("""
    ### Content-Based Filtering
    - **Method**: TF-IDF + Cosine Similarity
    - **Features**: Genres, Tags, Title Keywords
    - **Model Size**: ~300MB
    - **Coverage**: 100% (no cold-start)
    """)

# Dataset statistics
st.header("📊 Dataset Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Ratings", "32,000,000+")
col2.metric("Movies", "87,000+")
col3.metric("Users", "330,000+")

# Feature highlights
st.header("✨ Key Features")
features = {
    "F-01": "User Input & Validation",
    "F-02": "Top-N Recommendations (N=10)",
    "F-03": "Movie Display (Title, Genres, Rating)",
    "F-04": "Algorithm Selection (5 options)",
    "F-05": "Explainable Recommendations (XAI)",
    "F-06": "User Taste Profile",
    "F-07": "Feedback Collection",
    "F-08": "Serendipity (Random Discovery)",
    "F-09": "Analytics Dashboard",
    "F-10": "Algorithm Benchmarking"
}

for feature_id, description in features.items():
    st.markdown(f"- **{feature_id}**: {description}")
```

---

### `pages/2_🎬_Recommend.py`

**Purpose**: Core recommendation feature (F-02).

**Implementation**: 528 lines

**Key Components**:

```python
from src.algorithms.algorithm_manager import AlgorithmManager

# Initialize manager
manager = AlgorithmManager()

# Sidebar: Algorithm selection
algorithm_name = st.sidebar.selectbox(
    "🧠 Select Algorithm:",
    ["SVD", "User-KNN", "Item-KNN", "Content-Based", "Hybrid"]
)

# Main content: User input
user_id = st.number_input(
    "Enter User ID:",
    min_value=1,
    max_value=330000,
    value=123,
    step=1
)

# Get recommendations button
if st.button("🎬 Get Recommendations"):
    with st.spinner(f"Generating recommendations using {algorithm_name}..."):
        # Load algorithm
        recommender = manager.get_algorithm(
            algorithm_name, 
            ratings_df, 
            movies_df
        )
        
        # Generate recommendations
        recommendations = recommender.recommend(
            user_id=user_id,
            ratings_df=ratings_df,
            movies_df=movies_df,
            n=10
        )
        
        # Display results
        st.success(f"✅ Found {len(recommendations)} recommendations!")
        
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"#{i} {rec['title']} ({rec['predicted_rating']:.2f}⭐)"):
                st.markdown(f"**Genres**: {rec['genres']}")
                
                # F-05: Generate explanation
                explanation = generate_explanation(
                    user_id, 
                    rec['movieId'],
                    rec['predicted_rating'],
                    algorithm_name,
                    ratings_df,
                    movies_df
                )
                st.info(explanation)
                
                # F-07: Feedback buttons
                col1, col2 = st.columns(2)
                col1.button("👍 Like", key=f"like_{i}")
                col2.button("👎 Dislike", key=f"dislike_{i}")

# Sidebar: F-06 Taste Profile
st.sidebar.header("👤 Your Taste Profile")
top_genres = get_user_top_genres(user_id, ratings_df, movies_df)
for genre, score in top_genres:
    st.sidebar.markdown(f"- **{genre}**: {score:.1%}")

# F-08: Serendipity button
if st.sidebar.button("🎲 Surprise Me!"):
    # Random user discovery
    random_user = np.random.randint(1, 330000)
    st.sidebar.info(f"Showing recommendations for random user #{random_user}")
```

---

### `pages/3_📊_Analytics.py`

**Purpose**: Performance benchmarking and insights (F-09, F-10).

**Implementation**: 687 lines

**Key Features**:

```python
# Algorithm benchmarking table
st.header("🏆 Algorithm Performance Comparison")

manager = AlgorithmManager()
metrics_df = manager.get_all_algorithm_metrics()

st.dataframe(metrics_df, use_container_width=True)

# Columns: Algorithm, RMSE, MAE, Coverage, Load Time, Inference Time

# Interactive charts
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 RMSE Comparison",
    "⏱️ Performance",
    "📊 Coverage",
    "🎯 Accuracy Distribution",
    "💡 Insights"
])

with tab1:
    # RMSE bar chart
    fig = px.bar(
        metrics_df,
        x="Algorithm",
        y="RMSE",
        title="Lower is Better",
        color="RMSE",
        color_continuous_scale="RdYlGn_r"
    )
    st.plotly_chart(fig)

with tab2:
    # Load time vs inference time scatter
    fig = px.scatter(
        metrics_df,
        x="Load_Time",
        y="Inference_Time",
        size="Model_Size_MB",
        color="Algorithm",
        hover_data=["RMSE", "Coverage"],
        title="Performance Characteristics"
    )
    st.plotly_chart(fig)

with tab3:
    # Coverage comparison
    fig = px.pie(
        metrics_df,
        values="Coverage",
        names="Algorithm",
        title="Recommendation Coverage"
    )
    st.plotly_chart(fig)

with tab5:
    # Key insights
    st.markdown("""
    ### 🔍 Key Findings
    
    **Best Accuracy**: Hybrid (RMSE: 0.6543)
    - Combines strengths of all 4 sub-algorithms
    - 15% better than single algorithm average
    
    **Fastest**: Content-Based (Load: 0.8s, Inference: 0.5s)
    - No similarity matrix computations needed
    - 100% coverage (no cold-start)
    
    **Most Stable**: Item-KNN (RMSE: 0.7456)
    - Items change less frequently than users
    - Good balance of speed and accuracy
    
    **Best for New Users**: Content-Based
    - No user history required
    - Only needs movie features
    """)
```

---

## Utility Scripts (`scripts/`)

### Training Scripts

**`train_knn_models.py`**: Trains User-KNN and Item-KNN models on full 32M dataset.

```bash
# PowerShell
python scripts/train_knn_models.py --output models/

# Bash
python scripts/train_knn_models.py --output models/
```

**`train_content_based.py`**: Trains Content-Based TF-IDF model.

```bash
python scripts/train_content_based.py --features 5000 --output models/
```

### Validation Scripts

**`validate_system.py`**: End-to-end system validation.

```python
# Checks:
# 1. Data integrity (NF-01)
# 2. All 5 models loadable
# 3. Basic predictions work
# 4. Performance within thresholds
```

**`emergency_save_model.py`**: Emergency model serialization during training.

---

## 🔗 Integration Patterns

### Algorithm Usage Pattern

```python
# Standard workflow
from src.algorithms.algorithm_manager import AlgorithmManager

manager = AlgorithmManager()
recommender = manager.get_algorithm("Content-Based", ratings_df, movies_df)

# First call: Loads model (~1s)
recommendations = recommender.recommend(user_id=123, n=10)

# Subsequent calls: Cached (<0.01s)
recommendations = recommender.recommend(user_id=456, n=10)
```

### Streamlit Caching

```python
@st.cache_resource  # Model caching (persists across sessions)
def load_algorithm_manager():
    return AlgorithmManager()

@st.cache_data  # Data caching (serializable)
def load_datasets():
    ratings_df = load_ratings()
    movies_df = load_movies()
    return ratings_df, movies_df
```

---

## 📊 Performance Summary

| Module | Lines | Load Time | Key Responsibility |
|--------|-------|-----------|-------------------|
| data_processing.py | ~300 | <1s | NF-01 checks, data loading |
| utils.py | ~250 | <0.1s | XAI explanations |
| algorithm_manager.py | 562 | <0.01s | Algorithm coordination |
| content_based_recommender.py | 979 | ~0.8s | TF-IDF recommendations |
| user_knn_recommender.py | 681 | ~1.5s | Collaborative filtering |
| hybrid_recommender.py | 781 | ~3s | Ensemble predictions |
| main.py | 200 | <0.1s | App entry point |
| pages/2_Recommend.py | 528 | <0.1s | Core feature UI |
| pages/3_Analytics.py | 687 | <0.5s | Benchmarking dashboard |

**Total**: ~15,000+ lines across 40+ files

---

*Document Version: 1.0*
*Last Updated: November 13, 2025*
*Part of CineMatch V2.1.0 Documentation Suite*
