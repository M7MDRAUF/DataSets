# CineMatch V2.1.0 - API & Interface Documentation

## 📋 Overview

Complete API reference for CineMatch's public interfaces and integration points. This document covers the AlgorithmManager API, BaseRecommender contract, and all public methods for algorithm interaction.

---

## 🎯 Core API: AlgorithmManager

### Singleton Access

```python
from src.algorithms.algorithm_manager import AlgorithmManager

# Get singleton instance
manager = AlgorithmManager()

# Thread-safe: Same instance across all calls
manager2 = AlgorithmManager()
assert manager is manager2  # True
```

### Factory Method: `get_algorithm()`

**Purpose**: Creates or retrieves algorithm instance with lazy loading and caching.

```python
def get_algorithm(
    algorithm_name: str,
    ratings_df: pd.DataFrame = None,
    movies_df: pd.DataFrame = None
) -> BaseRecommender
```

**Parameters**:
- `algorithm_name` (str): One of `"SVD"`, `"User-KNN"`, `"Item-KNN"`, `"Content-Based"`, `"Hybrid"`
- `ratings_df` (pd.DataFrame, optional): Required for first-time algorithm initialization
  - Columns: `userId`, `movieId`, `rating`, `timestamp`
- `movies_df` (pd.DataFrame, optional): Required for Content-Based algorithm
  - Columns: `movieId`, `title`, `genres`

**Returns**:
- `BaseRecommender`: Algorithm instance implementing the recommender interface

**Raises**:
- `ValueError`: If `algorithm_name` not recognized
- `FileNotFoundError`: If pre-trained model not found and training data not provided
- `RuntimeError`: If Git LFS pointer file detected (model not pulled)

**Usage Examples**:

```python
# Example 1: Load pre-trained model (recommended)
manager = AlgorithmManager()
recommender = manager.get_algorithm("User-KNN")
# First call: Loads 266MB model (~1.5s)
# Subsequent calls: Returns cached instance (<0.01s)

# Example 2: Initialize with data (if model not found)
ratings_df = load_ratings()
movies_df = load_movies()
recommender = manager.get_algorithm(
    "Content-Based",
    ratings_df=ratings_df,
    movies_df=movies_df
)

# Example 3: Switch between algorithms
svd_rec = manager.get_algorithm("SVD")
knn_rec = manager.get_algorithm("User-KNN")
hybrid_rec = manager.get_algorithm("Hybrid")
```

**Performance**:
- First call: 0.8-2s (model loading)
- Cached call: <0.01s
- Memory: Models cached in RAM (total ~800MB for all 5)

---

### Algorithm Switching: `switch_algorithm()`

**Purpose**: Intelligent algorithm switching with state management.

```python
def switch_algorithm(
    new_algorithm: str,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame
)
```

**Parameters**:
- `new_algorithm` (str): Target algorithm name
- `ratings_df` (pd.DataFrame): Ratings data
- `movies_df` (pd.DataFrame): Movies metadata

**Example**:

```python
manager = AlgorithmManager()

# Start with SVD
manager.switch_algorithm("SVD", ratings_df, movies_df)
recommendations1 = manager.get_algorithm("SVD").recommend(user_id=123, n=10)

# Switch to Hybrid
manager.switch_algorithm("Hybrid", ratings_df, movies_df)
recommendations2 = manager.get_algorithm("Hybrid").recommend(user_id=123, n=10)
```

---

### Metrics Retrieval: `get_algorithm_metrics()`

**Purpose**: Retrieve performance metrics for a single algorithm.

```python
def get_algorithm_metrics(algorithm_name: str) -> Dict[str, float]
```

**Returns**:
```python
{
    "RMSE": 0.6829,         # Root Mean Square Error (lower is better)
    "MAE": 0.5234,          # Mean Absolute Error
    "Coverage": 0.241,      # Percentage of items recommendable
    "Load_Time": 1.5,       # Model load time (seconds)
    "Model_Size_MB": 266.0  # Disk size
}
```

**Example**:

```python
manager = AlgorithmManager()
metrics = manager.get_algorithm_metrics("User-KNN")
print(f"User-KNN RMSE: {metrics['RMSE']:.4f}")
```

---

### Benchmarking: `get_all_algorithm_metrics()`

**Purpose**: Compare all 5 algorithms for analytics dashboard.

```python
def get_all_algorithm_metrics() -> pd.DataFrame
```

**Returns**:
```python
   Algorithm  RMSE    MAE  Coverage  Load_Time  Model_Size_MB
0        SVD  0.6829  0.52      24.1       0.05           10.0
1   User-KNN  0.7234  0.55      45.2       1.50          266.0
2   Item-KNN  0.7456  0.57      42.8       1.00          260.0
3  Content-Based   N/A   N/A     100.0       0.80          300.0
4     Hybrid  0.6543  0.50      68.5       3.00          836.0
```

**Example**:

```python
manager = AlgorithmManager()
df = manager.get_all_algorithm_metrics()

# Find best RMSE
best = df.loc[df['RMSE'].idxmin()]
print(f"Best algorithm: {best['Algorithm']} (RMSE: {best['RMSE']})")
# Output: Best algorithm: Hybrid (RMSE: 0.6543)
```

---

## 🔌 BaseRecommender Interface

All algorithms implement this contract for consistent API.

### Abstract Base Class

```python
from abc import ABC, abstractmethod

class BaseRecommender(ABC):
    """Common interface for all recommendation algorithms."""
    
    @abstractmethod
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame = None):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating for user-movie pair."""
        pass
    
    @abstractmethod
    def recommend(self, user_id: int, ratings_df: pd.DataFrame,
                 movies_df: pd.DataFrame, n: int = 10) -> List[Dict]:
        """Generate Top-N recommendations."""
        pass
```

---

### Method: `fit()`

**Purpose**: Train the recommendation model on provided data.

```python
def fit(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame = None
)
```

**Parameters**:
- `ratings_df` (pd.DataFrame): Training data
  - Required columns: `userId`, `movieId`, `rating`
  - Optional: `timestamp`
- `movies_df` (pd.DataFrame, optional): Movie metadata
  - Required for Content-Based algorithm
  - Columns: `movieId`, `title`, `genres`

**Behavior**:
- **SVD**: Trains matrix factorization model (15-30 min on 32M)
- **User-KNN**: Builds user similarity matrix (2-4 hours)
- **Item-KNN**: Builds item similarity matrix (2-4 hours)
- **Content-Based**: TF-IDF vectorization + similarity matrix (10-20 min)
- **Hybrid**: Trains all 4 sub-algorithms (sum of above)

**Note**: In production, use pre-trained models via `git lfs pull` instead of calling `fit()`.

**Example**:

```python
# Training from scratch (not recommended for production)
ratings_df = load_ratings()
movies_df = load_movies()

recommender = ContentBasedRecommender()
recommender.fit(ratings_df, movies_df)
```

---

### Method: `predict()`

**Purpose**: Predict rating for a specific user-movie pair.

```python
def predict(user_id: int, movie_id: int) -> float
```

**Parameters**:
- `user_id` (int): User identifier (1 to 330,000)
- `movie_id` (int): Movie identifier

**Returns**:
- `float`: Predicted rating (1.0 to 5.0 scale)

**Example**:

```python
manager = AlgorithmManager()
recommender = manager.get_algorithm("SVD")

# Predict single rating
predicted_rating = recommender.predict(user_id=123, movie_id=456)
print(f"Predicted rating: {predicted_rating:.2f}⭐")
# Output: Predicted rating: 4.35⭐
```

**Performance**:
- **SVD**: <0.001s per prediction
- **User-KNN**: ~0.01s per prediction
- **Item-KNN**: ~0.01s per prediction
- **Content-Based**: ~0.005s per prediction
- **Hybrid**: ~0.04s (4 algorithm predictions)

---

### Method: `recommend()`

**Purpose**: Generate Top-N movie recommendations for a user (F-02).

```python
def recommend(
    user_id: int,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    n: int = 10
) -> List[Dict]
```

**Parameters**:
- `user_id` (int): User identifier
- `ratings_df` (pd.DataFrame): Ratings data (to filter out already-rated movies)
- `movies_df` (pd.DataFrame): Movie metadata for enrichment
- `n` (int, default=10): Number of recommendations

**Returns**:
```python
[
    {
        "movieId": 123,
        "title": "Inception (2010)",
        "genres": "Action|Sci-Fi|Thriller",
        "predicted_rating": 4.52
    },
    {
        "movieId": 456,
        "title": "The Matrix (1999)",
        "genres": "Action|Sci-Fi",
        "predicted_rating": 4.48
    },
    # ... 8 more recommendations
]
```

**Example**:

```python
manager = AlgorithmManager()
recommender = manager.get_algorithm("Hybrid")

# Generate Top-10 recommendations
recommendations = recommender.recommend(
    user_id=123,
    ratings_df=ratings_df,
    movies_df=movies_df,
    n=10
)

# Display results
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec['title']} - {rec['predicted_rating']:.2f}⭐")
```

**Performance**:
- **SVD**: ~2s for Top-10
- **User-KNN**: ~1s (with smart sampling)
- **Item-KNN**: ~1s
- **Content-Based**: ~0.5s
- **Hybrid**: ~3s (combines all 4)

---

## 🛠️ Utility Functions

### XAI: `generate_explanation()`

**Purpose**: Create human-readable explanations for recommendations (F-05).

```python
def generate_explanation(
    user_id: int,
    movie_id: int,
    predicted_rating: float,
    algorithm: str,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame
) -> str
```

**Parameters**:
- `user_id` (int): User identifier
- `movie_id` (int): Recommended movie ID
- `predicted_rating` (float): Predicted rating
- `algorithm` (str): Algorithm name for algorithm-specific explanations
- `ratings_df` (pd.DataFrame): User's rating history
- `movies_df` (pd.DataFrame): Movie metadata

**Returns**:
- `str`: Human-readable explanation

**Example**:

```python
from src.utils import generate_explanation

explanation = generate_explanation(
    user_id=123,
    movie_id=456,
    predicted_rating=4.5,
    algorithm="Content-Based",
    ratings_df=ratings_df,
    movies_df=movies_df
)

print(explanation)
# Output: "Because you rated 'Inception' (Sci-Fi) highly (4.5★),
#          you might enjoy this similar Sci-Fi thriller."
```

**Explanation Strategies**:
1. **Content-Based Similarity**: Match with user's highly-rated movies in same genre
2. **Collaborative Filtering**: "Users like you loved this movie"
3. **Genre Preference**: "Matches your love for [genres]"
4. **Fallback**: "This critically acclaimed film..."

---

### Taste Profile: `get_user_top_genres()`

**Purpose**: Identify user's favorite genres for F-06 feature.

```python
def get_user_top_genres(
    user_id: int,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    top_n: int = 3
) -> List[Tuple[str, float]]
```

**Returns**:
```python
[
    ("Drama", 0.35),     # 35% of user's preferences
    ("Action", 0.28),    # 28%
    ("Comedy", 0.18)     # 18%
]
```

**Example**:

```python
from src.utils import get_user_top_genres

top_genres = get_user_top_genres(user_id=123, ratings_df, movies_df, top_n=5)

print("Your Top Genres:")
for genre, score in top_genres:
    print(f"  {genre}: {score:.1%}")
```

---

### Diversity: `calculate_diversity_score()`

**Purpose**: Measure recommendation diversity for F-08 Serendipity feature.

```python
def calculate_diversity_score(
    recommendations: List[int],
    movies_df: pd.DataFrame
) -> float
```

**Parameters**:
- `recommendations` (List[int]): List of recommended movie IDs
- `movies_df` (pd.DataFrame): Movie metadata with genres

**Returns**:
- `float`: Diversity score (0.0 to 1.0)
  - 0.0 = All movies same genre
  - 1.0 = Maximum genre diversity

**Example**:

```python
from src.utils import calculate_diversity_score

movie_ids = [123, 456, 789]  # Recommended movies
diversity = calculate_diversity_score(movie_ids, movies_df)

if diversity > 0.7:
    print("🎉 Highly diverse recommendations!")
elif diversity > 0.4:
    print("✅ Good genre variety")
else:
    print("⚠️ Recommendations are similar")
```

---

## 🔄 Common Integration Patterns

### Pattern 1: Simple Recommendation Flow

```python
from src.algorithms.algorithm_manager import AlgorithmManager
from src.data_processing import load_ratings, load_movies

# Setup
manager = AlgorithmManager()
ratings_df = load_ratings()
movies_df = load_movies()

# Get recommendations
recommender = manager.get_algorithm("User-KNN")
recommendations = recommender.recommend(
    user_id=123,
    ratings_df=ratings_df,
    movies_df=movies_df,
    n=10
)

# Display
for rec in recommendations:
    print(f"{rec['title']}: {rec['predicted_rating']:.2f}⭐")
```

---

### Pattern 2: Algorithm Comparison

```python
algorithms = ["SVD", "User-KNN", "Item-KNN", "Content-Based", "Hybrid"]
user_id = 123

for algo_name in algorithms:
    recommender = manager.get_algorithm(algo_name)
    recs = recommender.recommend(user_id, ratings_df, movies_df, n=5)
    
    print(f"\n{algo_name} Top-5:")
    for i, rec in enumerate(recs, 1):
        print(f"  {i}. {rec['title']} ({rec['predicted_rating']:.2f}⭐)")
```

---

### Pattern 3: Streamlit Integration

```python
import streamlit as st
from src.algorithms.algorithm_manager import AlgorithmManager

@st.cache_resource
def get_manager():
    return AlgorithmManager()

@st.cache_data
def load_data():
    return load_ratings(), load_movies()

# UI
manager = get_manager()
ratings_df, movies_df = load_data()

algorithm = st.selectbox("Algorithm:", ["SVD", "User-KNN", "Hybrid"])
user_id = st.number_input("User ID:", min_value=1, value=123)

if st.button("Get Recommendations"):
    recommender = manager.get_algorithm(algorithm, ratings_df, movies_df)
    recs = recommender.recommend(user_id, ratings_df, movies_df, n=10)
    
    for rec in recs:
        st.write(f"**{rec['title']}** - {rec['predicted_rating']:.2f}⭐")
```

---

### Pattern 4: Error Handling

```python
from src.algorithms.algorithm_manager import AlgorithmManager

manager = AlgorithmManager()

try:
    recommender = manager.get_algorithm("User-KNN")
except FileNotFoundError as e:
    print("❌ Model not found. Please run: git lfs pull")
    print(f"Error: {e}")
except RuntimeError as e:
    print("❌ Git LFS pointer file detected")
    print("Run: git lfs pull")
except ValueError as e:
    print(f"❌ Invalid algorithm name: {e}")
```

---

## 📊 Performance Benchmarks

### API Call Performance

| Method | First Call | Cached Call | Notes |
|--------|-----------|-------------|-------|
| `get_algorithm("SVD")` | ~0.05s | <0.01s | Smallest model |
| `get_algorithm("User-KNN")` | ~1.5s | <0.01s | 266MB load |
| `get_algorithm("Content-Based")` | ~0.8s | <0.01s | 300MB load |
| `get_algorithm("Hybrid")` | ~3s | <0.01s | Loads all 4 |
| `recommend(n=10)` | 0.5-3s | 0.5-3s | Depends on algorithm |
| `predict()` | <0.001s | <0.001s | Single prediction |
| `get_all_algorithm_metrics()` | ~7s | ~7s | Benchmarks all 5 |

### Memory Usage

| Component | RAM Usage | Notes |
|-----------|-----------|-------|
| AlgorithmManager singleton | <1MB | Lightweight |
| SVD model | ~50MB | Small latent factors |
| User-KNN model | ~300MB | Similarity matrix in RAM |
| Item-KNN model | ~300MB | Similarity matrix in RAM |
| Content-Based model | ~400MB | TF-IDF + sparse matrices |
| Hybrid (all loaded) | ~1.1GB | Sum of all 4 |

**Recommendation**: Load algorithms on-demand, not all at startup.

---

## 🚨 Error Handling Guide

### Common Errors

**FileNotFoundError**:
```python
# Error: models/user_knn_model.pkl not found
# Solution: Run git lfs pull
```

**Git LFS Pointer Detection**:
```python
# Error: RuntimeError - File is a Git LFS pointer
# Solution: Install Git LFS, then run git lfs pull
```

**Invalid User ID**:
```python
# Error: User ID 999999999 not in dataset
# Solution: Validate user_id in range (1, 330000)
```

**Missing movies_df**:
```python
# Error: Content-Based requires movies_df parameter
# Solution: Pass movies_df to get_algorithm()
recommender = manager.get_algorithm(
    "Content-Based",
    ratings_df=ratings_df,
    movies_df=movies_df  # Required!
)
```

---

## 📚 Additional Resources

- **Module Documentation**: See `MODULE_DOCUMENTATION.md`
- **Architecture Guide**: See `ARCHITECTURE.md`
- **Training Guide**: See `TRAINING_WORKFLOW.md` (if exists)
- **Troubleshooting**: See `TROUBLESHOOTING.md`

---

*Document Version: 1.0*
*Last Updated: November 13, 2025*
*Part of CineMatch V2.1.0 Documentation Suite*
