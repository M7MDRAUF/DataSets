# ADR-001: KNN Template Method Pattern

**Status**: ✅ Accepted and Implemented  
**Date**: November 12, 2025  
**Authors**: CineMatch Development Team  
**Phase**: 1A - KNN Refactoring

---

## Context and Problem Statement

The CineMatch recommendation system had two KNN-based algorithms (`UserKNNRecommender` and `ItemKNNRecommender`) with significant code duplication (~215 lines). Both classes shared:
- Common data structures (rating matrices, similarity matrices)
- Similar initialization logic
- Identical model persistence methods (save/load)
- Common recommendation workflow
- Shared utility methods

This duplication violated the DRY (Don't Repeat Yourself) principle and made maintenance difficult - any bug fix or enhancement needed to be applied in two places.

## Decision Drivers

* **Code Maintainability**: Eliminate duplication to reduce maintenance burden
* **Bug Consistency**: Ensure bug fixes apply to both algorithms automatically
* **Extensibility**: Make it easier to add new KNN variants in the future
* **Test Coverage**: Improve testability by centralizing common logic
* **Performance**: No performance regression acceptable

## Considered Options

### Option 1: Keep Current Duplication ❌
- **Pros**: No refactoring effort, no risk
- **Cons**: Continued maintenance burden, high bug risk, poor extensibility

### Option 2: Extract Utility Module ⚠️
- **Pros**: Simple to implement
- **Cons**: Still requires manual coordination, doesn't enforce structure

### Option 3: Template Method Pattern ✅ (Selected)
- **Pros**: 
  - Enforces consistent structure via abstract base class
  - Eliminates duplication automatically
  - Clear extension points for new variants
  - Pythonic approach using ABC (Abstract Base Classes)
- **Cons**: 
  - Requires understanding of inheritance
  - Initial refactoring effort

## Decision Outcome

**Chosen option**: Template Method Pattern via abstract base class.

We created `KNNBaseRecommender` as an abstract base class that:
1. Defines the template algorithm structure in concrete methods
2. Declares abstract methods for algorithm-specific behavior
3. Provides common utility methods
4. Handles model persistence uniformly

### Implementation Details

```python
class KNNBaseRecommender(BaseRecommender, ABC):
    """Abstract base class for KNN-based recommenders using Template Method pattern"""
    
    # Template methods (concrete, shared logic)
    def fit(self, ratings_df, movies_df):
        """Template method - same for all KNN variants"""
        
    def predict(self, user_id, movie_id):
        """Template method - same for all KNN variants"""
        
    # Hook methods (abstract, algorithm-specific)
    @abstractmethod
    def _create_rating_matrix(self, ratings_df):
        """Subclass-specific: User-based vs Item-based"""
        
    @abstractmethod
    def _create_similarity_matrix(self, rating_matrix):
        """Subclass-specific: User similarity vs Item similarity"""
```

**Subclass Implementation**:
```python
class UserKNNRecommender(KNNBaseRecommender):
    def _create_rating_matrix(self, ratings_df):
        # User-based: rows=users, cols=movies
        
    def _create_similarity_matrix(self, rating_matrix):
        # User-user similarity
```

## Consequences

### Positive ✅

1. **Code Reduction**: 215 lines eliminated (30% reduction)
   - `UserKNNRecommender`: Simplified by reusing base class
   - `ItemKNNRecommender`: Simplified by reusing base class
   - `KNNBaseRecommender`: 278 lines of reusable code

2. **Maintainability**: Single source of truth for KNN logic
   - Bug fixes automatically apply to both algorithms
   - Enhancements benefit both implementations

3. **Extensibility**: Easy to add new KNN variants
   - Only need to implement 2 abstract methods
   - All other functionality inherited

4. **Type Safety**: Abstract base class enforces interface
   - Python ABC ensures subclasses implement required methods
   - Compile-time checking (via type checkers like mypy)

5. **Testing**: Common logic tested once
   - Base class tests cover shared behavior
   - Subclass tests focus on specific behavior

### Negative ⚠️

1. **Learning Curve**: Developers must understand inheritance
   - Mitigated by clear documentation
   - Standard OOP pattern, widely understood

2. **Indirection**: Logic split across base and derived classes
   - Mitigated by clear method names
   - IDE navigation makes this easy

3. **Rigidity**: Changes to base class affect all subclasses
   - Mitigated by careful design of extension points
   - Proper versioning and testing

## Validation

### Test Results
- ✅ All 7 regression tests passing (100% success rate)
- ✅ No performance regression (load time < 15s target)
- ✅ Both KNN variants working correctly
- ✅ Model persistence working (save/load)

### Metrics
- **Lines Eliminated**: 215 lines (-30%)
- **New Reusable Code**: 278 lines (base class)
- **Cyclomatic Complexity**: Reduced in both subclasses
- **Maintainability Index**: Improved

## Links

- **Implementation Commit**: Phase 1A commits (3 total)
  1. Add KNN base class with Template Method pattern
  2. Refactor User KNN to inherit from base
  3. Refactor Item KNN to inherit from base

- **Related Documents**:
  - `src/algorithms/knn_base.py` - Base class implementation
  - `src/algorithms/user_knn_recommender.py` - User-based implementation
  - `src/algorithms/item_knn_recommender.py` - Item-based implementation
  - `test_bug_fixes_regression.py` - Regression test suite

- **Design Pattern Reference**:
  - [Template Method Pattern - Wikipedia](https://en.wikipedia.org/wiki/Template_method_pattern)
  - [Python ABC Documentation](https://docs.python.org/3/library/abc.html)

## Notes

This refactoring establishes a foundation for future KNN variants (e.g., time-aware KNN, weighted KNN) and demonstrates SOLID principles in action, particularly the Open/Closed Principle (open for extension, closed for modification).

The pattern proved highly successful and could be applied to other algorithm families in the codebase (e.g., matrix factorization algorithms).
