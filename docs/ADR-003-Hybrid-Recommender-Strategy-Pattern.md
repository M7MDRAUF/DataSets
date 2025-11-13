# ADR-003: Hybrid Recommender Strategy Pattern

**Status**: ✅ Accepted and Implemented  
**Date**: November 12, 2025  
**Authors**: CineMatch Development Team  
**Phase**: 2 - Hybrid Strategy Refactoring

---

## Context and Problem Statement

The `HybridRecommender` contained a 140-line `get_recommendations()` method with complex conditional logic for different user profile types:
- **New users** (< 5 ratings): Cold-start strategy
- **Sparse users** (5-49 ratings): Balanced approach
- **Dense users** (50+ ratings): Full ensemble

This resulted in:
- Complex nested if-else statements
- Difficult to test each strategy in isolation
- Hard to add new user profile strategies
- Poor code readability and maintainability
- Duplication of algorithm weighting logic

## Decision Drivers

* **Code Clarity**: Reduce complexity of conditional logic
* **Testability**: Each strategy should be independently testable
* **Extensibility**: Easy to add new user profile strategies
* **Maintainability**: Changes to one strategy shouldn't affect others
* **Flexibility**: Algorithm weights should be strategy-specific

## Considered Options

### Option 1: Keep Conditional Logic ❌
- **Pros**: No refactoring effort, simple to understand initially
- **Cons**: 140-line method, complex nested logic, hard to extend

### Option 2: Extract Methods ⚠️
- **Pros**: Reduces main method size
- **Cons**: Still couples all strategies in one class, limited reusability

### Option 3: Strategy Pattern ✅ (Selected)
- **Pros**:
  - Each strategy is an independent class
  - Easy to test strategies in isolation
  - Simple to add new strategies
  - Clear separation of concerns
  - Follows Open/Closed Principle
- **Cons**:
  - More classes to manage
  - Need base class/interface

## Decision Outcome

**Chosen option**: Strategy Pattern with abstract base class.

We created a `hybrid_strategies/` module with four strategy classes:

### 1. BaseRecommendationStrategy (110 lines) - Abstract Base Class
**Responsibility**: Define strategy interface and common utilities

**Interface**:
```python
class BaseRecommendationStrategy(ABC):
    @abstractmethod
    def get_algorithm_weights(self, user_profile: Dict) -> Dict[str, float]:
        """Return algorithm weights for this strategy"""
    
    @abstractmethod
    def applies_to(self, user_profile: Dict) -> bool:
        """Check if strategy applies to this user"""
    
    def combine_recommendations(self, algorithm_recs: Dict, weights: Dict) -> List:
        """Weighted combination of recommendations (shared logic)"""
```

### 2. NewUserStrategy (92 lines) - Cold Start
**Responsibility**: Handle users with very few ratings

**Characteristics**:
- Applies to: Users with < 5 ratings
- Algorithm weights:
  - SVD: 40% (can extrapolate from little data)
  - Item KNN: 30% (stable, popular items)
  - Content-Based: 30% (no cold-start problem)
  - User KNN: 0% (insufficient user similarity)

**Rationale**: Focus on algorithms that work with sparse data

```python
class NewUserStrategy(BaseRecommendationStrategy):
    def applies_to(self, user_profile: Dict) -> bool:
        return user_profile['rating_count'] < 5
    
    def get_algorithm_weights(self, user_profile: Dict) -> Dict:
        return {'svd': 0.4, 'item_knn': 0.3, 'content_based': 0.3, 'user_knn': 0.0}
```

### 3. SparseUserStrategy (100 lines) - Balanced Approach
**Responsibility**: Handle users with moderate rating history

**Characteristics**:
- Applies to: Users with 5-49 ratings
- Algorithm weights:
  - User KNN: 50% (user similarity starts working)
  - SVD: 30% (good with moderate data)
  - Content-Based: 20% (diversity)
  - Item KNN: 0% (needs more data for stability)

**Rationale**: Balance collaborative and content-based filtering

```python
class SparseUserStrategy(BaseRecommendationStrategy):
    def applies_to(self, user_profile: Dict) -> bool:
        return 5 <= user_profile['rating_count'] < 50
    
    def get_algorithm_weights(self, user_profile: Dict) -> Dict:
        return {'user_knn': 0.5, 'svd': 0.3, 'content_based': 0.2, 'item_knn': 0.0}
```

### 4. DenseUserStrategy (112 lines) - Full Ensemble
**Responsibility**: Handle users with rich rating history

**Characteristics**:
- Applies to: Users with 50+ ratings
- Algorithm weights: **Dynamically calculated** based on:
  - Rating variance (diversity of taste)
  - Genre diversity
  - Rating recency
  - Activity level

**Rationale**: Use all algorithms with adaptive weighting

```python
class DenseUserStrategy(BaseRecommendationStrategy):
    def applies_to(self, user_profile: Dict) -> bool:
        return user_profile['rating_count'] >= 50
    
    def get_algorithm_weights(self, user_profile: Dict) -> Dict:
        # Dynamic calculation based on user characteristics
        weights = self._calculate_adaptive_weights(user_profile)
        return self._normalize_weights(weights)
```

### Integration

The `HybridRecommender` now uses strategy selection:

```python
class HybridRecommender(BaseRecommender):
    def __init__(self, ...):
        self.strategies = [
            NewUserStrategy(),
            SparseUserStrategy(),
            DenseUserStrategy()
        ]
    
    def get_recommendations(self, user_id, n=10):
        user_profile = self._build_user_profile(user_id)
        
        # Select appropriate strategy
        strategy = self._select_strategy(user_profile)
        
        # Get algorithm weights from strategy
        weights = strategy.get_algorithm_weights(user_profile)
        
        # Get recommendations from each algorithm
        algorithm_recs = self._get_algorithm_recommendations(user_id, n, weights)
        
        # Combine using strategy's method
        return strategy.combine_recommendations(algorithm_recs, weights)
    
    def _select_strategy(self, user_profile):
        for strategy in self.strategies:
            if strategy.applies_to(user_profile):
                return strategy
        return self.strategies[-1]  # Default to dense
```

## Consequences

### Positive ✅

1. **Code Reduction**: 60 lines eliminated (7% overall, 57% in get_recommendations)
   - `get_recommendations()`: 140 lines → 60 lines (-57%)
   - `HybridRecommender`: 843 → 783 lines
   - Complex conditional logic extracted to 440 lines of clean strategies

2. **Improved Readability**: Main method is now clear and linear
   - Before: Nested if-else with complex conditions
   - After: Select strategy → Get weights → Combine results

3. **Independent Testing**: Each strategy testable in isolation
   - Mock user profiles for different scenarios
   - Test weight calculation independently
   - Verify strategy selection logic

4. **Easy Extension**: Adding new strategies is trivial
   ```python
   class PowerUserStrategy(BaseRecommendationStrategy):
       def applies_to(self, user_profile):
           return user_profile['rating_count'] > 200
       # Implement custom weights...
   ```

5. **Clear Separation**: Each strategy encapsulates its logic
   - Algorithm weights specific to user type
   - Combination logic can be strategy-specific
   - No cross-contamination between strategies

6. **Runtime Flexibility**: Strategies can be added/removed dynamically
   - Could load strategies from configuration
   - A/B testing different strategies
   - Feature flags for strategy rollout

### Negative ⚠️

1. **More Classes**: 1 class → 5 classes (base + 4 strategies)
   - Mitigated by clear organization in `hybrid_strategies/` folder
   - Each class is small and focused (92-112 lines)

2. **Strategy Selection**: Need logic to choose strategy
   - Mitigated by simple `applies_to()` method
   - Clear priority order (new → sparse → dense)

3. **Shared State**: Strategies need access to algorithms
   - Mitigated by dependency injection
   - Strategies remain stateless (no algorithm references)

## Validation

### Test Results
- ✅ All 7 regression tests passing (100% success rate)
- ✅ No performance regression (load time < 15s)
- ✅ All three user profiles working correctly
- ✅ Content-Based invoked in all strategies (Bug #14 fixed)

### Metrics
- **Lines Eliminated**: 60 lines (-7% overall)
- **Method Complexity**: get_recommendations() reduced by 57%
- **New Reusable Code**: 440 lines (4 strategy classes)
- **Cyclomatic Complexity**: Significantly reduced
- **Maintainability Index**: Improved

### Strategy Distribution (Test Data)
- **New Users**: ~15% of users → NewUserStrategy
- **Sparse Users**: ~40% of users → SparseUserStrategy
- **Dense Users**: ~45% of users → DenseUserStrategy

## Design Principles Applied

1. **Strategy Pattern**: Encapsulate algorithm selection in separate classes
2. **Open/Closed Principle**: Open for extension (new strategies), closed for modification
3. **Single Responsibility**: Each strategy has one job
4. **Dependency Inversion**: HybridRecommender depends on strategy abstraction
5. **Liskov Substitution**: All strategies interchangeable

## Performance Characteristics

| Strategy | Computation Time | Memory Usage | Recommendation Quality |
|----------|------------------|--------------|------------------------|
| New User | Fast (fewer algos) | Low | Good (cold-start focus) |
| Sparse   | Medium | Medium | Better (balanced) |
| Dense    | Slower (all algos) | Higher | Best (full ensemble) |

## Links

- **Implementation Commits**: Phase 2 commits (2 total)
  1. Add Strategy Pattern for user profile recommendations
  2. Integrate Strategy Pattern into HybridRecommender

- **Related Documents**:
  - `src/algorithms/hybrid_strategies/base_strategy.py` - Abstract base class
  - `src/algorithms/hybrid_strategies/new_user_strategy.py` - Cold-start handling
  - `src/algorithms/hybrid_strategies/sparse_user_strategy.py` - Balanced approach
  - `src/algorithms/hybrid_strategies/dense_user_strategy.py` - Full ensemble
  - `src/algorithms/hybrid_recommender.py` - Main recommender (refactored)

- **Design Pattern Reference**:
  - [Strategy Pattern - Wikipedia](https://en.wikipedia.org/wiki/Strategy_pattern)
  - [Strategy Pattern - Refactoring Guru](https://refactoring.guru/design-patterns/strategy)

## Future Enhancements

This refactoring enables several exciting possibilities:

1. **Machine Learning-Based Strategy Selection**
   - Train a classifier to predict best strategy
   - Use user features beyond just rating count
   - Personalized strategy selection

2. **Hybrid Strategies**
   - Blend multiple strategies (e.g., 70% sparse + 30% dense)
   - Smooth transitions between strategies
   - Confidence-weighted strategy combination

3. **Context-Aware Strategies**
   - Time-of-day strategies (morning vs evening preferences)
   - Device-based strategies (mobile vs desktop)
   - Mood-based strategies (based on recent ratings)

4. **A/B Testing Framework**
   - Easy to test new strategies
   - Compare strategy performance
   - Gradual rollout of new strategies

5. **Explainability**
   - Each strategy can explain its recommendations
   - "We used the new user strategy because..."
   - User-friendly reasoning

## Notes

The Strategy Pattern proved extremely valuable here. The 140-line method with complex conditionals is now a clean 60-line orchestration method. Each strategy is:
- Small (~100 lines each)
- Testable independently
- Easy to understand
- Simple to extend

This pattern should be considered for other complex conditional logic in the codebase (e.g., recommendation explanation generation, algorithm selection logic).

**Key Insight**: Sometimes the best refactoring isn't about reducing lines, but about improving structure. The 440 lines of new code are worth it for the clarity and maintainability gained.
