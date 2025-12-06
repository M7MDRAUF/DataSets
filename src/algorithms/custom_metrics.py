"""
CineMatch V2.1.6 - Custom Metrics Interface

Extensible framework for defining custom evaluation metrics
for recommendation system performance assessment.

Author: CineMatch Development Team
"""

from abc import ABC, abstractmethod
from typing import (
    Dict, Any, Optional, List, Callable, Type, Union, Tuple
)
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import logging
from functools import wraps
import time


logger = logging.getLogger(__name__)


# =============================================================================
# Metric Types and Enums
# =============================================================================

class MetricCategory(Enum):
    """Categories of recommendation metrics"""
    ACCURACY = "accuracy"           # Prediction accuracy metrics
    RANKING = "ranking"             # Ranking quality metrics
    COVERAGE = "coverage"           # Item/user coverage metrics
    DIVERSITY = "diversity"         # Recommendation diversity metrics
    NOVELTY = "novelty"             # Item novelty metrics
    SERENDIPITY = "serendipity"     # Unexpected good recommendations
    FAIRNESS = "fairness"           # Fairness/bias metrics
    BUSINESS = "business"           # Business-oriented metrics
    CUSTOM = "custom"               # User-defined metrics


class MetricScope(Enum):
    """Scope at which metric is computed"""
    ITEM = "item"           # Per-item metric
    USER = "user"           # Per-user metric
    GLOBAL = "global"       # System-wide metric
    SEGMENT = "segment"     # User/item segment metric


class AggregationType(Enum):
    """How to aggregate metric values"""
    MEAN = "mean"
    MEDIAN = "median"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    WEIGHTED_MEAN = "weighted_mean"
    HARMONIC_MEAN = "harmonic_mean"


# =============================================================================
# Metric Configuration
# =============================================================================

@dataclass
class MetricConfig:
    """Configuration for a custom metric"""
    name: str
    category: MetricCategory
    scope: MetricScope
    description: str = ""
    aggregation: AggregationType = AggregationType.MEAN
    higher_is_better: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    requires_ground_truth: bool = True
    requires_predictions: bool = True
    requires_item_features: bool = False
    requires_user_features: bool = False


# =============================================================================
# Base Metric Class
# =============================================================================

class BaseMetric(ABC):
    """
    Abstract base class for recommendation metrics.
    
    All custom metrics should inherit from this class and implement
    the compute() method.
    """
    
    def __init__(self, config: MetricConfig):
        self.config = config
        self.name = config.name
        self._cache: Dict[str, Any] = {}
        
    @abstractmethod
    def compute(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        **kwargs
    ) -> float:
        """
        Compute the metric value.
        
        Args:
            predictions: Predicted scores or rankings
            ground_truth: Actual ratings or relevance labels
            **kwargs: Additional data (item features, user features, etc.)
            
        Returns:
            Computed metric value
        """
        pass
    
    def validate_inputs(
        self,
        predictions: Optional[np.ndarray],
        ground_truth: Optional[np.ndarray]
    ) -> bool:
        """Validate input data before computation"""
        if self.config.requires_predictions and predictions is None:
            raise ValueError(f"Metric {self.name} requires predictions")
        if self.config.requires_ground_truth and ground_truth is None:
            raise ValueError(f"Metric {self.name} requires ground truth")
        if predictions is not None and ground_truth is not None:
            if len(predictions) != len(ground_truth):
                raise ValueError("Predictions and ground truth must have same length")
        return True
    
    def normalize(self, value: float) -> float:
        """Normalize metric value to configured range"""
        if self.config.min_value is not None and self.config.max_value is not None:
            return (value - self.config.min_value) / (
                self.config.max_value - self.config.min_value
            )
        return value
    
    def aggregate(self, values: List[float], weights: Optional[List[float]] = None) -> float:
        """Aggregate multiple metric values"""
        if not values:
            return 0.0
            
        arr = np.array(values)
        
        if self.config.aggregation == AggregationType.MEAN:
            return float(np.mean(arr))
        elif self.config.aggregation == AggregationType.MEDIAN:
            return float(np.median(arr))
        elif self.config.aggregation == AggregationType.SUM:
            return float(np.sum(arr))
        elif self.config.aggregation == AggregationType.MIN:
            return float(np.min(arr))
        elif self.config.aggregation == AggregationType.MAX:
            return float(np.max(arr))
        elif self.config.aggregation == AggregationType.WEIGHTED_MEAN:
            if weights is None:
                weights = [1.0] * len(values)
            return float(np.average(arr, weights=weights))
        elif self.config.aggregation == AggregationType.HARMONIC_MEAN:
            return float(len(arr) / np.sum(1.0 / (arr + 1e-10)))
        else:
            return float(np.mean(arr))
    
    def __call__(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        **kwargs
    ) -> float:
        """Allow metric to be called like a function"""
        self.validate_inputs(predictions, ground_truth)
        return self.compute(predictions, ground_truth, **kwargs)


# =============================================================================
# Built-in Accuracy Metrics
# =============================================================================

class RMSEMetric(BaseMetric):
    """Root Mean Square Error"""
    
    def __init__(self):
        super().__init__(MetricConfig(
            name="RMSE",
            category=MetricCategory.ACCURACY,
            scope=MetricScope.GLOBAL,
            description="Root Mean Square Error",
            higher_is_better=False,
            min_value=0.0
        ))
    
    def compute(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        **kwargs
    ) -> float:
        return float(np.sqrt(np.mean((predictions - ground_truth) ** 2)))


class MAEMetric(BaseMetric):
    """Mean Absolute Error"""
    
    def __init__(self):
        super().__init__(MetricConfig(
            name="MAE",
            category=MetricCategory.ACCURACY,
            scope=MetricScope.GLOBAL,
            description="Mean Absolute Error",
            higher_is_better=False,
            min_value=0.0
        ))
    
    def compute(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        **kwargs
    ) -> float:
        return float(np.mean(np.abs(predictions - ground_truth)))


# =============================================================================
# Built-in Ranking Metrics
# =============================================================================

class PrecisionAtKMetric(BaseMetric):
    """Precision at K"""
    
    def __init__(self, k: int = 10):
        super().__init__(MetricConfig(
            name=f"Precision@{k}",
            category=MetricCategory.RANKING,
            scope=MetricScope.USER,
            description=f"Precision at top {k} recommendations",
            higher_is_better=True,
            min_value=0.0,
            max_value=1.0,
            parameters={'k': k}
        ))
        self.k = k
    
    def compute(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        **kwargs
    ) -> float:
        # predictions: ranked list of item indices
        # ground_truth: set of relevant item indices
        relevant = set(ground_truth.astype(int))
        recommended = predictions[:self.k].astype(int)
        hits = len(set(recommended) & relevant)
        return hits / self.k


class RecallAtKMetric(BaseMetric):
    """Recall at K"""
    
    def __init__(self, k: int = 10):
        super().__init__(MetricConfig(
            name=f"Recall@{k}",
            category=MetricCategory.RANKING,
            scope=MetricScope.USER,
            description=f"Recall at top {k} recommendations",
            higher_is_better=True,
            min_value=0.0,
            max_value=1.0,
            parameters={'k': k}
        ))
        self.k = k
    
    def compute(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        **kwargs
    ) -> float:
        relevant = set(ground_truth.astype(int))
        if len(relevant) == 0:
            return 0.0
        recommended = predictions[:self.k].astype(int)
        hits = len(set(recommended) & relevant)
        return hits / len(relevant)


class NDCGAtKMetric(BaseMetric):
    """Normalized Discounted Cumulative Gain at K"""
    
    def __init__(self, k: int = 10):
        super().__init__(MetricConfig(
            name=f"NDCG@{k}",
            category=MetricCategory.RANKING,
            scope=MetricScope.USER,
            description=f"NDCG at top {k} recommendations",
            higher_is_better=True,
            min_value=0.0,
            max_value=1.0,
            parameters={'k': k}
        ))
        self.k = k
    
    def compute(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        **kwargs
    ) -> float:
        # predictions: relevance scores for recommended items
        # ground_truth: ideal relevance scores
        def dcg(scores: np.ndarray, k: int) -> float:
            scores = scores[:k]
            gains = 2 ** scores - 1
            discounts = np.log2(np.arange(len(scores)) + 2)
            return float(np.sum(gains / discounts))
        
        actual_dcg = dcg(predictions, self.k)
        ideal_dcg = dcg(np.sort(ground_truth)[::-1], self.k)
        
        if ideal_dcg == 0:
            return 0.0
        return actual_dcg / ideal_dcg


class MAPMetric(BaseMetric):
    """Mean Average Precision"""
    
    def __init__(self, k: Optional[int] = None):
        name = f"MAP@{k}" if k else "MAP"
        super().__init__(MetricConfig(
            name=name,
            category=MetricCategory.RANKING,
            scope=MetricScope.USER,
            description="Mean Average Precision",
            higher_is_better=True,
            min_value=0.0,
            max_value=1.0,
            parameters={'k': k}
        ))
        self.k = k
    
    def compute(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        **kwargs
    ) -> float:
        relevant = set(ground_truth.astype(int))
        if len(relevant) == 0:
            return 0.0
            
        recommended = predictions.astype(int)
        if self.k:
            recommended = recommended[:self.k]
        
        precision_sum = 0.0
        hits = 0
        
        for i, item in enumerate(recommended):
            if item in relevant:
                hits += 1
                precision_sum += hits / (i + 1)
        
        return precision_sum / len(relevant) if relevant else 0.0


class MRRMetric(BaseMetric):
    """Mean Reciprocal Rank"""
    
    def __init__(self):
        super().__init__(MetricConfig(
            name="MRR",
            category=MetricCategory.RANKING,
            scope=MetricScope.USER,
            description="Mean Reciprocal Rank",
            higher_is_better=True,
            min_value=0.0,
            max_value=1.0
        ))
    
    def compute(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        **kwargs
    ) -> float:
        relevant = set(ground_truth.astype(int))
        recommended = predictions.astype(int)
        
        for i, item in enumerate(recommended):
            if item in relevant:
                return 1.0 / (i + 1)
        
        return 0.0


# =============================================================================
# Coverage Metrics
# =============================================================================

class CatalogCoverageMetric(BaseMetric):
    """Item catalog coverage"""
    
    def __init__(self, total_items: int):
        super().__init__(MetricConfig(
            name="Catalog Coverage",
            category=MetricCategory.COVERAGE,
            scope=MetricScope.GLOBAL,
            description="Fraction of items recommended at least once",
            higher_is_better=True,
            min_value=0.0,
            max_value=1.0,
            requires_ground_truth=False
        ))
        self.total_items = total_items
    
    def compute(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray = None,
        **kwargs
    ) -> float:
        # predictions: all recommended items across users
        unique_items = len(set(predictions.flatten().astype(int)))
        return unique_items / self.total_items


class UserCoverageMetric(BaseMetric):
    """User coverage - fraction of users who received recommendations"""
    
    def __init__(self, total_users: int):
        super().__init__(MetricConfig(
            name="User Coverage",
            category=MetricCategory.COVERAGE,
            scope=MetricScope.GLOBAL,
            description="Fraction of users receiving recommendations",
            higher_is_better=True,
            min_value=0.0,
            max_value=1.0,
            requires_ground_truth=False
        ))
        self.total_users = total_users
    
    def compute(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray = None,
        users_with_recs: Optional[int] = None,
        **kwargs
    ) -> float:
        if users_with_recs is not None:
            return users_with_recs / self.total_users
        return len(predictions) / self.total_users


# =============================================================================
# Diversity Metrics
# =============================================================================

class IntraListDiversityMetric(BaseMetric):
    """Intra-list diversity based on item similarity"""
    
    def __init__(self):
        super().__init__(MetricConfig(
            name="Intra-List Diversity",
            category=MetricCategory.DIVERSITY,
            scope=MetricScope.USER,
            description="Average dissimilarity between recommended items",
            higher_is_better=True,
            min_value=0.0,
            max_value=1.0,
            requires_item_features=True
        ))
    
    def compute(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray = None,
        item_features: np.ndarray = None,
        **kwargs
    ) -> float:
        if item_features is None:
            return 0.0
            
        n = len(predictions)
        if n < 2:
            return 1.0
            
        # Compute pairwise cosine distances
        features = item_features[predictions.astype(int)]
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = features / norms
        
        similarities = normalized @ normalized.T
        # Average dissimilarity (1 - similarity) for upper triangle
        total_dissim = (n * n - np.sum(similarities)) / 2
        pairs = n * (n - 1) / 2
        
        return total_dissim / pairs if pairs > 0 else 1.0


class GiniDiversityMetric(BaseMetric):
    """Gini index for recommendation diversity"""
    
    def __init__(self):
        super().__init__(MetricConfig(
            name="Gini Diversity",
            category=MetricCategory.DIVERSITY,
            scope=MetricScope.GLOBAL,
            description="1 - Gini coefficient of item recommendation frequency",
            higher_is_better=True,
            min_value=0.0,
            max_value=1.0,
            requires_ground_truth=False
        ))
    
    def compute(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray = None,
        **kwargs
    ) -> float:
        # Count item frequencies
        from collections import Counter
        counts = list(Counter(predictions.flatten().astype(int)).values())
        n = len(counts)
        
        if n == 0:
            return 0.0
            
        sorted_counts = np.sort(counts)
        cumulative = np.cumsum(sorted_counts)
        gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
        
        return 1 - gini  # Higher is more diverse


# =============================================================================
# Novelty Metrics
# =============================================================================

class PopularityNoveltyMetric(BaseMetric):
    """Novelty based on inverse item popularity"""
    
    def __init__(self, item_popularity: Dict[int, float]):
        super().__init__(MetricConfig(
            name="Popularity Novelty",
            category=MetricCategory.NOVELTY,
            scope=MetricScope.USER,
            description="Average self-information of recommended items",
            higher_is_better=True,
            min_value=0.0,
            requires_ground_truth=False
        ))
        self.item_popularity = item_popularity
    
    def compute(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray = None,
        **kwargs
    ) -> float:
        novelties = []
        for item in predictions.astype(int):
            pop = self.item_popularity.get(item, 0.001)  # Avoid log(0)
            novelty = -np.log2(pop)
            novelties.append(novelty)
        
        return float(np.mean(novelties)) if novelties else 0.0


# =============================================================================
# Metric Registry
# =============================================================================

class MetricRegistry:
    """
    Registry for managing custom and built-in metrics.
    """
    
    _instance: Optional['MetricRegistry'] = None
    
    def __new__(cls) -> 'MetricRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._metrics: Dict[str, Type[BaseMetric]] = {}
            cls._instance._instances: Dict[str, BaseMetric] = {}
            cls._instance._register_builtins()
        return cls._instance
    
    def _register_builtins(self):
        """Register built-in metrics"""
        self.register("rmse", RMSEMetric)
        self.register("mae", MAEMetric)
        self.register("precision@10", PrecisionAtKMetric, k=10)
        self.register("recall@10", RecallAtKMetric, k=10)
        self.register("ndcg@10", NDCGAtKMetric, k=10)
        self.register("map", MAPMetric)
        self.register("mrr", MRRMetric)
        self.register("intra_list_diversity", IntraListDiversityMetric)
        self.register("gini_diversity", GiniDiversityMetric)
    
    def register(
        self,
        name: str,
        metric_class: Type[BaseMetric],
        **default_params
    ):
        """Register a metric class"""
        self._metrics[name.lower()] = metric_class
        # Create default instance
        try:
            self._instances[name.lower()] = metric_class(**default_params)
            logger.info(f"Registered metric: {name}")
        except Exception as e:
            logger.warning(f"Could not create default instance for {name}: {e}")
    
    def get(self, name: str, **params) -> Optional[BaseMetric]:
        """Get a metric instance by name"""
        name_lower = name.lower()
        
        if params:
            # Create new instance with custom params
            if name_lower in self._metrics:
                return self._metrics[name_lower](**params)
        
        return self._instances.get(name_lower)
    
    def list_metrics(self, category: Optional[MetricCategory] = None) -> List[str]:
        """List all registered metrics"""
        if category is None:
            return list(self._metrics.keys())
        
        return [
            name for name, instance in self._instances.items()
            if instance.config.category == category
        ]
    
    def unregister(self, name: str):
        """Remove a metric from registry"""
        name_lower = name.lower()
        self._metrics.pop(name_lower, None)
        self._instances.pop(name_lower, None)


# =============================================================================
# Custom Metric Builder
# =============================================================================

class MetricBuilder:
    """
    Builder pattern for creating custom metrics.
    
    Example:
        metric = (MetricBuilder("my_metric")
            .category(MetricCategory.ACCURACY)
            .scope(MetricScope.USER)
            .description("My custom metric")
            .higher_is_better(False)
            .compute_function(lambda p, g: np.mean(p - g))
            .build())
    """
    
    def __init__(self, name: str):
        self._name = name
        self._category = MetricCategory.CUSTOM
        self._scope = MetricScope.GLOBAL
        self._description = ""
        self._higher_is_better = True
        self._min_value: Optional[float] = None
        self._max_value: Optional[float] = None
        self._compute_fn: Optional[Callable] = None
        self._aggregation = AggregationType.MEAN
        self._requires_ground_truth = True
        self._requires_predictions = True
    
    def category(self, cat: MetricCategory) -> 'MetricBuilder':
        self._category = cat
        return self
    
    def scope(self, scope: MetricScope) -> 'MetricBuilder':
        self._scope = scope
        return self
    
    def description(self, desc: str) -> 'MetricBuilder':
        self._description = desc
        return self
    
    def higher_is_better(self, value: bool) -> 'MetricBuilder':
        self._higher_is_better = value
        return self
    
    def value_range(self, min_val: float, max_val: float) -> 'MetricBuilder':
        self._min_value = min_val
        self._max_value = max_val
        return self
    
    def aggregation(self, agg: AggregationType) -> 'MetricBuilder':
        self._aggregation = agg
        return self
    
    def compute_function(self, fn: Callable[[np.ndarray, np.ndarray], float]) -> 'MetricBuilder':
        self._compute_fn = fn
        return self
    
    def requires(
        self,
        ground_truth: bool = True,
        predictions: bool = True
    ) -> 'MetricBuilder':
        self._requires_ground_truth = ground_truth
        self._requires_predictions = predictions
        return self
    
    def build(self) -> BaseMetric:
        """Build the custom metric"""
        if self._compute_fn is None:
            raise ValueError("Compute function is required")
        
        config = MetricConfig(
            name=self._name,
            category=self._category,
            scope=self._scope,
            description=self._description,
            aggregation=self._aggregation,
            higher_is_better=self._higher_is_better,
            min_value=self._min_value,
            max_value=self._max_value,
            requires_ground_truth=self._requires_ground_truth,
            requires_predictions=self._requires_predictions
        )
        
        compute_fn = self._compute_fn
        
        class CustomMetric(BaseMetric):
            def compute(self, predictions, ground_truth, **kwargs):
                return compute_fn(predictions, ground_truth, **kwargs)
        
        return CustomMetric(config)


# =============================================================================
# Metric Evaluator
# =============================================================================

class MetricEvaluator:
    """
    Evaluates recommendations using multiple metrics.
    """
    
    def __init__(self):
        self.registry = MetricRegistry()
        self._results_cache: Dict[str, Dict[str, float]] = {}
    
    def evaluate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        metrics: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate predictions using specified metrics.
        
        Args:
            predictions: Predicted scores/rankings
            ground_truth: Actual relevance/ratings
            metrics: List of metric names to compute
            **kwargs: Additional data for metrics
            
        Returns:
            Dictionary of metric name to value
        """
        results = {}
        
        for metric_name in metrics:
            metric = self.registry.get(metric_name)
            if metric is None:
                logger.warning(f"Unknown metric: {metric_name}")
                continue
            
            try:
                start = time.time()
                value = metric(predictions, ground_truth, **kwargs)
                elapsed = time.time() - start
                
                results[metric_name] = value
                logger.debug(f"Metric {metric_name}: {value:.4f} ({elapsed:.3f}s)")
                
            except Exception as e:
                logger.error(f"Error computing {metric_name}: {e}")
                results[metric_name] = float('nan')
        
        return results
    
    def evaluate_batch(
        self,
        user_predictions: Dict[int, np.ndarray],
        user_ground_truth: Dict[int, np.ndarray],
        metrics: List[str],
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate metrics per user and aggregate.
        
        Returns:
            Dictionary with 'per_user' and 'aggregated' results
        """
        per_user_results: Dict[int, Dict[str, float]] = {}
        metric_values: Dict[str, List[float]] = {m: [] for m in metrics}
        
        for user_id in user_predictions:
            preds = user_predictions[user_id]
            truth = user_ground_truth.get(user_id, np.array([]))
            
            user_results = self.evaluate(preds, truth, metrics, **kwargs)
            per_user_results[user_id] = user_results
            
            for metric_name, value in user_results.items():
                if not np.isnan(value):
                    metric_values[metric_name].append(value)
        
        # Aggregate results
        aggregated = {}
        for metric_name, values in metric_values.items():
            metric = self.registry.get(metric_name)
            if metric and values:
                aggregated[metric_name] = metric.aggregate(values)
            else:
                aggregated[metric_name] = 0.0
        
        return {
            'per_user': per_user_results,
            'aggregated': aggregated
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def register_metric(
    name: str,
    metric_class: Type[BaseMetric],
    **default_params
):
    """Register a custom metric"""
    MetricRegistry().register(name, metric_class, **default_params)


def get_metric(name: str, **params) -> Optional[BaseMetric]:
    """Get a metric by name"""
    return MetricRegistry().get(name, **params)


def list_metrics(category: Optional[MetricCategory] = None) -> List[str]:
    """List available metrics"""
    return MetricRegistry().list_metrics(category)


def evaluate_recommendations(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    metrics: List[str] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Evaluate recommendations with default metrics.
    
    Args:
        predictions: Predicted scores/rankings
        ground_truth: Actual relevance/ratings
        metrics: Metric names (default: RMSE, Precision@10, NDCG@10)
        **kwargs: Additional evaluation data
    """
    if metrics is None:
        metrics = ['rmse', 'precision@10', 'ndcg@10']
    
    evaluator = MetricEvaluator()
    return evaluator.evaluate(predictions, ground_truth, metrics, **kwargs)


# =============================================================================
# Example Custom Metric
# =============================================================================

class SerendipityMetric(BaseMetric):
    """
    Serendipity metric - measures unexpected but relevant recommendations.
    
    Serendipity = (1/|L|) * Σ (relevant(i) * (1 - expected(i)))
    """
    
    def __init__(self, expected_model: Optional[Callable] = None):
        super().__init__(MetricConfig(
            name="Serendipity",
            category=MetricCategory.SERENDIPITY,
            scope=MetricScope.USER,
            description="Measures unexpected but relevant recommendations",
            higher_is_better=True,
            min_value=0.0,
            max_value=1.0
        ))
        self.expected_model = expected_model or (lambda x: 0.5)  # Default: 50% expected
    
    def compute(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        **kwargs
    ) -> float:
        relevant = set(ground_truth.astype(int))
        
        serendipity_sum = 0.0
        for item in predictions.astype(int):
            is_relevant = 1.0 if item in relevant else 0.0
            expected = self.expected_model(item)
            unexpectedness = 1.0 - expected
            serendipity_sum += is_relevant * unexpectedness
        
        return serendipity_sum / len(predictions) if len(predictions) > 0 else 0.0


# Register serendipity metric
register_metric("serendipity", SerendipityMetric)
