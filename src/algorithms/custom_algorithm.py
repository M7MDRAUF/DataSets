"""
Custom Algorithm Support Module for CineMatch V2.1.6

Provides interfaces and utilities for users to create and integrate
custom recommendation algorithms into the CineMatch system.

Phase 6 - Task 6.1: Custom Algorithm Support
"""

import hashlib
import json
import logging
import pickle
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class AlgorithmMetadata:
    """Metadata for custom algorithms."""
    name: str
    version: str
    author: str
    description: str
    algorithm_type: str  # collaborative, content-based, hybrid
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "algorithm_type": self.algorithm_type,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "parameters": self.parameters,
            "performance_metrics": self.performance_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlgorithmMetadata':
        """Create from dictionary."""
        return cls(
            name=data["name"],
            version=data["version"],
            author=data["author"],
            description=data["description"],
            algorithm_type=data["algorithm_type"],
            created_at=datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.utcnow().isoformat())),
            tags=data.get("tags", []),
            parameters=data.get("parameters", {}),
            performance_metrics=data.get("performance_metrics", {})
        )


@dataclass
class AlgorithmCapabilities:
    """Capabilities that an algorithm supports."""
    supports_batch: bool = True
    supports_incremental: bool = False
    supports_cold_start: bool = False
    supports_explanations: bool = False
    supports_diversity: bool = False
    min_ratings_required: int = 10
    max_recommendations: int = 1000
    required_features: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "supports_batch": self.supports_batch,
            "supports_incremental": self.supports_incremental,
            "supports_cold_start": self.supports_cold_start,
            "supports_explanations": self.supports_explanations,
            "supports_diversity": self.supports_diversity,
            "min_ratings_required": self.min_ratings_required,
            "max_recommendations": self.max_recommendations,
            "required_features": list(self.required_features)
        }


class CustomAlgorithmInterface(ABC):
    """
    Base interface for custom recommendation algorithms.
    
    Users must implement this interface to create custom algorithms
    that can be integrated into CineMatch.
    
    Example:
        class MyRecommender(CustomAlgorithmInterface):
            def __init__(self):
                super().__init__()
                self._model = None
            
            def get_metadata(self) -> AlgorithmMetadata:
                return AlgorithmMetadata(
                    name="My Recommender",
                    version="1.0.0",
                    author="User",
                    description="My custom algorithm",
                    algorithm_type="collaborative"
                )
            
            def train(self, ratings_df, movies_df, **kwargs):
                # Training logic
                pass
            
            def predict(self, user_id, n_recommendations, **kwargs):
                # Prediction logic
                return recommendations, scores
    """
    
    def __init__(self):
        self._is_trained = False
        self._training_time: Optional[float] = None
        self._last_trained: Optional[datetime] = None
        
    @abstractmethod
    def get_metadata(self) -> AlgorithmMetadata:
        """
        Get algorithm metadata.
        
        Returns:
            AlgorithmMetadata containing algorithm information
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> AlgorithmCapabilities:
        """
        Get algorithm capabilities.
        
        Returns:
            AlgorithmCapabilities describing what the algorithm supports
        """
        pass
    
    @abstractmethod
    def train(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        **kwargs: Any
    ) -> None:
        """
        Train the algorithm on the provided data.
        
        Args:
            ratings_df: DataFrame with columns [user_id, movie_id, rating]
            movies_df: DataFrame with movie information
            **kwargs: Additional training parameters
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        user_id: int,
        n_recommendations: int = 10,
        **kwargs: Any
    ) -> tuple[List[int], List[float]]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User to generate recommendations for
            n_recommendations: Number of recommendations
            **kwargs: Additional prediction parameters
            
        Returns:
            Tuple of (movie_ids, scores)
        """
        pass
    
    def predict_batch(
        self,
        user_ids: List[int],
        n_recommendations: int = 10,
        **kwargs: Any
    ) -> Dict[int, tuple[List[int], List[float]]]:
        """
        Generate recommendations for multiple users.
        
        Default implementation calls predict() for each user.
        Override for more efficient batch processing.
        
        Args:
            user_ids: Users to generate recommendations for
            n_recommendations: Number of recommendations per user
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping user_id to (movie_ids, scores)
        """
        results = {}
        for user_id in user_ids:
            try:
                results[user_id] = self.predict(user_id, n_recommendations, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to predict for user {user_id}: {e}")
                results[user_id] = ([], [])
        return results
    
    def get_similar_items(
        self,
        movie_id: int,
        n_similar: int = 10,
        **kwargs: Any
    ) -> tuple[List[int], List[float]]:
        """
        Get similar items to a given movie.
        
        Args:
            movie_id: Movie to find similar items for
            n_similar: Number of similar items
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (movie_ids, similarity_scores)
        """
        raise NotImplementedError("This algorithm doesn't support similar items")
    
    def explain_recommendation(
        self,
        user_id: int,
        movie_id: int,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Explain why a movie was recommended.
        
        Args:
            user_id: User the recommendation was for
            movie_id: Recommended movie
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with explanation data
        """
        raise NotImplementedError("This algorithm doesn't support explanations")
    
    def save(self, path: Path) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Directory to save model to
        """
        raise NotImplementedError("This algorithm doesn't support saving")
    
    def load(self, path: Path) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Directory to load model from
        """
        raise NotImplementedError("This algorithm doesn't support loading")
    
    @property
    def is_trained(self) -> bool:
        """Check if the algorithm is trained."""
        return self._is_trained
    
    @property
    def training_time(self) -> Optional[float]:
        """Get training time in seconds."""
        return self._training_time
    
    @property
    def last_trained(self) -> Optional[datetime]:
        """Get timestamp of last training."""
        return self._last_trained


class AlgorithmValidator:
    """Validates custom algorithms before registration."""
    
    @staticmethod
    def validate_metadata(metadata: AlgorithmMetadata) -> List[str]:
        """
        Validate algorithm metadata.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not metadata.name or len(metadata.name) < 2:
            errors.append("Algorithm name must be at least 2 characters")
        
        if not metadata.version:
            errors.append("Version is required")
        
        if not metadata.author:
            errors.append("Author is required")
        
        valid_types = {"collaborative", "content-based", "hybrid", "custom"}
        if metadata.algorithm_type not in valid_types:
            errors.append(f"Algorithm type must be one of: {valid_types}")
        
        return errors
    
    @staticmethod
    def validate_algorithm(algorithm: CustomAlgorithmInterface) -> List[str]:
        """
        Validate a custom algorithm implementation.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required methods
        required_methods = ['get_metadata', 'get_capabilities', 'train', 'predict']
        for method in required_methods:
            if not hasattr(algorithm, method):
                errors.append(f"Missing required method: {method}")
            elif not callable(getattr(algorithm, method)):
                errors.append(f"{method} is not callable")
        
        # Validate metadata
        try:
            metadata = algorithm.get_metadata()
            errors.extend(AlgorithmValidator.validate_metadata(metadata))
        except Exception as e:
            errors.append(f"Error getting metadata: {e}")
        
        # Validate capabilities
        try:
            capabilities = algorithm.get_capabilities()
            if not isinstance(capabilities, AlgorithmCapabilities):
                errors.append("get_capabilities must return AlgorithmCapabilities")
        except Exception as e:
            errors.append(f"Error getting capabilities: {e}")
        
        return errors
    
    @staticmethod
    def benchmark_algorithm(
        algorithm: CustomAlgorithmInterface,
        test_ratings: pd.DataFrame,
        test_movies: pd.DataFrame,
        test_users: List[int],
        n_recommendations: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark algorithm performance.
        
        Returns:
            Dictionary with benchmark results
        """
        results = {
            "training_time": None,
            "prediction_time": None,
            "predictions_per_second": None,
            "memory_usage": None,
            "errors": []
        }
        
        # Benchmark training
        try:
            start = time.time()
            algorithm.train(test_ratings, test_movies)
            results["training_time"] = time.time() - start
        except Exception as e:
            results["errors"].append(f"Training error: {e}")
            return results
        
        # Benchmark prediction
        try:
            start = time.time()
            for user_id in test_users:
                algorithm.predict(user_id, n_recommendations)
            total_time = time.time() - start
            results["prediction_time"] = total_time
            results["predictions_per_second"] = len(test_users) / total_time if total_time > 0 else 0
        except Exception as e:
            results["errors"].append(f"Prediction error: {e}")
        
        return results


class CustomAlgorithmRegistry:
    """Registry for managing custom algorithms."""
    
    _instance: Optional['CustomAlgorithmRegistry'] = None
    
    def __new__(cls) -> 'CustomAlgorithmRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._algorithms: Dict[str, Type[CustomAlgorithmInterface]] = {}
            cls._instance._instances: Dict[str, CustomAlgorithmInterface] = {}
            cls._instance._metadata: Dict[str, AlgorithmMetadata] = {}
        return cls._instance
    
    def register(
        self,
        algorithm_class: Type[CustomAlgorithmInterface],
        validate: bool = True
    ) -> bool:
        """
        Register a custom algorithm class.
        
        Args:
            algorithm_class: Algorithm class to register
            validate: Whether to validate before registration
            
        Returns:
            True if registration successful
        """
        try:
            # Create instance for validation
            instance = algorithm_class()
            
            if validate:
                errors = AlgorithmValidator.validate_algorithm(instance)
                if errors:
                    logger.error(f"Algorithm validation failed: {errors}")
                    return False
            
            metadata = instance.get_metadata()
            name = metadata.name
            
            self._algorithms[name] = algorithm_class
            self._instances[name] = instance
            self._metadata[name] = metadata
            
            logger.info(f"Registered custom algorithm: {name} v{metadata.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register algorithm: {e}")
            return False
    
    def unregister(self, name: str) -> bool:
        """Unregister an algorithm by name."""
        if name in self._algorithms:
            del self._algorithms[name]
            del self._instances[name]
            del self._metadata[name]
            logger.info(f"Unregistered algorithm: {name}")
            return True
        return False
    
    def get(self, name: str) -> Optional[CustomAlgorithmInterface]:
        """Get algorithm instance by name."""
        return self._instances.get(name)
    
    def get_class(self, name: str) -> Optional[Type[CustomAlgorithmInterface]]:
        """Get algorithm class by name."""
        return self._algorithms.get(name)
    
    def list_algorithms(self) -> List[str]:
        """List registered algorithm names."""
        return list(self._algorithms.keys())
    
    def get_metadata(self, name: str) -> Optional[AlgorithmMetadata]:
        """Get algorithm metadata by name."""
        return self._metadata.get(name)
    
    def get_all_metadata(self) -> Dict[str, AlgorithmMetadata]:
        """Get all algorithm metadata."""
        return dict(self._metadata)
    
    def filter_by_type(self, algorithm_type: str) -> List[str]:
        """Filter algorithms by type."""
        return [
            name for name, meta in self._metadata.items()
            if meta.algorithm_type == algorithm_type
        ]
    
    def filter_by_capability(self, capability: str) -> List[str]:
        """Filter algorithms by capability."""
        results = []
        for name, instance in self._instances.items():
            caps = instance.get_capabilities()
            if getattr(caps, capability, False):
                results.append(name)
        return results
    
    def clear(self) -> None:
        """Clear all registered algorithms."""
        self._algorithms.clear()
        self._instances.clear()
        self._metadata.clear()


def custom_algorithm(
    name: Optional[str] = None,
    version: str = "1.0.0",
    author: str = "Unknown",
    algorithm_type: str = "custom"
) -> Callable:
    """
    Decorator for registering custom algorithms.
    
    Example:
        @custom_algorithm(name="My Algo", version="1.0.0", author="Me")
        class MyAlgorithm(CustomAlgorithmInterface):
            ...
    """
    def decorator(cls: Type[CustomAlgorithmInterface]) -> Type[CustomAlgorithmInterface]:
        # Store metadata on class
        cls._custom_name = name or cls.__name__
        cls._custom_version = version
        cls._custom_author = author
        cls._custom_type = algorithm_type
        
        # Register automatically
        registry = CustomAlgorithmRegistry()
        registry.register(cls)
        
        return cls
    
    return decorator


class AlgorithmTemplate:
    """Template for creating custom algorithms."""
    
    COLLABORATIVE_TEMPLATE = '''
class {name}(CustomAlgorithmInterface):
    """
    {description}
    
    A collaborative filtering algorithm.
    """
    
    def __init__(self):
        super().__init__()
        self._model = None
    
    def get_metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="{name}",
            version="1.0.0",
            author="{author}",
            description="{description}",
            algorithm_type="collaborative"
        )
    
    def get_capabilities(self) -> AlgorithmCapabilities:
        return AlgorithmCapabilities(
            supports_batch=True,
            supports_incremental=False,
            supports_cold_start=False,
            supports_explanations=False,
            min_ratings_required=10
        )
    
    def train(self, ratings_df, movies_df, **kwargs):
        # TODO: Implement training
        # ratings_df has columns: user_id, movie_id, rating
        self._is_trained = True
    
    def predict(self, user_id, n_recommendations=10, **kwargs):
        # TODO: Implement prediction
        # Return tuple of (movie_ids, scores)
        return [], []
'''
    
    CONTENT_BASED_TEMPLATE = '''
class {name}(CustomAlgorithmInterface):
    """
    {description}
    
    A content-based filtering algorithm.
    """
    
    def __init__(self):
        super().__init__()
        self._feature_matrix = None
    
    def get_metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="{name}",
            version="1.0.0",
            author="{author}",
            description="{description}",
            algorithm_type="content-based"
        )
    
    def get_capabilities(self) -> AlgorithmCapabilities:
        return AlgorithmCapabilities(
            supports_batch=True,
            supports_cold_start=True,
            supports_explanations=True,
            required_features={{"genres", "title"}}
        )
    
    def train(self, ratings_df, movies_df, **kwargs):
        # TODO: Build feature matrix from movie metadata
        self._is_trained = True
    
    def predict(self, user_id, n_recommendations=10, **kwargs):
        # TODO: Generate content-based recommendations
        return [], []
    
    def get_similar_items(self, movie_id, n_similar=10, **kwargs):
        # TODO: Find similar movies based on content
        return [], []
'''
    
    HYBRID_TEMPLATE = '''
class {name}(CustomAlgorithmInterface):
    """
    {description}
    
    A hybrid algorithm combining multiple approaches.
    """
    
    def __init__(self):
        super().__init__()
        self._collaborative_model = None
        self._content_model = None
        self._weights = {{}}
    
    def get_metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="{name}",
            version="1.0.0",
            author="{author}",
            description="{description}",
            algorithm_type="hybrid"
        )
    
    def get_capabilities(self) -> AlgorithmCapabilities:
        return AlgorithmCapabilities(
            supports_batch=True,
            supports_cold_start=True,
            supports_explanations=True,
            supports_diversity=True
        )
    
    def train(self, ratings_df, movies_df, **kwargs):
        # TODO: Train both collaborative and content-based models
        self._is_trained = True
    
    def predict(self, user_id, n_recommendations=10, **kwargs):
        # TODO: Combine predictions from both models
        return [], []
'''
    
    @classmethod
    def generate(
        cls,
        algorithm_type: str,
        name: str,
        author: str = "User",
        description: str = "Custom recommendation algorithm"
    ) -> str:
        """Generate algorithm template code."""
        templates = {
            "collaborative": cls.COLLABORATIVE_TEMPLATE,
            "content-based": cls.CONTENT_BASED_TEMPLATE,
            "hybrid": cls.HYBRID_TEMPLATE
        }
        
        template = templates.get(algorithm_type, cls.COLLABORATIVE_TEMPLATE)
        return template.format(
            name=name,
            author=author,
            description=description
        )


class AlgorithmLoader:
    """Load custom algorithms from files."""
    
    @staticmethod
    def load_from_file(path: Path) -> Optional[Type[CustomAlgorithmInterface]]:
        """
        Load algorithm class from Python file.
        
        Args:
            path: Path to Python file containing algorithm class
            
        Returns:
            Algorithm class or None if loading failed
        """
        import importlib.util
        
        try:
            spec = importlib.util.spec_from_file_location("custom_algorithm", path)
            if spec is None or spec.loader is None:
                logger.error(f"Could not load spec from {path}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find algorithm class in module
            for name in dir(module):
                obj = getattr(module, name)
                if (isinstance(obj, type) and 
                    issubclass(obj, CustomAlgorithmInterface) and
                    obj is not CustomAlgorithmInterface):
                    return obj
            
            logger.error(f"No algorithm class found in {path}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load algorithm from {path}: {e}")
            return None
    
    @staticmethod
    def load_from_directory(directory: Path) -> List[Type[CustomAlgorithmInterface]]:
        """
        Load all algorithms from a directory.
        
        Args:
            directory: Directory containing algorithm files
            
        Returns:
            List of loaded algorithm classes
        """
        algorithms = []
        
        if not directory.exists():
            return algorithms
        
        for path in directory.glob("*.py"):
            if path.name.startswith("_"):
                continue
            
            algo_class = AlgorithmLoader.load_from_file(path)
            if algo_class:
                algorithms.append(algo_class)
        
        return algorithms


class AlgorithmExporter:
    """Export custom algorithms to portable formats."""
    
    @staticmethod
    def export_to_pickle(
        algorithm: CustomAlgorithmInterface,
        path: Path
    ) -> bool:
        """Export trained algorithm to pickle file."""
        try:
            with open(path, 'wb') as f:
                pickle.dump(algorithm, f)
            return True
        except Exception as e:
            logger.error(f"Failed to export algorithm: {e}")
            return False
    
    @staticmethod
    def import_from_pickle(path: Path) -> Optional[CustomAlgorithmInterface]:
        """Import algorithm from pickle file."""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to import algorithm: {e}")
            return None
    
    @staticmethod
    def export_metadata(
        algorithm: CustomAlgorithmInterface,
        path: Path
    ) -> bool:
        """Export algorithm metadata to JSON."""
        try:
            metadata = algorithm.get_metadata()
            capabilities = algorithm.get_capabilities()
            
            data = {
                "metadata": metadata.to_dict(),
                "capabilities": capabilities.to_dict()
            }
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to export metadata: {e}")
            return False


# Example implementation
class SimpleAverageRecommender(CustomAlgorithmInterface):
    """
    Simple recommender based on average ratings.
    
    A basic example implementation for demonstration.
    """
    
    def __init__(self):
        super().__init__()
        self._movie_avg: Dict[int, float] = {}
        self._global_avg: float = 0.0
        self._user_history: Dict[int, Set[int]] = {}
    
    def get_metadata(self) -> AlgorithmMetadata:
        return AlgorithmMetadata(
            name="Simple Average",
            version="1.0.0",
            author="CineMatch Team",
            description="Recommends movies based on average ratings",
            algorithm_type="collaborative",
            tags=["simple", "baseline", "non-personalized"]
        )
    
    def get_capabilities(self) -> AlgorithmCapabilities:
        return AlgorithmCapabilities(
            supports_batch=True,
            supports_cold_start=True,
            supports_explanations=True,
            min_ratings_required=0
        )
    
    def train(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        **kwargs: Any
    ) -> None:
        """Train on ratings data."""
        start = time.time()
        
        # Calculate movie averages
        self._movie_avg = ratings_df.groupby('movie_id')['rating'].mean().to_dict()
        self._global_avg = ratings_df['rating'].mean()
        
        # Track user history
        for user_id, group in ratings_df.groupby('user_id'):
            self._user_history[user_id] = set(group['movie_id'].tolist())
        
        self._is_trained = True
        self._training_time = time.time() - start
        self._last_trained = datetime.utcnow()
    
    def predict(
        self,
        user_id: int,
        n_recommendations: int = 10,
        **kwargs: Any
    ) -> tuple[List[int], List[float]]:
        """Generate recommendations based on average ratings."""
        if not self._is_trained:
            raise RuntimeError("Algorithm not trained")
        
        # Get movies user hasn't seen
        seen = self._user_history.get(user_id, set())
        candidates = [
            (movie_id, avg)
            for movie_id, avg in self._movie_avg.items()
            if movie_id not in seen
        ]
        
        # Sort by average rating
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N
        top_n = candidates[:n_recommendations]
        movie_ids = [m[0] for m in top_n]
        scores = [m[1] for m in top_n]
        
        return movie_ids, scores
    
    def explain_recommendation(
        self,
        user_id: int,
        movie_id: int,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Explain why a movie was recommended."""
        avg = self._movie_avg.get(movie_id, self._global_avg)
        return {
            "reason": "high_average_rating",
            "average_rating": avg,
            "global_average": self._global_avg,
            "explanation": f"This movie has an average rating of {avg:.2f}, which is above the global average of {self._global_avg:.2f}"
        }
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        data = {
            "movie_avg": self._movie_avg,
            "global_avg": self._global_avg,
            "user_history": {k: list(v) for k, v in self._user_history.items()}
        }
        
        with open(path / "model.pkl", 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: Path) -> None:
        """Load model from disk."""
        with open(Path(path) / "model.pkl", 'rb') as f:
            data = pickle.load(f)
        
        self._movie_avg = data["movie_avg"]
        self._global_avg = data["global_avg"]
        self._user_history = {k: set(v) for k, v in data["user_history"].items()}
        self._is_trained = True
