"""
Algorithm Factory

Responsible for algorithm instantiation, registration, and information retrieval.
Handles the "what" and "how" of creating algorithm instances.

Author: CineMatch Development Team
Date: November 12, 2025
"""

from typing import Dict, Any, Type, List
from enum import Enum

# Import algorithm classes
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.algorithms.base_recommender import BaseRecommender
from src.algorithms.svd_recommender import SVDRecommender
from src.algorithms.user_knn_recommender import UserKNNRecommender
from src.algorithms.item_knn_recommender import ItemKNNRecommender
from src.algorithms.content_based_recommender import ContentBasedRecommender
from src.algorithms.hybrid_recommender import HybridRecommender


class AlgorithmType(Enum):
    """Enumeration of available recommendation algorithms"""
    SVD = "SVD Matrix Factorization"
    USER_KNN = "KNN User-Based"
    ITEM_KNN = "KNN Item-Based"
    CONTENT_BASED = "Content-Based Filtering"
    HYBRID = "Hybrid (Best of All)"


class AlgorithmFactory:
    """
    Factory for creating and managing algorithm instances.
    
    Responsibilities:
    - Registering available algorithms
    - Creating algorithm instances with parameters
    - Providing algorithm information and metadata
    - Generating recommendation explanations
    """
    
    def __init__(self):
        """Initialize the algorithm factory with algorithm registry."""
        self._algorithm_classes: Dict[AlgorithmType, Type[BaseRecommender]] = {
            AlgorithmType.SVD: SVDRecommender,
            AlgorithmType.USER_KNN: UserKNNRecommender,
            AlgorithmType.ITEM_KNN: ItemKNNRecommender,
            AlgorithmType.CONTENT_BASED: ContentBasedRecommender,
            AlgorithmType.HYBRID: HybridRecommender
        }
        
        self._default_params: Dict[AlgorithmType, Dict[str, Any]] = {
            AlgorithmType.SVD: {'n_components': 100},
            AlgorithmType.USER_KNN: {'n_neighbors': 50, 'similarity_metric': 'cosine'},
            AlgorithmType.ITEM_KNN: {'n_neighbors': 30, 'similarity_metric': 'cosine', 'min_ratings': 5},
            AlgorithmType.CONTENT_BASED: {
                'genre_weight': 0.5, 
                'tag_weight': 0.3, 
                'title_weight': 0.2,
                'min_similarity': 0.01
            },
            AlgorithmType.HYBRID: {
                'svd_params': {'n_components': 100},
                'user_knn_params': {'n_neighbors': 50, 'similarity_metric': 'cosine'},
                'item_knn_params': {'n_neighbors': 30, 'similarity_metric': 'cosine', 'min_ratings': 5},
                'content_based_params': {
                    'genre_weight': 0.5, 
                    'tag_weight': 0.3, 
                    'title_weight': 0.2,
                    'min_similarity': 0.01
                },
                'weighting_strategy': 'adaptive'
            }
        }
    
    def create_algorithm(self, 
                        algorithm_type: AlgorithmType,
                        custom_params: Dict[str, Any] = None) -> BaseRecommender:
        """
        Create a new algorithm instance.
        
        Args:
            algorithm_type: Type of algorithm to create
            custom_params: Optional custom parameters (overrides defaults)
            
        Returns:
            New BaseRecommender instance
        """
        # Merge default and custom parameters
        params = self._default_params[algorithm_type].copy()
        if custom_params:
            params.update(custom_params)
        
        # Instantiate algorithm
        algorithm_class = self._algorithm_classes[algorithm_type]
        return algorithm_class(**params)
    
    def get_available_algorithms(self) -> List[AlgorithmType]:
        """Get list of all available algorithm types."""
        return list(AlgorithmType)
    
    def get_algorithm_info(self, algorithm_type: AlgorithmType) -> Dict[str, Any]:
        """
        Get detailed information about an algorithm without instantiating it.
        
        Returns:
            Dictionary with algorithm description, capabilities, strengths, etc.
        """
        info_map = {
            AlgorithmType.SVD: {
                'name': 'SVD Matrix Factorization',
                'description': 'Uses Singular Value Decomposition to discover hidden patterns in user ratings. Excellent for finding complex relationships between users and movies.',
                'strengths': ['High accuracy', 'Handles sparse data well', 'Discovers latent factors', 'Good for diverse recommendations'],
                'ideal_for': ['Users with varied taste', 'Discovering hidden gems', 'Academic research', 'High-accuracy needs'],
                'complexity': 'High',
                'speed': 'Medium',
                'interpretability': 'Medium',
                'icon': '🔮'
            },
            AlgorithmType.USER_KNN: {
                'name': 'KNN User-Based',
                'description': 'Finds users with similar taste and recommends movies those users loved. Simple and intuitive approach.',
                'strengths': ['Highly interpretable', 'Good for sparse users', 'Handles new items well', 'Community-based recommendations'],
                'ideal_for': ['New users', 'Sparse rating profiles', 'Social recommendations', 'Explainable results'],
                'complexity': 'Low',
                'speed': 'Fast',
                'interpretability': 'Very High',
                'icon': '👥'
            },
            AlgorithmType.ITEM_KNN: {
                'name': 'KNN Item-Based',
                'description': 'Analyzes movies with similar rating patterns and recommends items similar to what you enjoyed.',
                'strengths': ['Stable recommendations', 'Good for frequent users', 'Pre-computed similarities', 'Genre-aware'],
                'ideal_for': ['Users with many ratings', 'Discovering similar movies', 'Stable preferences', 'Genre exploration'],
                'complexity': 'Medium',
                'speed': 'Fast',
                'interpretability': 'High',
                'icon': '🎬'
            },
            AlgorithmType.CONTENT_BASED: {
                'name': 'Content-Based Filtering',
                'description': 'Analyzes movie features (genres, tags, titles) and recommends movies similar to what you enjoyed. Perfect for cold-start scenarios.',
                'strengths': ['No cold-start problem', 'Feature-based recommendations', 'Highly interpretable', 'Tag and genre aware', 'Works for new users'],
                'ideal_for': ['New users', 'Genre-specific discovery', 'Feature-based exploration', 'Cold-start scenarios', 'Explainable recommendations'],
                'complexity': 'Medium',
                'speed': 'Fast',
                'interpretability': 'Very High',
                'icon': '🔍'
            },
            AlgorithmType.HYBRID: {
                'name': 'Hybrid (Best of All)',
                'description': 'Intelligently combines all algorithms with dynamic weighting based on your profile and context.',
                'strengths': ['Best overall accuracy', 'Adapts to user type', 'Robust performance', 'Combines multiple paradigms'],
                'ideal_for': ['Production systems', 'Best accuracy', 'All user types', 'Research comparison'],
                'complexity': 'Very High',
                'speed': 'Medium',
                'interpretability': 'Medium',
                'icon': '🚀'
            }
        }
        
        return info_map.get(algorithm_type, {})
    
    def get_recommendation_explanation(self,
                                      algorithm_type: AlgorithmType,
                                      context: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation for a recommendation.
        
        Args:
            algorithm_type: Algorithm that made the recommendation
            context: Context dictionary from algorithm.get_explanation_context()
            
        Returns:
            Human-readable explanation string
        """
        if not context:
            return "Unable to generate explanation."
        
        # Generate explanation based on algorithm type and context
        if algorithm_type == AlgorithmType.SVD:
            return self._explain_svd(context)
        elif algorithm_type == AlgorithmType.USER_KNN:
            return self._explain_user_knn(context)
        elif algorithm_type == AlgorithmType.ITEM_KNN:
            return self._explain_item_knn(context)
        elif algorithm_type == AlgorithmType.HYBRID:
            return self._explain_hybrid(context)
        elif algorithm_type == AlgorithmType.CONTENT_BASED:
            return self._explain_content_based(context)
        
        return "Explanation not available."
    
    def _explain_svd(self, context: Dict[str, Any]) -> str:
        """Generate SVD-specific explanation."""
        pred = context.get('prediction', 0)
        return (f"SVD predicts you'll rate this movie **{pred:.1f}/5.0** based on "
               f"latent patterns discovered in your rating history and similar users' preferences.")
    
    def _explain_user_knn(self, context: Dict[str, Any]) -> str:
        """Generate User KNN-specific explanation."""
        similar_users = context.get('similar_users_count', 0)
        pred = context.get('prediction', 0)
        
        if similar_users > 0:
            return (f"**{similar_users} users** with similar taste loved this movie! "
                   f"Predicted rating: **{pred:.1f}/5.0**")
        else:
            return f"Based on users with similar preferences. Predicted rating: **{pred:.1f}/5.0**"
    
    def _explain_item_knn(self, context: Dict[str, Any]) -> str:
        """Generate Item KNN-specific explanation."""
        similar_movies = context.get('similar_movies_count', 0)
        pred = context.get('prediction', 0)
        
        if similar_movies > 0:
            return (f"Because you enjoyed **{similar_movies} similar movies**. "
                   f"Predicted rating: **{pred:.1f}/5.0**")
        else:
            return f"Based on movies with similar rating patterns. Predicted rating: **{pred:.1f}/5.0**"
    
    def _explain_content_based(self, context: Dict[str, Any]) -> str:
        """Generate Content-Based-specific explanation."""
        pred = context.get('prediction', 0)
        genres = context.get('matching_genres', [])
        tags = context.get('matching_tags', [])
        
        explanation = f"Predicted rating: **{pred:.1f}/5.0** based on content similarity. "
        if genres:
            explanation += f"Matching genres: {', '.join(genres[:3])}. "
        if tags:
            explanation += f"Shared tags: {', '.join(tags[:3])}."
        
        return explanation
    
    def _explain_hybrid(self, context: Dict[str, Any]) -> str:
        """Generate Hybrid-specific explanation."""
        primary = context.get('primary_algorithm', 'multiple algorithms')
        pred = context.get('prediction', 0)
        weights = context.get('algorithm_weights', {})
        
        return (f"Hybrid prediction (**{pred:.1f}/5.0**) combining multiple algorithms. "
               f"Primary method: **{primary}**. Algorithm weights: "
               f"SVD {weights.get('svd', 0):.2f}, User KNN {weights.get('user_knn', 0):.2f}, "
               f"Item KNN {weights.get('item_knn', 0):.2f}, CBF {weights.get('content_based', 0):.2f}")
