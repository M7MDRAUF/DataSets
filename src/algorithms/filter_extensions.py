"""
CineMatch V2.1.6 - Filter Extensions

Extensible framework for custom recommendation filters.
Allows users to define and chain filtering rules.

Author: CineMatch Development Team
"""

from abc import ABC, abstractmethod
from typing import (
    Dict, Any, Optional, List, Callable, Type, Union, Set
)
from dataclasses import dataclass, field
from enum import Enum
import logging
import re
from datetime import datetime
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


# =============================================================================
# Filter Types
# =============================================================================

class FilterType(Enum):
    """Types of recommendation filters"""
    GENRE = "genre"
    YEAR = "year"
    RATING = "rating"
    POPULARITY = "popularity"
    CONTENT = "content"
    USER_PREFERENCE = "user_preference"
    BUSINESS_RULE = "business_rule"
    DIVERSITY = "diversity"
    FRESHNESS = "freshness"
    CUSTOM = "custom"


class FilterOperation(Enum):
    """Filter comparison operations"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    MATCHES = "matches"  # Regex
    BETWEEN = "between"


class CombineMode(Enum):
    """How to combine multiple filters"""
    AND = "and"
    OR = "or"


# =============================================================================
# Filter Configuration
# =============================================================================

@dataclass
class FilterConfig:
    """Configuration for a filter"""
    name: str
    filter_type: FilterType
    enabled: bool = True
    priority: int = 0  # Higher priority filters run first
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterResult:
    """Result of applying a filter"""
    filter_name: str
    items_before: int
    items_after: int
    items_removed: List[int] = field(default_factory=list)
    execution_time_ms: float = 0.0


# =============================================================================
# Base Filter Class
# =============================================================================

class BaseFilter(ABC):
    """
    Abstract base class for recommendation filters.
    
    All custom filters should inherit from this class and implement
    the apply() method.
    """
    
    def __init__(self, config: FilterConfig):
        self.config = config
        self.name = config.name
        self.enabled = config.enabled
        
    @abstractmethod
    def apply(
        self,
        items: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Apply filter to items.
        
        Args:
            items: DataFrame with recommendations (must have 'movieId')
            context: Optional context (user_id, preferences, etc.)
            
        Returns:
            Filtered DataFrame
        """
        pass
    
    def validate(self, items: pd.DataFrame) -> bool:
        """Validate input DataFrame"""
        if items.empty:
            return True
        if 'movieId' not in items.columns:
            logger.warning(f"Filter {self.name}: 'movieId' column required")
            return False
        return True
    
    def __call__(
        self,
        items: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Allow filter to be called like a function"""
        if not self.enabled:
            return items
        if not self.validate(items):
            return items
        return self.apply(items, context)


# =============================================================================
# Built-in Filters
# =============================================================================

class GenreFilter(BaseFilter):
    """Filter by genre inclusion/exclusion"""
    
    def __init__(
        self,
        include_genres: Optional[List[str]] = None,
        exclude_genres: Optional[List[str]] = None,
        require_all: bool = False
    ):
        super().__init__(FilterConfig(
            name="Genre Filter",
            filter_type=FilterType.GENRE,
            parameters={
                'include': include_genres or [],
                'exclude': exclude_genres or [],
                'require_all': require_all
            }
        ))
        self.include_genres = set(g.lower() for g in (include_genres or []))
        self.exclude_genres = set(g.lower() for g in (exclude_genres or []))
        self.require_all = require_all
    
    def apply(
        self,
        items: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        if 'genres' not in items.columns:
            return items
        
        def matches_genres(genres) -> bool:
            if isinstance(genres, str):
                item_genres = set(g.lower() for g in genres.split('|'))
            elif isinstance(genres, list):
                item_genres = set(g.lower() for g in genres)
            else:
                return True
            
            # Check exclusions first
            if self.exclude_genres and (item_genres & self.exclude_genres):
                return False
            
            # Check inclusions
            if self.include_genres:
                if self.require_all:
                    return self.include_genres <= item_genres
                else:
                    return bool(self.include_genres & item_genres)
            
            return True
        
        mask = items['genres'].apply(matches_genres)
        return items[mask]


class YearFilter(BaseFilter):
    """Filter by release year"""
    
    def __init__(
        self,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        specific_years: Optional[List[int]] = None,
        exclude_years: Optional[List[int]] = None
    ):
        super().__init__(FilterConfig(
            name="Year Filter",
            filter_type=FilterType.YEAR,
            parameters={
                'min_year': min_year,
                'max_year': max_year,
                'specific_years': specific_years,
                'exclude_years': exclude_years
            }
        ))
        self.min_year = min_year
        self.max_year = max_year
        self.specific_years = set(specific_years) if specific_years else None
        self.exclude_years = set(exclude_years) if exclude_years else set()
    
    def apply(
        self,
        items: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        if 'year' not in items.columns:
            # Try to extract year from title
            if 'title' in items.columns:
                items = items.copy()
                items['year'] = items['title'].str.extract(r'\((\d{4})\)').astype(float)
            else:
                return items
        
        mask = pd.Series(True, index=items.index)
        
        if self.min_year:
            mask &= items['year'] >= self.min_year
        if self.max_year:
            mask &= items['year'] <= self.max_year
        if self.specific_years:
            mask &= items['year'].isin(self.specific_years)
        if self.exclude_years:
            mask &= ~items['year'].isin(self.exclude_years)
        
        return items[mask]


class RatingFilter(BaseFilter):
    """Filter by predicted/average rating threshold"""
    
    def __init__(
        self,
        min_rating: float = 0.0,
        max_rating: float = 5.0,
        rating_column: str = 'predicted_rating'
    ):
        super().__init__(FilterConfig(
            name="Rating Filter",
            filter_type=FilterType.RATING,
            parameters={
                'min_rating': min_rating,
                'max_rating': max_rating,
                'rating_column': rating_column
            }
        ))
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.rating_column = rating_column
    
    def apply(
        self,
        items: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        if self.rating_column not in items.columns:
            return items
        
        mask = (
            (items[self.rating_column] >= self.min_rating) &
            (items[self.rating_column] <= self.max_rating)
        )
        return items[mask]


class PopularityFilter(BaseFilter):
    """Filter by item popularity"""
    
    def __init__(
        self,
        min_ratings_count: int = 0,
        max_ratings_count: Optional[int] = None,
        percentile_range: Optional[Tuple[float, float]] = None
    ):
        super().__init__(FilterConfig(
            name="Popularity Filter",
            filter_type=FilterType.POPULARITY,
            parameters={
                'min_ratings_count': min_ratings_count,
                'max_ratings_count': max_ratings_count,
                'percentile_range': percentile_range
            }
        ))
        self.min_ratings_count = min_ratings_count
        self.max_ratings_count = max_ratings_count
        self.percentile_range = percentile_range
    
    def apply(
        self,
        items: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        if 'ratings_count' not in items.columns:
            return items
        
        mask = items['ratings_count'] >= self.min_ratings_count
        
        if self.max_ratings_count:
            mask &= items['ratings_count'] <= self.max_ratings_count
        
        if self.percentile_range:
            low, high = self.percentile_range
            low_val = items['ratings_count'].quantile(low / 100)
            high_val = items['ratings_count'].quantile(high / 100)
            mask &= (items['ratings_count'] >= low_val) & (items['ratings_count'] <= high_val)
        
        return items[mask]


class WatchedFilter(BaseFilter):
    """Filter out items user has already watched/rated"""
    
    def __init__(self):
        super().__init__(FilterConfig(
            name="Watched Filter",
            filter_type=FilterType.USER_PREFERENCE,
            priority=100  # High priority - run early
        ))
    
    def apply(
        self,
        items: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        if context is None:
            return items
        
        watched_items = context.get('watched_items', set())
        if not watched_items:
            return items
        
        return items[~items['movieId'].isin(watched_items)]


class DiversityFilter(BaseFilter):
    """Ensure diversity in recommendations"""
    
    def __init__(
        self,
        max_per_genre: int = 3,
        max_per_year_decade: int = 5,
        min_genres: int = 3
    ):
        super().__init__(FilterConfig(
            name="Diversity Filter",
            filter_type=FilterType.DIVERSITY,
            parameters={
                'max_per_genre': max_per_genre,
                'max_per_year_decade': max_per_year_decade,
                'min_genres': min_genres
            }
        ))
        self.max_per_genre = max_per_genre
        self.max_per_year_decade = max_per_year_decade
        self.min_genres = min_genres
    
    def apply(
        self,
        items: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        if items.empty or 'genres' not in items.columns:
            return items
        
        result_indices = []
        genre_counts: Dict[str, int] = {}
        decade_counts: Dict[int, int] = {}
        genres_seen: Set[str] = set()
        
        for idx, row in items.iterrows():
            # Parse genres
            genres = row['genres']
            if isinstance(genres, str):
                item_genres = [g.lower() for g in genres.split('|')]
            elif isinstance(genres, list):
                item_genres = [g.lower() for g in genres]
            else:
                item_genres = []
            
            # Check genre limits
            genre_ok = all(
                genre_counts.get(g, 0) < self.max_per_genre
                for g in item_genres
            )
            
            # Check decade limit
            year = row.get('year', 2000)
            decade = (int(year) // 10) * 10 if pd.notna(year) else 2000
            decade_ok = decade_counts.get(decade, 0) < self.max_per_year_decade
            
            # Check if we need more genres
            needs_genre = len(genres_seen) < self.min_genres
            new_genre = bool(set(item_genres) - genres_seen)
            
            if genre_ok and decade_ok and (not needs_genre or new_genre):
                result_indices.append(idx)
                for g in item_genres:
                    genre_counts[g] = genre_counts.get(g, 0) + 1
                    genres_seen.add(g)
                decade_counts[decade] = decade_counts.get(decade, 0) + 1
        
        return items.loc[result_indices]


class FreshnessFilter(BaseFilter):
    """Filter by content freshness/recency"""
    
    def __init__(
        self,
        max_age_years: Optional[int] = None,
        prefer_recent: bool = False,
        recent_boost_years: int = 5
    ):
        super().__init__(FilterConfig(
            name="Freshness Filter",
            filter_type=FilterType.FRESHNESS,
            parameters={
                'max_age_years': max_age_years,
                'prefer_recent': prefer_recent,
                'recent_boost_years': recent_boost_years
            }
        ))
        self.max_age_years = max_age_years
        self.prefer_recent = prefer_recent
        self.recent_boost_years = recent_boost_years
    
    def apply(
        self,
        items: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        if 'year' not in items.columns:
            return items
        
        current_year = datetime.now().year
        items = items.copy()
        
        if self.max_age_years:
            min_year = current_year - self.max_age_years
            items = items[items['year'] >= min_year]
        
        if self.prefer_recent and 'score' in items.columns:
            # Boost recent items
            recent_threshold = current_year - self.recent_boost_years
            boost = items['year'].apply(
                lambda y: 1.1 if y >= recent_threshold else 1.0
            )
            items['score'] = items['score'] * boost
            items = items.sort_values('score', ascending=False)
        
        return items


class ContentFilter(BaseFilter):
    """Filter by content attributes (keywords, tags)"""
    
    def __init__(
        self,
        include_keywords: Optional[List[str]] = None,
        exclude_keywords: Optional[List[str]] = None,
        keyword_column: str = 'keywords'
    ):
        super().__init__(FilterConfig(
            name="Content Filter",
            filter_type=FilterType.CONTENT,
            parameters={
                'include_keywords': include_keywords,
                'exclude_keywords': exclude_keywords,
                'keyword_column': keyword_column
            }
        ))
        self.include_keywords = [k.lower() for k in (include_keywords or [])]
        self.exclude_keywords = [k.lower() for k in (exclude_keywords or [])]
        self.keyword_column = keyword_column
    
    def apply(
        self,
        items: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        if self.keyword_column not in items.columns:
            # Try title search
            if 'title' not in items.columns:
                return items
            search_col = 'title'
        else:
            search_col = self.keyword_column
        
        def matches_content(content) -> bool:
            if pd.isna(content):
                return True
            content_lower = str(content).lower()
            
            # Check exclusions
            for kw in self.exclude_keywords:
                if kw in content_lower:
                    return False
            
            # Check inclusions
            if self.include_keywords:
                return any(kw in content_lower for kw in self.include_keywords)
            
            return True
        
        mask = items[search_col].apply(matches_content)
        return items[mask]


class BusinessRuleFilter(BaseFilter):
    """Apply business rules to recommendations"""
    
    def __init__(
        self,
        exclude_ids: Optional[List[int]] = None,
        boost_ids: Optional[List[int]] = None,
        boost_factor: float = 1.5,
        min_score: Optional[float] = None
    ):
        super().__init__(FilterConfig(
            name="Business Rule Filter",
            filter_type=FilterType.BUSINESS_RULE,
            parameters={
                'exclude_ids': exclude_ids,
                'boost_ids': boost_ids,
                'boost_factor': boost_factor,
                'min_score': min_score
            }
        ))
        self.exclude_ids = set(exclude_ids or [])
        self.boost_ids = set(boost_ids or [])
        self.boost_factor = boost_factor
        self.min_score = min_score
    
    def apply(
        self,
        items: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        if items.empty:
            return items
        
        items = items.copy()
        
        # Exclude specific items
        if self.exclude_ids:
            items = items[~items['movieId'].isin(self.exclude_ids)]
        
        # Boost specific items
        if self.boost_ids and 'score' in items.columns:
            boost_mask = items['movieId'].isin(self.boost_ids)
            items.loc[boost_mask, 'score'] *= self.boost_factor
            items = items.sort_values('score', ascending=False)
        
        # Apply minimum score threshold
        if self.min_score and 'score' in items.columns:
            items = items[items['score'] >= self.min_score]
        
        return items


# =============================================================================
# Custom Filter Builder
# =============================================================================

class FilterBuilder:
    """
    Builder pattern for creating custom filters.
    
    Example:
        filter = (FilterBuilder("my_filter")
            .filter_type(FilterType.CUSTOM)
            .condition(lambda row: row['rating'] > 4.0)
            .priority(50)
            .build())
    """
    
    def __init__(self, name: str):
        self._name = name
        self._filter_type = FilterType.CUSTOM
        self._condition: Optional[Callable] = None
        self._priority = 0
        self._description = ""
        self._enabled = True
    
    def filter_type(self, ft: FilterType) -> 'FilterBuilder':
        self._filter_type = ft
        return self
    
    def condition(self, fn: Callable[[pd.Series], bool]) -> 'FilterBuilder':
        self._condition = fn
        return self
    
    def priority(self, p: int) -> 'FilterBuilder':
        self._priority = p
        return self
    
    def description(self, desc: str) -> 'FilterBuilder':
        self._description = desc
        return self
    
    def enabled(self, e: bool) -> 'FilterBuilder':
        self._enabled = e
        return self
    
    def build(self) -> BaseFilter:
        """Build the custom filter"""
        if self._condition is None:
            raise ValueError("Condition function is required")
        
        config = FilterConfig(
            name=self._name,
            filter_type=self._filter_type,
            enabled=self._enabled,
            priority=self._priority,
            description=self._description
        )
        
        condition = self._condition
        
        class CustomFilter(BaseFilter):
            def apply(self, items, context=None):
                mask = items.apply(condition, axis=1)
                return items[mask]
        
        return CustomFilter(config)


# =============================================================================
# Filter Chain
# =============================================================================

class FilterChain:
    """
    Chain of filters to apply in sequence.
    
    Filters are sorted by priority (higher first) and applied in order.
    """
    
    def __init__(self, name: str = "default"):
        self.name = name
        self._filters: List[BaseFilter] = []
        self._results: List[FilterResult] = []
    
    def add(self, filter_: BaseFilter) -> 'FilterChain':
        """Add a filter to the chain"""
        self._filters.append(filter_)
        # Sort by priority (higher first)
        self._filters.sort(key=lambda f: -f.config.priority)
        return self
    
    def remove(self, filter_name: str) -> 'FilterChain':
        """Remove a filter by name"""
        self._filters = [f for f in self._filters if f.name != filter_name]
        return self
    
    def apply(
        self,
        items: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Apply all filters in the chain"""
        import time
        
        self._results = []
        result = items
        
        for filter_ in self._filters:
            if not filter_.enabled:
                continue
            
            start = time.time()
            items_before = len(result)
            
            filtered = filter_(result, context)
            
            items_after = len(filtered)
            elapsed = (time.time() - start) * 1000
            
            removed_ids = list(
                set(result['movieId'].tolist()) - set(filtered['movieId'].tolist())
            ) if 'movieId' in result.columns else []
            
            self._results.append(FilterResult(
                filter_name=filter_.name,
                items_before=items_before,
                items_after=items_after,
                items_removed=removed_ids,
                execution_time_ms=elapsed
            ))
            
            result = filtered
            
            logger.debug(
                f"Filter {filter_.name}: {items_before} -> {items_after} "
                f"({elapsed:.2f}ms)"
            )
        
        return result
    
    def get_results(self) -> List[FilterResult]:
        """Get results from last apply() call"""
        return self._results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of filter chain execution"""
        if not self._results:
            return {}
        
        return {
            'chain_name': self.name,
            'filters_applied': len(self._results),
            'total_items_removed': sum(
                r.items_before - r.items_after for r in self._results
            ),
            'total_time_ms': sum(r.execution_time_ms for r in self._results),
            'filter_results': [
                {
                    'name': r.filter_name,
                    'removed': r.items_before - r.items_after,
                    'time_ms': r.execution_time_ms
                }
                for r in self._results
            ]
        }
    
    def __len__(self) -> int:
        return len(self._filters)
    
    def __iter__(self):
        return iter(self._filters)


# =============================================================================
# Filter Registry
# =============================================================================

class FilterRegistry:
    """Registry for managing filters"""
    
    _instance: Optional['FilterRegistry'] = None
    
    def __new__(cls) -> 'FilterRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._filters: Dict[str, Type[BaseFilter]] = {}
            cls._instance._chains: Dict[str, FilterChain] = {}
            cls._instance._register_builtins()
        return cls._instance
    
    def _register_builtins(self):
        """Register built-in filters"""
        self.register_type("genre", GenreFilter)
        self.register_type("year", YearFilter)
        self.register_type("rating", RatingFilter)
        self.register_type("popularity", PopularityFilter)
        self.register_type("watched", WatchedFilter)
        self.register_type("diversity", DiversityFilter)
        self.register_type("freshness", FreshnessFilter)
        self.register_type("content", ContentFilter)
        self.register_type("business_rule", BusinessRuleFilter)
    
    def register_type(self, name: str, filter_class: Type[BaseFilter]):
        """Register a filter type"""
        self._filters[name.lower()] = filter_class
        logger.info(f"Registered filter: {name}")
    
    def create(self, filter_type: str, **kwargs) -> Optional[BaseFilter]:
        """Create a filter instance"""
        filter_class = self._filters.get(filter_type.lower())
        if filter_class is None:
            logger.error(f"Unknown filter type: {filter_type}")
            return None
        
        try:
            return filter_class(**kwargs)
        except Exception as e:
            logger.error(f"Error creating filter: {e}")
            return None
    
    def create_chain(self, name: str) -> FilterChain:
        """Create a new filter chain"""
        chain = FilterChain(name)
        self._chains[name] = chain
        return chain
    
    def get_chain(self, name: str) -> Optional[FilterChain]:
        """Get a filter chain by name"""
        return self._chains.get(name)
    
    def list_types(self) -> List[str]:
        """List available filter types"""
        return list(self._filters.keys())


# =============================================================================
# Convenience Functions
# =============================================================================

def create_filter(filter_type: str, **kwargs) -> Optional[BaseFilter]:
    """Create a filter"""
    return FilterRegistry().create(filter_type, **kwargs)


def create_filter_chain(name: str) -> FilterChain:
    """Create a filter chain"""
    return FilterRegistry().create_chain(name)


def apply_filters(
    items: pd.DataFrame,
    filters: List[BaseFilter],
    context: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Apply multiple filters to items"""
    chain = FilterChain()
    for f in filters:
        chain.add(f)
    return chain.apply(items, context)


def filter_recommendations(
    items: pd.DataFrame,
    user_id: Optional[int] = None,
    watched_items: Optional[Set[int]] = None,
    include_genres: Optional[List[str]] = None,
    exclude_genres: Optional[List[str]] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    min_rating: Optional[float] = None
) -> pd.DataFrame:
    """
    Convenience function to apply common filters.
    
    Args:
        items: DataFrame with recommendations
        user_id: User ID for context
        watched_items: Items to exclude
        include_genres: Genres to include
        exclude_genres: Genres to exclude
        min_year: Minimum release year
        max_year: Maximum release year
        min_rating: Minimum predicted rating
        
    Returns:
        Filtered recommendations
    """
    chain = FilterChain("quick_filter")
    
    context = {
        'user_id': user_id,
        'watched_items': watched_items or set()
    }
    
    # Add watched filter
    if watched_items:
        chain.add(WatchedFilter())
    
    # Add genre filter
    if include_genres or exclude_genres:
        chain.add(GenreFilter(
            include_genres=include_genres,
            exclude_genres=exclude_genres
        ))
    
    # Add year filter
    if min_year or max_year:
        chain.add(YearFilter(
            min_year=min_year,
            max_year=max_year
        ))
    
    # Add rating filter
    if min_rating:
        chain.add(RatingFilter(min_rating=min_rating))
    
    return chain.apply(items, context)


# Type alias for cleaner imports
from typing import Tuple
