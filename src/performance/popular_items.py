"""
Popular Items Cache for CineMatch V2.1.6

Implements caching for popular/trending items:
- Time-windowed popularity
- Precomputed rankings
- Efficient retrieval
- Cache warming

Phase 2 - Task 2.8: Popular Items Caching
"""

import logging
import time
import threading
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
import heapq

logger = logging.getLogger(__name__)

T = TypeVar('T')


class PopularityWindow(Enum):
    """Time window for popularity calculation."""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    ALL_TIME = "all_time"


@dataclass
class PopularItem(Generic[T]):
    """A popular item with score."""
    item: T
    score: float
    count: int
    last_updated: float
    window: PopularityWindow


@dataclass 
class PopularityConfig:
    """Configuration for popularity tracking."""
    max_items: int = 1000
    windows: List[PopularityWindow] = field(
        default_factory=lambda: [
            PopularityWindow.DAY,
            PopularityWindow.WEEK,
            PopularityWindow.ALL_TIME
        ]
    )
    decay_factor: float = 0.9
    update_interval: float = 300  # 5 minutes
    precompute_top_n: int = 100


class PopularItemsCache:
    """
    Cache for popular items with time-windowed popularity.
    
    Features:
    - Multiple time windows
    - Decay-based scoring
    - Efficient top-k retrieval
    - Automatic updates
    """
    
    def __init__(self, config: Optional[PopularityConfig] = None):
        self.config = config or PopularityConfig()
        
        # Storage: window -> item_id -> PopularItem
        self._cache: Dict[PopularityWindow, Dict[Any, PopularItem]] = {
            window: {} for window in self.config.windows
        }
        
        # Precomputed top items
        self._top_items: Dict[PopularityWindow, List[PopularItem]] = {
            window: [] for window in self.config.windows
        }
        
        # Interaction tracking
        self._interactions: Dict[PopularityWindow, Dict[Any, List[float]]] = {
            window: defaultdict(list) for window in self.config.windows
        }
        
        self._lock = threading.RLock()
        self._last_update = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def record_interaction(
        self,
        item_id: Any,
        weight: float = 1.0,
        timestamp: Optional[float] = None
    ) -> None:
        """Record an interaction with an item."""
        ts = timestamp or time.time()
        
        with self._lock:
            for window in self.config.windows:
                self._interactions[window][item_id].append(ts)
    
    def get_top_items(
        self,
        window: PopularityWindow = PopularityWindow.DAY,
        n: int = 10,
        offset: int = 0
    ) -> List[PopularItem]:
        """Get top N popular items."""
        self._maybe_update()
        
        with self._lock:
            items = self._top_items.get(window, [])
            return items[offset:offset + n]
    
    def get_item_rank(
        self,
        item_id: Any,
        window: PopularityWindow = PopularityWindow.DAY
    ) -> Optional[int]:
        """Get rank of an item (1-indexed)."""
        with self._lock:
            items = self._top_items.get(window, [])
            for i, item in enumerate(items):
                if item.item == item_id:
                    return i + 1
        return None
    
    def get_item_score(
        self,
        item_id: Any,
        window: PopularityWindow = PopularityWindow.DAY
    ) -> Optional[float]:
        """Get popularity score of an item."""
        with self._lock:
            if window in self._cache and item_id in self._cache[window]:
                return self._cache[window][item_id].score
        return None
    
    def _maybe_update(self) -> None:
        """Update if interval has passed."""
        now = time.time()
        if now - self._last_update >= self.config.update_interval:
            self._update_all()
    
    def _update_all(self) -> None:
        """Update all popularity scores."""
        now = time.time()
        
        with self._lock:
            for window in self.config.windows:
                self._update_window(window, now)
            
            self._last_update = now
    
    def _update_window(self, window: PopularityWindow, now: float) -> None:
        """Update scores for a time window."""
        cutoff = self._get_window_cutoff(window, now)
        
        # Calculate scores
        scores: Dict[Any, float] = {}
        counts: Dict[Any, int] = {}
        
        for item_id, timestamps in self._interactions[window].items():
            # Filter by time window
            valid_times = [t for t in timestamps if t >= cutoff]
            
            if not valid_times:
                continue
            
            # Calculate score with decay
            score = 0.0
            for ts in valid_times:
                age = now - ts
                decay = self.config.decay_factor ** (age / 3600)  # Decay per hour
                score += decay
            
            scores[item_id] = score
            counts[item_id] = len(valid_times)
        
        # Update cache
        new_cache: Dict[Any, PopularItem] = {}
        for item_id, score in scores.items():
            new_cache[item_id] = PopularItem(
                item=item_id,
                score=score,
                count=counts[item_id],
                last_updated=now,
                window=window
            )
        
        self._cache[window] = new_cache
        
        # Update top items
        top_items = heapq.nlargest(
            self.config.precompute_top_n,
            new_cache.values(),
            key=lambda x: x.score
        )
        self._top_items[window] = top_items
        
        # Cleanup old interactions
        for item_id in list(self._interactions[window].keys()):
            self._interactions[window][item_id] = [
                t for t in self._interactions[window][item_id]
                if t >= cutoff
            ]
            if not self._interactions[window][item_id]:
                del self._interactions[window][item_id]
    
    def _get_window_cutoff(self, window: PopularityWindow, now: float) -> float:
        """Get cutoff timestamp for a window."""
        if window == PopularityWindow.HOUR:
            return now - 3600
        elif window == PopularityWindow.DAY:
            return now - 86400
        elif window == PopularityWindow.WEEK:
            return now - 604800
        elif window == PopularityWindow.MONTH:
            return now - 2592000
        else:  # ALL_TIME
            return 0
    
    def start_background_update(self) -> None:
        """Start background update thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._update_loop,
            daemon=True
        )
        self._thread.start()
        logger.info("Popular items background update started")
    
    def stop_background_update(self) -> None:
        """Stop background update."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
    
    def _update_loop(self) -> None:
        """Background update loop."""
        while self._running:
            try:
                self._update_all()
            except Exception as e:
                logger.error(f"Popular items update error: {e}")
            time.sleep(self.config.update_interval)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'windows': {
                    w.value: {
                        'items_tracked': len(self._cache.get(w, {})),
                        'top_items_count': len(self._top_items.get(w, []))
                    }
                    for w in self.config.windows
                },
                'last_update': self._last_update,
                'running': self._running
            }
    
    def warm_cache(
        self,
        items: List[tuple[Any, float, float]]
    ) -> None:
        """
        Warm cache with historical data.
        
        Args:
            items: List of (item_id, count, timestamp) tuples
        """
        for item_id, count, timestamp in items:
            for _ in range(int(count)):
                self.record_interaction(item_id, timestamp=timestamp)
        
        self._update_all()
        logger.info(f"Warmed cache with {len(items)} items")


class MoviePopularityCache(PopularItemsCache):
    """
    Specialized popularity cache for movies.
    
    Features:
    - Genre-based filtering
    - Rating-weighted popularity
    - Trending detection
    """
    
    def __init__(self, config: Optional[PopularityConfig] = None):
        super().__init__(config)
        self._movie_metadata: Dict[int, Dict[str, Any]] = {}
        self._genre_index: Dict[str, set] = defaultdict(set)
    
    def set_movie_metadata(
        self,
        movie_id: int,
        metadata: Dict[str, Any]
    ) -> None:
        """Set metadata for a movie."""
        self._movie_metadata[movie_id] = metadata
        
        # Index by genre
        genres = metadata.get('genres', [])
        if isinstance(genres, str):
            genres = [g.strip() for g in genres.split('|')]
        
        for genre in genres:
            self._genre_index[genre.lower()].add(movie_id)
    
    def record_rating(
        self,
        movie_id: int,
        rating: float,
        user_id: Optional[int] = None,
        timestamp: Optional[float] = None
    ) -> None:
        """Record a movie rating."""
        # Weight by rating (higher ratings = more weight)
        weight = rating / 5.0  # Normalize to 0-1
        self.record_interaction(movie_id, weight=weight, timestamp=timestamp)
    
    def get_popular_by_genre(
        self,
        genre: str,
        window: PopularityWindow = PopularityWindow.DAY,
        n: int = 10
    ) -> List[PopularItem]:
        """Get popular movies in a genre."""
        self._maybe_update()
        
        genre_lower = genre.lower()
        genre_movies = self._genre_index.get(genre_lower, set())
        
        with self._lock:
            all_items = self._top_items.get(window, [])
            filtered = [
                item for item in all_items
                if item.item in genre_movies
            ]
            return filtered[:n]
    
    def get_trending(
        self,
        short_window: PopularityWindow = PopularityWindow.HOUR,
        long_window: PopularityWindow = PopularityWindow.WEEK,
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get trending movies (rising popularity).
        
        Compares short-term to long-term popularity.
        """
        self._maybe_update()
        
        with self._lock:
            short_scores = {
                item.item: item.score
                for item in self._top_items.get(short_window, [])
            }
            long_scores = {
                item.item: item.score
                for item in self._top_items.get(long_window, [])
            }
        
        # Calculate trending score
        trending: List[Dict[str, Any]] = []
        
        for movie_id, short_score in short_scores.items():
            long_score = long_scores.get(movie_id, 0.01)  # Avoid division by zero
            trend_score = short_score / long_score
            
            if trend_score > 1.0:  # Rising
                trending.append({
                    'movie_id': movie_id,
                    'trend_score': trend_score,
                    'short_term_score': short_score,
                    'long_term_score': long_score,
                    'metadata': self._movie_metadata.get(movie_id, {})
                })
        
        # Sort by trend score
        trending.sort(key=lambda x: x['trend_score'], reverse=True)
        return trending[:n]


# Global instances
_movie_popularity_cache: Optional[MoviePopularityCache] = None


def get_movie_popularity_cache() -> MoviePopularityCache:
    global _movie_popularity_cache
    if _movie_popularity_cache is None:
        _movie_popularity_cache = MoviePopularityCache()
    return _movie_popularity_cache


def get_popular_movies(
    window: PopularityWindow = PopularityWindow.DAY,
    n: int = 10
) -> List[PopularItem]:
    """Get popular movies."""
    return get_movie_popularity_cache().get_top_items(window, n)


def get_trending_movies(n: int = 10) -> List[Dict[str, Any]]:
    """Get trending movies."""
    return get_movie_popularity_cache().get_trending(n=n)


def record_movie_interaction(
    movie_id: int,
    rating: Optional[float] = None
) -> None:
    """Record a movie interaction."""
    cache = get_movie_popularity_cache()
    if rating:
        cache.record_rating(movie_id, rating)
    else:
        cache.record_interaction(movie_id)


if __name__ == "__main__":
    import random
    
    print("Popular Items Cache Demo")
    print("=" * 50)
    
    # Create cache
    config = PopularityConfig(
        windows=[
            PopularityWindow.HOUR,
            PopularityWindow.DAY,
            PopularityWindow.ALL_TIME
        ],
        update_interval=1  # Fast updates for demo
    )
    
    cache = MoviePopularityCache(config)
    
    # Add movie metadata
    movies = {
        1: {'title': 'Action Hero', 'genres': 'Action|Adventure'},
        2: {'title': 'Romantic Story', 'genres': 'Romance|Drama'},
        3: {'title': 'Comedy Night', 'genres': 'Comedy'},
        4: {'title': 'Thriller Movie', 'genres': 'Thriller|Action'},
        5: {'title': 'Drama Film', 'genres': 'Drama'},
    }
    
    for movie_id, metadata in movies.items():
        cache.set_movie_metadata(movie_id, metadata)
    
    # Simulate interactions
    print("\n1. Simulating interactions...")
    now = time.time()
    
    # Movie 1 very popular in last hour
    for _ in range(50):
        cache.record_rating(1, random.uniform(4, 5), timestamp=now - random.uniform(0, 3600))
    
    # Movie 2 moderately popular
    for _ in range(30):
        cache.record_rating(2, random.uniform(3, 5), timestamp=now - random.uniform(0, 86400))
    
    # Movie 3 popular yesterday
    for _ in range(100):
        cache.record_rating(3, random.uniform(3, 4), timestamp=now - random.uniform(43200, 86400))
    
    # Movie 4 steady popularity
    for _ in range(20):
        cache.record_rating(4, random.uniform(3, 5), timestamp=now - random.uniform(0, 86400))
    
    # Movie 5 low popularity
    for _ in range(5):
        cache.record_rating(5, random.uniform(2, 4), timestamp=now - random.uniform(0, 86400))
    
    # Force update
    cache._update_all()
    
    print("\n2. Popular Movies by Window")
    print("-" * 30)
    
    for window in config.windows:
        top = cache.get_top_items(window, n=3)
        print(f"\n{window.value}:")
        for item in top:
            movie = movies.get(item.item, {})
            print(f"  #{cache.get_item_rank(item.item, window)}: "
                  f"{movie.get('title', 'Unknown')} "
                  f"(score: {item.score:.2f}, count: {item.count})")
    
    print("\n\n3. Popular by Genre (Action)")
    print("-" * 30)
    
    action_popular = cache.get_popular_by_genre('Action', n=3)
    for item in action_popular:
        movie = movies.get(item.item, {})
        print(f"  {movie.get('title')}: {item.score:.2f}")
    
    print("\n\n4. Trending Movies")
    print("-" * 30)
    
    trending = cache.get_trending(
        short_window=PopularityWindow.HOUR,
        long_window=PopularityWindow.DAY,
        n=3
    )
    
    for t in trending:
        movie = movies.get(t['movie_id'], {})
        print(f"  {movie.get('title')}: trend={t['trend_score']:.2f}x")
    
    print("\n\n5. Cache Stats")
    print("-" * 30)
    print(cache.get_stats())
