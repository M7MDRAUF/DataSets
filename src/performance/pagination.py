"""
Pagination Module for CineMatch V2.1.6

Implements pagination for large result sets:
- Offset-based pagination
- Cursor-based pagination
- Page metadata
- Performance optimization

Phase 2 - Task 2.7: Pagination
"""

import logging
import base64
import json
import hashlib
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')


class PaginationType(Enum):
    """Type of pagination."""
    OFFSET = "offset"     # Traditional offset-based
    CURSOR = "cursor"     # Cursor-based (better for real-time data)
    KEYSET = "keyset"     # Keyset pagination


@dataclass
class PageRequest:
    """Request for a page of results."""
    page: int = 1
    page_size: int = 20
    cursor: Optional[str] = None
    sort_by: Optional[str] = None
    sort_order: str = "asc"  # asc or desc
    
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size
    
    def validate(self) -> None:
        """Validate page request."""
        if self.page < 1:
            raise ValueError("Page must be >= 1")
        if self.page_size < 1:
            raise ValueError("Page size must be >= 1")
        if self.page_size > 1000:
            raise ValueError("Page size must be <= 1000")
        if self.sort_order not in ("asc", "desc"):
            raise ValueError("Sort order must be 'asc' or 'desc'")


@dataclass
class PageInfo:
    """Information about pagination state."""
    total_items: int
    total_pages: int
    current_page: int
    page_size: int
    has_next: bool
    has_previous: bool
    next_cursor: Optional[str] = None
    previous_cursor: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_items': self.total_items,
            'total_pages': self.total_pages,
            'current_page': self.current_page,
            'page_size': self.page_size,
            'has_next': self.has_next,
            'has_previous': self.has_previous,
            'next_cursor': self.next_cursor,
            'previous_cursor': self.previous_cursor
        }


@dataclass
class Page(Generic[T]):
    """A page of results."""
    items: List[T]
    page_info: PageInfo
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __iter__(self):
        return iter(self.items)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'items': self.items,
            'page_info': self.page_info.to_dict()
        }


class Paginator(Generic[T]):
    """
    Paginator for lists and iterables.
    
    Features:
    - Offset-based pagination
    - Cursor-based pagination
    - Sorting support
    - Metadata generation
    """
    
    def __init__(
        self,
        items: List[T],
        default_page_size: int = 20,
        max_page_size: int = 100
    ):
        self.items = items
        self.default_page_size = default_page_size
        self.max_page_size = max_page_size
        self.total_items = len(items)
    
    def get_page(self, request: Optional[PageRequest] = None) -> Page[T]:
        """Get a page of results."""
        if request is None:
            request = PageRequest(page_size=self.default_page_size)
        
        request.validate()
        
        # Enforce max page size
        page_size = min(request.page_size, self.max_page_size)
        
        # Calculate pagination
        total_pages = max(1, (self.total_items + page_size - 1) // page_size)
        
        # Get slice
        start = request.offset
        end = start + page_size
        page_items = self.items[start:end]
        
        # Build page info
        page_info = PageInfo(
            total_items=self.total_items,
            total_pages=total_pages,
            current_page=request.page,
            page_size=page_size,
            has_next=request.page < total_pages,
            has_previous=request.page > 1,
            next_cursor=self._encode_cursor(end) if request.page < total_pages else None,
            previous_cursor=self._encode_cursor(start - page_size) if request.page > 1 else None
        )
        
        return Page(items=page_items, page_info=page_info)
    
    def get_all_pages(self, page_size: Optional[int] = None) -> List[Page[T]]:
        """Get all pages."""
        page_size = page_size or self.default_page_size
        pages = []
        page = 1
        
        while True:
            result = self.get_page(PageRequest(page=page, page_size=page_size))
            pages.append(result)
            
            if not result.page_info.has_next:
                break
            page += 1
        
        return pages
    
    def _encode_cursor(self, offset: int) -> str:
        """Encode a cursor."""
        data = {'offset': offset}
        json_str = json.dumps(data)
        return base64.b64encode(json_str.encode()).decode()
    
    def _decode_cursor(self, cursor: str) -> Dict[str, Any]:
        """Decode a cursor."""
        try:
            json_str = base64.b64decode(cursor.encode()).decode()
            return json.loads(json_str)
        except Exception:
            return {'offset': 0}


class CursorPaginator(Generic[T]):
    """
    Cursor-based paginator for efficient pagination.
    
    Features:
    - No offset calculations
    - Consistent results with changing data
    - Better performance for large datasets
    """
    
    def __init__(
        self,
        items: List[T],
        id_getter: Callable[[T], Any],
        default_page_size: int = 20
    ):
        self.items = items
        self.id_getter = id_getter
        self.default_page_size = default_page_size
        
        # Build index
        self._index: Dict[Any, int] = {
            id_getter(item): i
            for i, item in enumerate(items)
        }
    
    def get_page(
        self,
        after: Optional[str] = None,
        before: Optional[str] = None,
        first: Optional[int] = None,
        last: Optional[int] = None
    ) -> Page[T]:
        """
        Get a page using cursor-based pagination.
        
        Args:
            after: Cursor to start after
            before: Cursor to end before
            first: Number of items from start
            last: Number of items from end
        """
        page_size = first or last or self.default_page_size
        
        start_idx = 0
        end_idx = len(self.items)
        
        # Handle after cursor
        if after:
            after_data = self._decode_cursor(after)
            after_id = after_data.get('id')
            if after_id in self._index:
                start_idx = self._index[after_id] + 1
        
        # Handle before cursor
        if before:
            before_data = self._decode_cursor(before)
            before_id = before_data.get('id')
            if before_id in self._index:
                end_idx = self._index[before_id]
        
        # Get items
        available = self.items[start_idx:end_idx]
        
        if first:
            page_items = available[:first]
        elif last:
            page_items = available[-last:] if len(available) >= last else available
        else:
            page_items = available[:page_size]
        
        # Build cursors
        next_cursor = None
        prev_cursor = None
        
        if page_items:
            # Next cursor
            last_item = page_items[-1]
            last_id = self.id_getter(last_item)
            if last_id in self._index and self._index[last_id] < len(self.items) - 1:
                next_cursor = self._encode_cursor(last_id)
            
            # Previous cursor
            first_item = page_items[0]
            first_id = self.id_getter(first_item)
            if first_id in self._index and self._index[first_id] > 0:
                prev_cursor = self._encode_cursor(
                    self.id_getter(self.items[self._index[first_id] - 1])
                )
        
        page_info = PageInfo(
            total_items=len(self.items),
            total_pages=-1,  # Not applicable for cursor pagination
            current_page=-1,  # Not applicable
            page_size=page_size,
            has_next=next_cursor is not None,
            has_previous=prev_cursor is not None,
            next_cursor=next_cursor,
            previous_cursor=prev_cursor
        )
        
        return Page(items=page_items, page_info=page_info)
    
    def _encode_cursor(self, item_id: Any) -> str:
        """Encode a cursor from item ID."""
        data = {'id': item_id, 'ts': datetime.utcnow().isoformat()}
        json_str = json.dumps(data, default=str)
        return base64.b64encode(json_str.encode()).decode()
    
    def _decode_cursor(self, cursor: str) -> Dict[str, Any]:
        """Decode a cursor."""
        try:
            json_str = base64.b64decode(cursor.encode()).decode()
            return json.loads(json_str)
        except Exception:
            return {}


class RecommendationPaginator:
    """
    Specialized paginator for movie recommendations.
    
    Features:
    - Score-based sorting
    - Diversity filtering
    - Genre grouping
    """
    
    def __init__(
        self,
        recommendations: List[Dict[str, Any]],
        default_page_size: int = 10
    ):
        self.recommendations = recommendations
        self.default_page_size = default_page_size
    
    def get_page(
        self,
        page: int = 1,
        page_size: Optional[int] = None,
        min_score: float = 0.0,
        genre: Optional[str] = None
    ) -> Page[Dict[str, Any]]:
        """Get a page of recommendations with filters."""
        page_size = page_size or self.default_page_size
        
        # Filter
        filtered = [
            r for r in self.recommendations
            if r.get('score', 0) >= min_score
            and (genre is None or genre.lower() in str(r.get('genres', '')).lower())
        ]
        
        # Use base paginator
        paginator: Paginator[Dict[str, Any]] = Paginator(filtered, default_page_size=page_size)
        return paginator.get_page(PageRequest(page=page, page_size=page_size))
    
    def get_diverse_page(
        self,
        page_size: int = 10,
        diversity_factor: float = 0.3
    ) -> Page[Dict[str, Any]]:
        """
        Get a diverse page of recommendations.
        
        Ensures variety in genres while maintaining quality.
        """
        if not self.recommendations:
            return Page(items=[], page_info=PageInfo(
                total_items=0, total_pages=0, current_page=1,
                page_size=page_size, has_next=False, has_previous=False
            ))
        
        selected: List[Dict[str, Any]] = []
        seen_genres: set = set()
        
        # Sort by score
        sorted_recs = sorted(
            self.recommendations,
            key=lambda x: x.get('score', 0),
            reverse=True
        )
        
        # Select diverse items
        for rec in sorted_recs:
            if len(selected) >= page_size:
                break
            
            genres = rec.get('genres', [])
            if isinstance(genres, str):
                genres = [g.strip() for g in genres.split(',')]
            
            # Check diversity
            new_genres = set(genres) - seen_genres
            diversity_score = len(new_genres) / max(1, len(genres)) if genres else 0
            
            # Add if diverse enough or high score
            if diversity_score >= diversity_factor or rec.get('score', 0) > 0.9:
                selected.append(rec)
                seen_genres.update(genres)
        
        # Fill remaining with top scores
        if len(selected) < page_size:
            for rec in sorted_recs:
                if rec not in selected:
                    selected.append(rec)
                    if len(selected) >= page_size:
                        break
        
        return Page(
            items=selected,
            page_info=PageInfo(
                total_items=len(self.recommendations),
                total_pages=(len(self.recommendations) + page_size - 1) // page_size,
                current_page=1,
                page_size=page_size,
                has_next=len(self.recommendations) > page_size,
                has_previous=False
            )
        )


# Convenience functions
def paginate(
    items: List[T],
    page: int = 1,
    page_size: int = 20
) -> Page[T]:
    """Paginate a list."""
    paginator: Paginator[T] = Paginator(items, default_page_size=page_size)
    return paginator.get_page(PageRequest(page=page, page_size=page_size))


def paginate_recommendations(
    recommendations: List[Dict[str, Any]],
    page: int = 1,
    page_size: int = 10
) -> Page[Dict[str, Any]]:
    """Paginate recommendations."""
    paginator = RecommendationPaginator(recommendations, default_page_size=page_size)
    return paginator.get_page(page=page, page_size=page_size)


if __name__ == "__main__":
    print("Pagination Module Demo")
    print("=" * 50)
    
    # Demo basic pagination
    print("\n1. Basic Pagination")
    print("-" * 30)
    
    items = list(range(1, 101))  # 100 items
    paginator: Paginator[int] = Paginator(items, default_page_size=10)
    
    # Get first page
    page1 = paginator.get_page(PageRequest(page=1, page_size=10))
    print(f"Page 1: {page1.items}")
    print(f"Page Info: {page1.page_info.to_dict()}")
    
    # Get last page
    page10 = paginator.get_page(PageRequest(page=10, page_size=10))
    print(f"\nPage 10: {page10.items}")
    
    # Demo cursor pagination
    print("\n\n2. Cursor Pagination")
    print("-" * 30)
    
    movies = [
        {'id': 1, 'title': 'Movie A'},
        {'id': 2, 'title': 'Movie B'},
        {'id': 3, 'title': 'Movie C'},
        {'id': 4, 'title': 'Movie D'},
        {'id': 5, 'title': 'Movie E'},
    ]
    
    cursor_paginator: CursorPaginator[Dict[str, Any]] = CursorPaginator(
        movies,
        id_getter=lambda x: x['id'],
        default_page_size=2
    )
    
    page1 = cursor_paginator.get_page(first=2)
    print(f"First page: {[m['title'] for m in page1.items]}")
    print(f"Next cursor: {page1.page_info.next_cursor}")
    
    if page1.page_info.next_cursor:
        page2 = cursor_paginator.get_page(after=page1.page_info.next_cursor, first=2)
        print(f"Second page: {[m['title'] for m in page2.items]}")
    
    # Demo recommendation pagination
    print("\n\n3. Recommendation Pagination")
    print("-" * 30)
    
    recommendations = [
        {'title': 'Action Movie', 'score': 0.95, 'genres': 'Action,Thriller'},
        {'title': 'Drama Film', 'score': 0.90, 'genres': 'Drama'},
        {'title': 'Comedy Show', 'score': 0.85, 'genres': 'Comedy'},
        {'title': 'Action Sequel', 'score': 0.80, 'genres': 'Action'},
        {'title': 'Romance Story', 'score': 0.75, 'genres': 'Romance,Drama'},
        {'title': 'Horror Night', 'score': 0.70, 'genres': 'Horror'},
    ]
    
    rec_paginator = RecommendationPaginator(recommendations)
    
    # Regular page
    reg_page = rec_paginator.get_page(page_size=3)
    print(f"Regular page: {[r['title'] for r in reg_page.items]}")
    
    # Diverse page
    div_page = rec_paginator.get_diverse_page(page_size=4, diversity_factor=0.3)
    print(f"Diverse page: {[r['title'] for r in div_page.items]}")
    
    # Filtered page
    filt_page = rec_paginator.get_page(min_score=0.8)
    print(f"Filtered (score >= 0.8): {[r['title'] for r in filt_page.items]}")
