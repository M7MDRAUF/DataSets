"""
CineMatch V2.1.6 - Pagination Utilities

Pagination support for large result sets.
Reduces memory usage and improves UI responsiveness.

Author: CineMatch Development Team
Date: December 5, 2025

Features:
    - DataFrame pagination
    - Cursor-based pagination for streaming
    - Page metadata (total, has_next, etc.)
    - Streamlit integration helpers
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Iterator, List, Optional, TypeVar
import pandas as pd

T = TypeVar('T')


@dataclass
class Page(Generic[T]):
    """A single page of results."""
    
    items: List[T]
    page_number: int
    page_size: int
    total_items: int
    total_pages: int
    
    @property
    def has_next(self) -> bool:
        return self.page_number < self.total_pages
    
    @property
    def has_previous(self) -> bool:
        return self.page_number > 1
    
    @property
    def next_page(self) -> Optional[int]:
        return self.page_number + 1 if self.has_next else None
    
    @property
    def previous_page(self) -> Optional[int]:
        return self.page_number - 1 if self.has_previous else None
    
    @property
    def start_index(self) -> int:
        """1-based start index of items on this page."""
        return (self.page_number - 1) * self.page_size + 1
    
    @property
    def end_index(self) -> int:
        """1-based end index of items on this page."""
        return min(self.page_number * self.page_size, self.total_items)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'items': self.items,
            'pagination': {
                'page': self.page_number,
                'page_size': self.page_size,
                'total_items': self.total_items,
                'total_pages': self.total_pages,
                'has_next': self.has_next,
                'has_previous': self.has_previous
            }
        }


@dataclass
class DataFramePage:
    """A paginated DataFrame result."""
    
    data: pd.DataFrame
    page_number: int
    page_size: int
    total_rows: int
    total_pages: int
    
    @property
    def has_next(self) -> bool:
        return self.page_number < self.total_pages
    
    @property
    def has_previous(self) -> bool:
        return self.page_number > 1
    
    @property
    def start_row(self) -> int:
        """1-based start row index."""
        return (self.page_number - 1) * self.page_size + 1
    
    @property
    def end_row(self) -> int:
        """1-based end row index."""
        return min(self.page_number * self.page_size, self.total_rows)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'data': self.data.to_dict('records'),
            'pagination': {
                'page': self.page_number,
                'page_size': self.page_size,
                'total_rows': self.total_rows,
                'total_pages': self.total_pages,
                'has_next': self.has_next,
                'has_previous': self.has_previous,
                'showing': f"{self.start_row}-{self.end_row} of {self.total_rows}"
            }
        }


class Paginator:
    """
    Generic paginator for lists and iterables.
    
    Usage:
        items = list(range(100))
        paginator = Paginator(items, page_size=10)
        
        page1 = paginator.get_page(1)
        page2 = paginator.get_page(2)
    """
    
    def __init__(self, items: List[T], page_size: int = 10):
        """
        Initialize paginator.
        
        Args:
            items: List of items to paginate
            page_size: Items per page
        """
        self.items = items
        self.page_size = page_size
        self.total_items = len(items)
        self.total_pages = max(1, math.ceil(self.total_items / page_size))
    
    def get_page(self, page_number: int) -> Page[T]:
        """
        Get a specific page.
        
        Args:
            page_number: 1-based page number
            
        Returns:
            Page with items
        """
        # Clamp page number
        page_number = max(1, min(page_number, self.total_pages))
        
        start_idx = (page_number - 1) * self.page_size
        end_idx = start_idx + self.page_size
        
        return Page(
            items=self.items[start_idx:end_idx],
            page_number=page_number,
            page_size=self.page_size,
            total_items=self.total_items,
            total_pages=self.total_pages
        )
    
    def __iter__(self) -> Iterator[Page[T]]:
        """Iterate through all pages."""
        for page_num in range(1, self.total_pages + 1):
            yield self.get_page(page_num)


class DataFramePaginator:
    """
    Paginator specifically for DataFrames.
    
    Usage:
        paginator = DataFramePaginator(df, page_size=20)
        page = paginator.get_page(1)
        print(page.data)  # DataFrame slice
    """
    
    def __init__(self, df: pd.DataFrame, page_size: int = 20):
        """
        Initialize DataFrame paginator.
        
        Args:
            df: DataFrame to paginate
            page_size: Rows per page
        """
        self.df = df
        self.page_size = page_size
        self.total_rows = len(df)
        self.total_pages = max(1, math.ceil(self.total_rows / page_size))
    
    def get_page(self, page_number: int) -> DataFramePage:
        """
        Get a specific page of the DataFrame.
        
        Args:
            page_number: 1-based page number
            
        Returns:
            DataFramePage with sliced DataFrame
        """
        # Clamp page number
        page_number = max(1, min(page_number, self.total_pages))
        
        start_idx = (page_number - 1) * self.page_size
        end_idx = start_idx + self.page_size
        
        return DataFramePage(
            data=self.df.iloc[start_idx:end_idx].copy(),
            page_number=page_number,
            page_size=self.page_size,
            total_rows=self.total_rows,
            total_pages=self.total_pages
        )
    
    def __iter__(self) -> Iterator[DataFramePage]:
        """Iterate through all pages."""
        for page_num in range(1, self.total_pages + 1):
            yield self.get_page(page_num)


def paginate_dataframe(
    df: pd.DataFrame,
    page: int = 1,
    page_size: int = 20
) -> DataFramePage:
    """
    Convenience function to paginate a DataFrame.
    
    Args:
        df: DataFrame to paginate
        page: Page number (1-based)
        page_size: Rows per page
        
    Returns:
        DataFramePage
    """
    paginator = DataFramePaginator(df, page_size)
    return paginator.get_page(page)


def paginate_recommendations(
    recommendations: pd.DataFrame,
    page: int = 1,
    page_size: int = 10
) -> DataFramePage:
    """
    Paginate recommendation results.
    
    Args:
        recommendations: Recommendations DataFrame
        page: Page number (1-based)
        page_size: Recommendations per page
        
    Returns:
        DataFramePage with recommendation slice
    """
    return paginate_dataframe(recommendations, page, page_size)


# =============================================================================
# STREAMLIT INTEGRATION
# =============================================================================

def streamlit_paginator(
    df: pd.DataFrame,
    key: str = "paginator",
    page_size: int = 20,
    show_page_info: bool = True
) -> pd.DataFrame:
    """
    Streamlit-integrated paginator with navigation controls.
    
    Usage in Streamlit:
        displayed_df = streamlit_paginator(full_df, key="my_table")
        st.dataframe(displayed_df)
    
    Args:
        df: Full DataFrame to paginate
        key: Unique key for Streamlit state
        page_size: Rows per page
        show_page_info: Show "Showing X-Y of Z" info
        
    Returns:
        DataFrame slice for current page
    """
    try:
        import streamlit as st
    except ImportError:
        # Return first page if Streamlit not available
        return df.head(page_size)
    
    paginator = DataFramePaginator(df, page_size)
    
    # Get current page from session state
    page_key = f"{key}_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    
    current_page = st.session_state[page_key]
    page_result = paginator.get_page(current_page)
    
    # Navigation controls
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        if st.button("⏮️ First", key=f"{key}_first", disabled=not page_result.has_previous):
            st.session_state[page_key] = 1
            st.rerun()
    
    with col2:
        if st.button("◀️ Prev", key=f"{key}_prev", disabled=not page_result.has_previous):
            st.session_state[page_key] = current_page - 1
            st.rerun()
    
    with col3:
        if show_page_info:
            st.markdown(
                f"<div style='text-align: center; padding-top: 5px;'>"
                f"Page {page_result.page_number} of {page_result.total_pages} "
                f"({page_result.start_row}-{page_result.end_row} of {page_result.total_rows})"
                f"</div>",
                unsafe_allow_html=True
            )
    
    with col4:
        if st.button("Next ▶️", key=f"{key}_next", disabled=not page_result.has_next):
            st.session_state[page_key] = current_page + 1
            st.rerun()
    
    with col5:
        if st.button("Last ⏭️", key=f"{key}_last", disabled=not page_result.has_next):
            st.session_state[page_key] = page_result.total_pages
            st.rerun()
    
    return page_result.data


def get_page_selector(
    total_pages: int,
    current_page: int,
    key: str = "page_select"
) -> int:
    """
    Create a page selector dropdown for Streamlit.
    
    Args:
        total_pages: Total number of pages
        current_page: Current page number
        key: Streamlit widget key
        
    Returns:
        Selected page number
    """
    try:
        import streamlit as st
        
        options = list(range(1, total_pages + 1))
        selected = st.selectbox(
            "Page",
            options=options,
            index=current_page - 1,
            key=key
        )
        return selected
    except ImportError:
        return current_page


# =============================================================================
# CLI & TESTING
# =============================================================================

if __name__ == '__main__':
    print("Pagination Demo")
    print("=" * 40)
    
    # Create sample data
    import numpy as np
    
    df = pd.DataFrame({
        'movieId': range(1, 101),
        'title': [f'Movie {i}' for i in range(1, 101)],
        'rating': np.random.uniform(1, 5, 100).round(1)
    })
    
    print(f"\nTotal rows: {len(df)}")
    
    # Paginate
    paginator = DataFramePaginator(df, page_size=10)
    
    print(f"Total pages: {paginator.total_pages}")
    
    # Get first page
    page1 = paginator.get_page(1)
    print(f"\nPage 1:")
    print(f"  Rows: {page1.start_row}-{page1.end_row}")
    print(f"  Has next: {page1.has_next}")
    print(page1.data.head())
    
    # Get last page
    page10 = paginator.get_page(10)
    print(f"\nPage 10:")
    print(f"  Rows: {page10.start_row}-{page10.end_row}")
    print(f"  Has next: {page10.has_next}")
    print(page10.data.head())
    
    # Iterate all pages
    print("\n\nIterating all pages:")
    for page in paginator:
        print(f"  Page {page.page_number}: rows {page.start_row}-{page.end_row}")
