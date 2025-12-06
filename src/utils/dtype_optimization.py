"""
CineMatch V2.1.6 - Data Type Optimization

Utilities for reducing memory usage through efficient data types.
Converts float64 to float32, uses categorical for strings, etc.

Author: CineMatch Development Team
Date: December 5, 2025

Typical Memory Savings:
    - float64 → float32: 50% reduction
    - int64 → int32: 50% reduction  
    - object → category: 90%+ reduction for low cardinality columns
"""

import logging
from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def optimize_dtypes(
    df: pd.DataFrame,
    float_precision: str = 'float32',
    int_precision: str = 'int32',
    categorical_threshold: float = 0.5,
    inplace: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Optimize DataFrame dtypes to reduce memory usage.
    
    Args:
        df: DataFrame to optimize
        float_precision: Target float type ('float32' or 'float16')
        int_precision: Target int type ('int32' or 'int16')
        categorical_threshold: Max unique/total ratio for categorical conversion
        inplace: Modify DataFrame in place
        verbose: Print optimization details
        
    Returns:
        Optimized DataFrame
        
    Example:
        df = optimize_dtypes(ratings_df, verbose=True)
        # Reduced memory from 1.2GB to 600MB
    """
    if not inplace:
        df = df.copy()
    
    initial_mem = df.memory_usage(deep=True).sum()
    optimizations = []
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Float optimization
        if col_type == 'float64':
            if float_precision == 'float32':
                df[col] = df[col].astype(np.float32)
                optimizations.append((col, 'float64 → float32'))
            elif float_precision == 'float16':
                # Check range for float16 safety
                col_min = df[col].min()
                col_max = df[col].max()
                if col_min >= -65504 and col_max <= 65504:
                    df[col] = df[col].astype(np.float16)
                    optimizations.append((col, 'float64 → float16'))
                else:
                    df[col] = df[col].astype(np.float32)
                    optimizations.append((col, 'float64 → float32 (range too large for float16)'))
        
        # Integer optimization
        elif col_type == 'int64':
            col_min = df[col].min()
            col_max = df[col].max()
            
            # Try to use smallest possible int type
            if col_min >= 0:
                if col_max <= 255:
                    df[col] = df[col].astype(np.uint8)
                    optimizations.append((col, 'int64 → uint8'))
                elif col_max <= 65535:
                    df[col] = df[col].astype(np.uint16)
                    optimizations.append((col, 'int64 → uint16'))
                elif col_max <= 4294967295:
                    df[col] = df[col].astype(np.uint32)
                    optimizations.append((col, 'int64 → uint32'))
            else:
                if col_min >= -128 and col_max <= 127:
                    df[col] = df[col].astype(np.int8)
                    optimizations.append((col, 'int64 → int8'))
                elif col_min >= -32768 and col_max <= 32767:
                    df[col] = df[col].astype(np.int16)
                    optimizations.append((col, 'int64 → int16'))
                elif int_precision == 'int32' and col_min >= -2147483648 and col_max <= 2147483647:
                    df[col] = df[col].astype(np.int32)
                    optimizations.append((col, 'int64 → int32'))
        
        # Object/string to categorical
        elif col_type == 'object':
            n_unique = df[col].nunique()
            n_total = len(df[col])
            
            if n_unique / n_total < categorical_threshold:
                df[col] = df[col].astype('category')
                optimizations.append((col, f'object → category ({n_unique} unique)'))
    
    final_mem = df.memory_usage(deep=True).sum()
    reduction = (1 - final_mem / initial_mem) * 100
    
    if verbose:
        print(f"\n📊 Memory Optimization Results:")
        print(f"  Initial: {initial_mem / 1e6:.1f} MB")
        print(f"  Final:   {final_mem / 1e6:.1f} MB")
        print(f"  Reduction: {reduction:.1f}%")
        print(f"\n  Optimizations applied:")
        for col, change in optimizations:
            print(f"    • {col}: {change}")
    
    return df


def get_memory_usage(df: pd.DataFrame) -> Dict[str, float]:
    """
    Get detailed memory usage breakdown by column.
    
    Returns:
        Dict mapping column name to memory usage in MB
    """
    mem_usage = df.memory_usage(deep=True)
    
    result = {
        'total_mb': mem_usage.sum() / 1e6,
        'index_mb': mem_usage['Index'] / 1e6 if 'Index' in mem_usage else 0,
        'columns': {}
    }
    
    for col in df.columns:
        result['columns'][col] = {
            'mb': mem_usage[col] / 1e6,
            'dtype': str(df[col].dtype)
        }
    
    return result


def optimize_ratings_df(ratings_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Optimize ratings DataFrame specifically.
    
    MovieLens ratings have specific ranges:
    - userId: positive integers (uint32 sufficient for 32M users)
    - movieId: positive integers (uint32 sufficient)
    - rating: 0.5-5.0 range (float32 sufficient)
    - timestamp: Unix timestamp (uint32 valid until 2106)
    """
    df = ratings_df.copy()
    initial_mem = df.memory_usage(deep=True).sum()
    
    # Optimize specific columns
    if 'userId' in df.columns:
        df['userId'] = df['userId'].astype(np.uint32)
    
    if 'movieId' in df.columns:
        df['movieId'] = df['movieId'].astype(np.uint32)
    
    if 'rating' in df.columns:
        df['rating'] = df['rating'].astype(np.float32)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = df['timestamp'].astype(np.uint32)
    
    final_mem = df.memory_usage(deep=True).sum()
    
    if verbose:
        reduction = (1 - final_mem / initial_mem) * 100
        print(f"📊 Ratings DataFrame Optimization:")
        print(f"  {initial_mem / 1e6:.1f} MB → {final_mem / 1e6:.1f} MB ({reduction:.1f}% reduction)")
    
    return df


def optimize_movies_df(movies_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Optimize movies DataFrame specifically.
    
    MovieLens movies:
    - movieId: positive integer
    - title: string (keep as-is for search functionality)
    - genres: pipe-separated string → category
    """
    df = movies_df.copy()
    initial_mem = df.memory_usage(deep=True).sum()
    
    # Optimize movieId
    if 'movieId' in df.columns:
        df['movieId'] = df['movieId'].astype(np.uint32)
    
    # Genres has limited unique values - use category
    if 'genres' in df.columns:
        # Keep as string for search, but note that category would save memory
        # if search compatibility isn't needed
        pass
    
    # Poster paths can be categorical if there are many duplicates
    if 'poster_path' in df.columns:
        n_unique = df['poster_path'].nunique()
        if n_unique < len(df) * 0.9:  # Less than 90% unique
            df['poster_path'] = df['poster_path'].astype('category')
    
    final_mem = df.memory_usage(deep=True).sum()
    
    if verbose:
        reduction = (1 - final_mem / initial_mem) * 100
        print(f"📊 Movies DataFrame Optimization:")
        print(f"  {initial_mem / 1e6:.1f} MB → {final_mem / 1e6:.1f} MB ({reduction:.1f}% reduction)")
    
    return df


def estimate_optimal_batch_size(
    data_size_mb: float,
    available_memory_mb: float = 1000,
    safety_factor: float = 0.5
) -> int:
    """
    Estimate optimal batch size for memory-constrained operations.
    
    Args:
        data_size_mb: Size of full dataset in MB
        available_memory_mb: Available memory for processing
        safety_factor: Fraction of memory to use (default 50%)
        
    Returns:
        Recommended batch size as fraction of total data
    """
    usable_memory = available_memory_mb * safety_factor
    
    if data_size_mb <= usable_memory:
        return 1.0  # Can process all at once
    
    return usable_memory / data_size_mb


# =============================================================================
# SPARSE MATRIX OPTIMIZATION
# =============================================================================

def optimize_sparse_matrix(matrix, format: str = 'csr') -> Tuple:
    """
    Optimize sparse matrix format and dtype.
    
    Args:
        matrix: scipy sparse matrix
        format: Target format ('csr' or 'csc')
        
    Returns:
        Tuple of (optimized_matrix, original_memory, optimized_memory)
    """
    try:
        from scipy import sparse
    except ImportError:
        logger.warning("scipy not available for sparse optimization")
        return matrix, 0, 0
    
    # Get original memory
    if hasattr(matrix, 'data'):
        original_mem = (
            matrix.data.nbytes + 
            matrix.indices.nbytes + 
            matrix.indptr.nbytes
        )
    else:
        original_mem = 0
    
    # Convert to efficient format
    if format == 'csr' and not sparse.isspmatrix_csr(matrix):
        matrix = matrix.tocsr()
    elif format == 'csc' and not sparse.isspmatrix_csc(matrix):
        matrix = matrix.tocsc()
    
    # Optimize data dtype
    if matrix.data.dtype == np.float64:
        matrix.data = matrix.data.astype(np.float32)
    
    # Optimize index dtype if possible
    max_dim = max(matrix.shape)
    if max_dim < 2**31 and matrix.indices.dtype == np.int64:
        matrix.indices = matrix.indices.astype(np.int32)
        matrix.indptr = matrix.indptr.astype(np.int32)
    
    # Eliminate zeros
    matrix.eliminate_zeros()
    
    # Get optimized memory
    optimized_mem = (
        matrix.data.nbytes + 
        matrix.indices.nbytes + 
        matrix.indptr.nbytes
    )
    
    return matrix, original_mem, optimized_mem


# =============================================================================
# CLI & TESTING
# =============================================================================

if __name__ == '__main__':
    print("Data Type Optimization Demo")
    print("=" * 40)
    
    # Create sample DataFrame
    n_rows = 100000
    sample_df = pd.DataFrame({
        'userId': np.random.randint(1, 100000, n_rows),
        'movieId': np.random.randint(1, 50000, n_rows),
        'rating': np.random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], n_rows),
        'timestamp': np.random.randint(1000000000, 1700000000, n_rows),
        'genre': np.random.choice(['Action', 'Comedy', 'Drama', 'Horror', 'Romance'], n_rows)
    })
    
    print(f"\nOriginal DataFrame:")
    print(f"  Shape: {sample_df.shape}")
    print(f"  Dtypes: {dict(sample_df.dtypes)}")
    
    # Optimize
    optimized = optimize_dtypes(sample_df, verbose=True)
    
    print(f"\nOptimized Dtypes: {dict(optimized.dtypes)}")
