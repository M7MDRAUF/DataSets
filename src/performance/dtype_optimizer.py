"""
Data Type Optimizer for CineMatch V2.1.6

Optimizes data types for memory and performance:
- NumPy dtype selection
- Sparse array conversion
- Categorical encoding
- Type inference

Phase 2 - Task 2.6: Data Type Optimization
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DTypeStrategy(Enum):
    """Strategy for dtype selection."""
    MEMORY = "memory"       # Minimize memory
    PERFORMANCE = "performance"  # Maximize speed
    BALANCED = "balanced"   # Balance both


@dataclass
class DTypeRecommendation:
    """Recommendation for a dtype change."""
    column: str
    current_dtype: str
    recommended_dtype: str
    memory_savings_bytes: int
    reason: str
    
    @property
    def memory_savings_percent(self) -> float:
        # Estimate based on dtype sizes
        return 0.0  # Would need current memory to calculate


@dataclass
class DTypeReport:
    """Report of dtype optimization."""
    total_columns: int
    optimized_columns: int
    original_memory_mb: float
    optimized_memory_mb: float
    recommendations: List[DTypeRecommendation]
    
    @property
    def reduction_percent(self) -> float:
        if self.original_memory_mb == 0:
            return 0.0
        return (self.original_memory_mb - self.optimized_memory_mb) / self.original_memory_mb * 100


class DTypeOptimizer:
    """
    Optimizes data types for DataFrames and arrays.
    
    Features:
    - Automatic dtype inference
    - Integer downcasting
    - Float precision selection
    - Category conversion
    - Sparse array support
    """
    
    # Dtype hierarchy for integers
    INT_DTYPES = [
        (np.int8, -128, 127),
        (np.int16, -32768, 32767),
        (np.int32, -2147483648, 2147483647),
        (np.int64, np.iinfo(np.int64).min, np.iinfo(np.int64).max)
    ]
    
    UINT_DTYPES = [
        (np.uint8, 0, 255),
        (np.uint16, 0, 65535),
        (np.uint32, 0, 4294967295),
        (np.uint64, 0, np.iinfo(np.uint64).max)
    ]
    
    def __init__(
        self,
        strategy: DTypeStrategy = DTypeStrategy.BALANCED,
        category_threshold: float = 0.5
    ):
        self.strategy = strategy
        self.category_threshold = category_threshold
    
    def optimize_array(self, arr: np.ndarray) -> np.ndarray:
        """Optimize a numpy array's dtype."""
        if np.issubdtype(arr.dtype, np.integer):
            return self._optimize_int_array(arr)
        elif np.issubdtype(arr.dtype, np.floating):
            return self._optimize_float_array(arr)
        return arr
    
    def _optimize_int_array(self, arr: np.ndarray) -> np.ndarray:
        """Downcast integer array."""
        min_val = arr.min()
        max_val = arr.max()
        
        # Try unsigned if no negatives
        if min_val >= 0:
            for dtype, d_min, d_max in self.UINT_DTYPES:
                if min_val >= d_min and max_val <= d_max:
                    return arr.astype(dtype)
        else:
            for dtype, d_min, d_max in self.INT_DTYPES:
                if min_val >= d_min and max_val <= d_max:
                    return arr.astype(dtype)
        
        return arr
    
    def _optimize_float_array(self, arr: np.ndarray) -> np.ndarray:
        """Downcast float array."""
        if self.strategy == DTypeStrategy.PERFORMANCE:
            # Keep float64 for performance
            return arr
        
        # Check if float32 is sufficient
        if arr.dtype == np.float64:
            arr32 = arr.astype(np.float32)
            # Check precision loss
            if np.allclose(arr, arr32, rtol=1e-5, atol=1e-8):
                return arr32
        
        return arr
    
    def optimize_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, DTypeReport]:
        """Optimize all columns in a DataFrame."""
        df = df.copy()
        original_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        recommendations: List[DTypeRecommendation] = []
        optimized_count = 0
        
        for col in df.columns:
            current_dtype = str(df[col].dtype)
            recommendation = self._optimize_column(df, col)
            
            if recommendation:
                recommendations.append(recommendation)
                optimized_count += 1
        
        optimized_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        report = DTypeReport(
            total_columns=len(df.columns),
            optimized_columns=optimized_count,
            original_memory_mb=original_memory,
            optimized_memory_mb=optimized_memory,
            recommendations=recommendations
        )
        
        logger.info(
            f"DType optimization: {report.original_memory_mb:.2f}MB -> "
            f"{report.optimized_memory_mb:.2f}MB ({report.reduction_percent:.1f}% reduction)"
        )
        
        return df, report
    
    def _optimize_column(
        self,
        df: pd.DataFrame,
        col: str
    ) -> Optional[DTypeRecommendation]:
        """Optimize a single column."""
        dtype = df[col].dtype
        current_dtype = str(dtype)
        original_memory = df[col].memory_usage(deep=True)
        
        # Object (string) columns
        if dtype == 'object':
            unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 1.0
            
            if unique_ratio < self.category_threshold:
                df[col] = df[col].astype('category')
                new_memory = df[col].memory_usage(deep=True)
                
                return DTypeRecommendation(
                    column=col,
                    current_dtype=current_dtype,
                    recommended_dtype='category',
                    memory_savings_bytes=original_memory - new_memory,
                    reason=f"Low cardinality ({unique_ratio:.1%} unique values)"
                )
        
        # Integer columns
        elif np.issubdtype(dtype, np.integer):
            optimized = self._optimize_int_series(df[col])
            new_dtype = str(optimized.dtype)
            
            if new_dtype != current_dtype:
                df[col] = optimized
                new_memory = df[col].memory_usage(deep=True)
                
                return DTypeRecommendation(
                    column=col,
                    current_dtype=current_dtype,
                    recommended_dtype=new_dtype,
                    memory_savings_bytes=original_memory - new_memory,
                    reason=f"Range fits in smaller dtype"
                )
        
        # Float columns
        elif np.issubdtype(dtype, np.floating):
            if dtype == np.float64 and self.strategy != DTypeStrategy.PERFORMANCE:
                # Try float32
                try:
                    optimized = df[col].astype(np.float32)
                    # Check precision
                    if np.allclose(df[col].dropna(), optimized.dropna(), rtol=1e-5):
                        df[col] = optimized
                        new_memory = df[col].memory_usage(deep=True)
                        
                        return DTypeRecommendation(
                            column=col,
                            current_dtype=current_dtype,
                            recommended_dtype='float32',
                            memory_savings_bytes=original_memory - new_memory,
                            reason="float32 precision sufficient"
                        )
                except (ValueError, TypeError):
                    pass
        
        return None
    
    def _optimize_int_series(self, series: pd.Series) -> pd.Series:
        """Optimize integer series."""
        if series.isna().any():
            # Use nullable integers
            min_val = series.min()
            max_val = series.max()
            
            if min_val >= 0:
                if max_val <= 255:
                    return series.astype('UInt8')
                elif max_val <= 65535:
                    return series.astype('UInt16')
                elif max_val <= 4294967295:
                    return series.astype('UInt32')
            else:
                if min_val >= -128 and max_val <= 127:
                    return series.astype('Int8')
                elif min_val >= -32768 and max_val <= 32767:
                    return series.astype('Int16')
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    return series.astype('Int32')
            return series
        
        # Non-nullable
        return pd.to_numeric(series, downcast='integer')
    
    def get_dtype_info(self, dtype: Union[str, Type]) -> Dict[str, Any]:
        """Get information about a dtype."""
        dtype = np.dtype(dtype)
        
        info = {
            'name': str(dtype),
            'kind': dtype.kind,
            'itemsize': dtype.itemsize,
            'is_numeric': np.issubdtype(dtype, np.number),
            'is_integer': np.issubdtype(dtype, np.integer),
            'is_float': np.issubdtype(dtype, np.floating),
            'is_bool': np.issubdtype(dtype, np.bool_),
        }
        
        if np.issubdtype(dtype, np.integer):
            info['min'] = np.iinfo(dtype).min
            info['max'] = np.iinfo(dtype).max
        elif np.issubdtype(dtype, np.floating):
            info['min'] = np.finfo(dtype).min
            info['max'] = np.finfo(dtype).max
            info['precision'] = np.finfo(dtype).precision
        
        return info
    
    def recommend_dtype(
        self,
        values: Union[list, np.ndarray, pd.Series],
        allow_null: bool = False
    ) -> str:
        """Recommend optimal dtype for values."""
        if isinstance(values, (list, pd.Series)):
            values = np.array(values)
        
        # Check for strings
        if values.dtype == object:
            return 'object'
        
        # Check for booleans
        unique = np.unique(values[~pd.isna(values)])
        if len(unique) <= 2 and all(v in [0, 1, True, False] for v in unique):
            return 'bool' if not allow_null else 'boolean'
        
        # Numeric types
        if np.issubdtype(values.dtype, np.number):
            has_null = pd.isna(values).any()
            has_float = np.any(values != np.floor(values))
            
            if has_float:
                # Float
                if self.strategy == DTypeStrategy.MEMORY:
                    return 'float32'
                return 'float64'
            else:
                # Integer
                min_val = np.nanmin(values)
                max_val = np.nanmax(values)
                
                if min_val >= 0:
                    dtypes = self.UINT_DTYPES
                else:
                    dtypes = self.INT_DTYPES
                
                for dtype, d_min, d_max in dtypes:
                    if min_val >= d_min and max_val <= d_max:
                        if has_null:
                            # Use nullable
                            name = dtype.__name__
                            return name[0].upper() + name[1:]
                        return str(dtype.__name__)
        
        return str(values.dtype)


# Convenience functions
_optimizer: Optional[DTypeOptimizer] = None


def get_optimizer(strategy: DTypeStrategy = DTypeStrategy.BALANCED) -> DTypeOptimizer:
    global _optimizer
    if _optimizer is None:
        _optimizer = DTypeOptimizer(strategy=strategy)
    return _optimizer


def optimize_dtypes(df: pd.DataFrame) -> tuple[pd.DataFrame, DTypeReport]:
    """Optimize DataFrame dtypes."""
    return get_optimizer().optimize_dataframe(df)


def recommend_dtype(values: Any, allow_null: bool = False) -> str:
    """Recommend dtype for values."""
    return get_optimizer().recommend_dtype(values, allow_null)


def downcast_array(arr: np.ndarray) -> np.ndarray:
    """Downcast numpy array."""
    return get_optimizer().optimize_array(arr)


if __name__ == "__main__":
    print("Data Type Optimizer Demo")
    print("=" * 50)
    
    # Create sample DataFrame
    np.random.seed(42)
    df = pd.DataFrame({
        'user_id': np.random.randint(1, 1000, 10000).astype(np.int64),
        'movie_id': np.random.randint(1, 5000, 10000).astype(np.int64),
        'rating': np.random.uniform(1, 5, 10000),
        'count': np.random.randint(0, 100, 10000).astype(np.int64),
        'genre': np.random.choice(['Action', 'Drama', 'Comedy', 'Horror'], 10000),
        'flag': np.random.choice([True, False], 10000),
    })
    
    print("\nOriginal DataFrame:")
    print(df.dtypes)
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Optimize
    optimizer = DTypeOptimizer(strategy=DTypeStrategy.MEMORY)
    optimized, report = optimizer.optimize_dataframe(df)
    
    print("\n\nOptimized DataFrame:")
    print(optimized.dtypes)
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    print(f"\n\nReport:")
    print(f"  Total columns: {report.total_columns}")
    print(f"  Optimized: {report.optimized_columns}")
    print(f"  Memory: {report.original_memory_mb:.2f}MB -> {report.optimized_memory_mb:.2f}MB")
    print(f"  Reduction: {report.reduction_percent:.1f}%")
    
    print("\n\nRecommendations:")
    for rec in report.recommendations:
        print(f"  {rec.column}: {rec.current_dtype} -> {rec.recommended_dtype}")
        print(f"    Reason: {rec.reason}")
        print(f"    Savings: {rec.memory_savings_bytes / 1024:.1f} KB")
    
    # Demo array optimization
    print("\n\n\nArray Optimization Demo")
    print("-" * 30)
    
    arr = np.array([1, 2, 3, 100, 200], dtype=np.int64)
    print(f"Original: dtype={arr.dtype}, nbytes={arr.nbytes}")
    
    optimized_arr = optimizer.optimize_array(arr)
    print(f"Optimized: dtype={optimized_arr.dtype}, nbytes={optimized_arr.nbytes}")
    
    # Demo dtype recommendation
    print("\n\nDType Recommendation Demo")
    print("-" * 30)
    
    values_list = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, None, 5],
        [1.0, 2.5, 3.7],
        ['a', 'b', 'c'],
        [True, False, True]
    ]
    
    for values in values_list:
        recommended = optimizer.recommend_dtype(values, allow_null=True)
        print(f"  {values[:3]}... -> {recommended}")
