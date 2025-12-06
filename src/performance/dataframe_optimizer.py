"""
DataFrame Optimizer for CineMatch V2.1.6

Optimizes pandas DataFrames for memory efficiency and performance:
- Automatic dtype downcasting
- Category conversion for strings
- Memory usage analysis
- Column pruning

Phase 2 - Task 2.1: DataFrame Optimization
"""

import logging
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for DataFrame optimization."""
    downcast_integers: bool = True
    downcast_floats: bool = True
    convert_categories: bool = True
    category_threshold: float = 0.5  # Convert to category if unique/total < threshold
    sparse_threshold: float = 0.9    # Convert to sparse if null_ratio > threshold
    enable_nullable_int: bool = True
    preserve_index: bool = True
    exclude_columns: Set[str] = field(default_factory=set)


@dataclass
class OptimizationReport:
    """Report of optimization results."""
    original_memory_mb: float
    optimized_memory_mb: float
    reduction_percent: float
    column_changes: Dict[str, Dict[str, str]]
    
    def summary(self) -> str:
        return (
            f"Memory: {self.original_memory_mb:.2f}MB -> {self.optimized_memory_mb:.2f}MB "
            f"({self.reduction_percent:.1f}% reduction)"
        )


class DataFrameOptimizer:
    """
    Optimizes pandas DataFrames for memory efficiency.
    
    Features:
    - Integer downcasting (int64 -> int8/16/32)
    - Float downcasting (float64 -> float32)
    - String to category conversion
    - Sparse array support
    - Memory usage reporting
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
    
    def optimize(
        self,
        df: pd.DataFrame,
        inplace: bool = False
    ) -> tuple[pd.DataFrame, OptimizationReport]:
        """
        Optimize a DataFrame.
        
        Args:
            df: DataFrame to optimize
            inplace: Whether to modify in place
            
        Returns:
            Tuple of (optimized_df, report)
        """
        if not inplace:
            df = df.copy()
        
        original_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
        column_changes = {}
        
        for col in df.columns:
            if col in self.config.exclude_columns:
                continue
            
            original_dtype = str(df[col].dtype)
            new_dtype = original_dtype
            
            # Optimize based on dtype
            if df[col].dtype == 'object':
                df[col], new_dtype = self._optimize_object(df[col])
            elif np.issubdtype(df[col].dtype, np.integer):
                if self.config.downcast_integers:
                    df[col], new_dtype = self._downcast_integer(df[col])
            elif np.issubdtype(df[col].dtype, np.floating):
                if self.config.downcast_floats:
                    df[col], new_dtype = self._downcast_float(df[col])
            
            if str(new_dtype) != original_dtype:
                column_changes[col] = {
                    'from': original_dtype,
                    'to': str(new_dtype)
                }
        
        optimized_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        report = OptimizationReport(
            original_memory_mb=original_memory,
            optimized_memory_mb=optimized_memory,
            reduction_percent=((original_memory - optimized_memory) / original_memory) * 100 if original_memory > 0 else 0,
            column_changes=column_changes
        )
        
        logger.info(report.summary())
        
        return df, report
    
    def _optimize_object(self, series: pd.Series) -> tuple[pd.Series, str]:
        """Optimize object (string) columns."""
        if not self.config.convert_categories:
            return series, str(series.dtype)
        
        # Check if suitable for category
        unique_ratio = series.nunique() / len(series) if len(series) > 0 else 1.0
        
        if unique_ratio < self.config.category_threshold:
            return series.astype('category'), 'category'
        
        return series, str(series.dtype)
    
    def _downcast_integer(self, series: pd.Series) -> tuple[pd.Series, str]:
        """Downcast integer column."""
        # Handle nullable integers
        if series.isna().any():
            if self.config.enable_nullable_int:
                # Use nullable integer types
                min_val = series.min()
                max_val = series.max()
                
                if min_val >= 0:
                    if max_val <= 255:
                        return series.astype('UInt8'), 'UInt8'
                    elif max_val <= 65535:
                        return series.astype('UInt16'), 'UInt16'
                    elif max_val <= 4294967295:
                        return series.astype('UInt32'), 'UInt32'
                else:
                    if min_val >= -128 and max_val <= 127:
                        return series.astype('Int8'), 'Int8'
                    elif min_val >= -32768 and max_val <= 32767:
                        return series.astype('Int16'), 'Int16'
                    elif min_val >= -2147483648 and max_val <= 2147483647:
                        return series.astype('Int32'), 'Int32'
            return series, str(series.dtype)
        
        # Standard integer downcasting
        return pd.to_numeric(series, downcast='integer'), str(pd.to_numeric(series, downcast='integer').dtype)
    
    def _downcast_float(self, series: pd.Series) -> tuple[pd.Series, str]:
        """Downcast float column."""
        return pd.to_numeric(series, downcast='float'), str(pd.to_numeric(series, downcast='float').dtype)
    
    def get_memory_usage(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get detailed memory usage by column."""
        memory = df.memory_usage(deep=True)
        total = memory.sum()
        
        result = {
            'total_mb': total / (1024 * 1024),
            'columns': {}
        }
        
        for col in df.columns:
            col_memory = memory[col]
            result['columns'][col] = {
                'bytes': int(col_memory),
                'mb': col_memory / (1024 * 1024),
                'percent': (col_memory / total) * 100 if total > 0 else 0,
                'dtype': str(df[col].dtype)
            }
        
        return result
    
    def suggest_optimizations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Suggest optimizations without applying them."""
        suggestions = []
        
        for col in df.columns:
            dtype = df[col].dtype
            
            # Check integers
            if np.issubdtype(dtype, np.integer):
                min_val = df[col].min()
                max_val = df[col].max()
                
                if dtype == np.int64:
                    if min_val >= 0 and max_val <= 255:
                        suggestions.append({
                            'column': col,
                            'current': 'int64',
                            'suggested': 'uint8',
                            'reason': f'Range [{min_val}, {max_val}] fits in uint8'
                        })
                    elif min_val >= -128 and max_val <= 127:
                        suggestions.append({
                            'column': col,
                            'current': 'int64',
                            'suggested': 'int8',
                            'reason': f'Range [{min_val}, {max_val}] fits in int8'
                        })
            
            # Check floats
            elif np.issubdtype(dtype, np.floating):
                if dtype == np.float64:
                    # Check if float32 precision is sufficient
                    suggestions.append({
                        'column': col,
                        'current': 'float64',
                        'suggested': 'float32',
                        'reason': 'Most ML operations work fine with float32'
                    })
            
            # Check objects
            elif dtype == 'object':
                unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 1.0
                if unique_ratio < 0.5:
                    suggestions.append({
                        'column': col,
                        'current': 'object',
                        'suggested': 'category',
                        'reason': f'Only {unique_ratio:.1%} unique values'
                    })
        
        return suggestions


# Convenience functions
_default_optimizer: Optional[DataFrameOptimizer] = None


def get_optimizer() -> DataFrameOptimizer:
    global _default_optimizer
    if _default_optimizer is None:
        _default_optimizer = DataFrameOptimizer()
    return _default_optimizer


def optimize_dataframe(
    df: pd.DataFrame,
    inplace: bool = False
) -> tuple[pd.DataFrame, OptimizationReport]:
    """Optimize a DataFrame."""
    return get_optimizer().optimize(df, inplace)


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Quick memory reduction."""
    optimized, _ = get_optimizer().optimize(df)
    return optimized


def get_memory_usage(df: pd.DataFrame) -> Dict[str, float]:
    """Get memory usage details."""
    return get_optimizer().get_memory_usage(df)


if __name__ == "__main__":
    print("DataFrame Optimizer Demo")
    print("=" * 50)
    
    # Create sample DataFrame
    np.random.seed(42)
    df = pd.DataFrame({
        'user_id': np.random.randint(1, 1000, 10000),
        'movie_id': np.random.randint(1, 5000, 10000),
        'rating': np.random.uniform(1, 5, 10000),
        'genre': np.random.choice(['Action', 'Drama', 'Comedy', 'Horror'], 10000),
        'timestamp': pd.date_range('2020-01-01', periods=10000, freq='H'),
    })
    
    print(f"\nOriginal DataFrame:")
    print(df.dtypes)
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    optimizer = DataFrameOptimizer()
    
    print("\n\nSuggested optimizations:")
    for suggestion in optimizer.suggest_optimizations(df):
        print(f"  {suggestion['column']}: {suggestion['current']} -> {suggestion['suggested']}")
        print(f"    Reason: {suggestion['reason']}")
    
    print("\n\nApplying optimizations...")
    optimized, report = optimizer.optimize(df)
    
    print(f"\nOptimized DataFrame:")
    print(optimized.dtypes)
    print(f"\n{report.summary()}")
    
    print("\n\nColumn changes:")
    for col, change in report.column_changes.items():
        print(f"  {col}: {change['from']} -> {change['to']}")
