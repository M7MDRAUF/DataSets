#!/usr/bin/env python3
"""
CineMatch V2.1.6 - CSV to Parquet Conversion Script

This script converts MovieLens CSV files to Parquet format for:
- 70% storage reduction (compression)
- 5x faster load times (columnar format)
- Better memory efficiency (on-demand column loading)

Usage:
    python scripts/convert_csv_to_parquet.py
    python scripts/convert_csv_to_parquet.py --keep-csv  # Keep original CSVs

Author: CineMatch Team
Date: December 2025
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path("data/ml-32m")

# File configurations with optimal dtypes and compression settings
FILE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ratings.csv": {
        "dtypes": {
            'userId': 'int32',
            'movieId': 'int32',
            'rating': 'float32',
            'timestamp': 'int32'  # Unix timestamps fit in int32 until 2038
        },
        "compression": "zstd",  # Best compression ratio
        "row_group_size": 500_000,  # Optimize for analytical queries
    },
    "movies.csv": {
        "dtypes": {
            'movieId': 'int32',
            'title': 'str',
            'genres': 'str'
        },
        "compression": "zstd",
        "row_group_size": 50_000,
    },
    "movies_with_TMDB_image_links.csv": {
        "dtypes": {
            'movieId': 'int32',
            'title': 'str',
            'genres': 'str',
            'backdrop_path': 'str',
            'poster_path': 'str'
        },
        "compression": "zstd",
        "row_group_size": 50_000,
    },
    "links.csv": {
        "dtypes": {
            'movieId': 'int32',
            'imdbId': 'str',
            'tmdbId': 'float32'  # Has NaN values
        },
        "compression": "zstd",
        "row_group_size": 50_000,
    },
    "tags.csv": {
        "dtypes": {
            'userId': 'int32',
            'movieId': 'int32',
            'tag': 'str',
            'timestamp': 'int32'
        },
        "compression": "zstd",
        "row_group_size": 100_000,
    },
}


def get_file_size_mb(path: Path) -> float:
    """Get file size in megabytes."""
    if path.exists():
        return path.stat().st_size / (1024 * 1024)
    return 0.0


def convert_csv_to_parquet(
    csv_path: Path,
    parquet_path: Path,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert a single CSV file to Parquet format.
    
    Args:
        csv_path: Path to source CSV file
        parquet_path: Path to output Parquet file
        config: Configuration dict with dtypes, compression, row_group_size
    
    Returns:
        Dict with conversion statistics
    """
    logger.info(f"Converting: {csv_path.name}")
    
    # Get original size
    csv_size = get_file_size_mb(csv_path)
    logger.info(f"  CSV size: {csv_size:.2f} MB")
    
    # Load CSV with optimized dtypes
    start_time = time.time()
    logger.info("  Loading CSV...")
    
    df = pd.read_csv(csv_path, dtype=config["dtypes"])
    load_time = time.time() - start_time
    logger.info(f"  CSV load time: {load_time:.2f}s ({len(df):,} rows)")
    
    # Convert to PyArrow Table for better Parquet writing
    logger.info("  Converting to Parquet...")
    start_time = time.time()
    
    # Write Parquet with compression
    df.to_parquet(
        parquet_path,
        engine='pyarrow',
        compression=config["compression"],
        index=False,
        row_group_size=config["row_group_size"]
    )
    
    write_time = time.time() - start_time
    parquet_size = get_file_size_mb(parquet_path)
    
    # Calculate compression ratio
    compression_ratio = (1 - parquet_size / csv_size) * 100 if csv_size > 0 else 0
    
    logger.info(f"  Parquet size: {parquet_size:.2f} MB")
    logger.info(f"  Compression: {compression_ratio:.1f}% reduction")
    logger.info(f"  Write time: {write_time:.2f}s")
    
    # Verify by reading back
    logger.info("  Verifying Parquet file...")
    start_time = time.time()
    df_verify = pd.read_parquet(parquet_path)
    verify_time = time.time() - start_time
    
    if len(df_verify) != len(df):
        raise ValueError(f"Row count mismatch: CSV={len(df)}, Parquet={len(df_verify)}")
    
    speedup = load_time / verify_time if verify_time > 0 else 0
    logger.info(f"  Parquet load time: {verify_time:.2f}s ({speedup:.1f}x faster)")
    logger.info(f"  ✓ Verified: {len(df_verify):,} rows")
    
    return {
        "file": csv_path.name,
        "csv_size_mb": csv_size,
        "parquet_size_mb": parquet_size,
        "compression_ratio": compression_ratio,
        "rows": len(df),
        "csv_load_time": load_time,
        "parquet_load_time": verify_time,
        "speedup": speedup
    }


def main():
    parser = argparse.ArgumentParser(description="Convert MovieLens CSV files to Parquet")
    parser.add_argument(
        "--keep-csv",
        action="store_true",
        help="Keep original CSV files after conversion"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Path to data directory"
    )
    args = parser.parse_args()
    
    data_dir = args.data_dir
    
    logger.info("=" * 70)
    logger.info("CineMatch - CSV to Parquet Conversion")
    logger.info("=" * 70)
    logger.info(f"Data directory: {data_dir.absolute()}")
    logger.info("")
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Track overall statistics
    total_csv_size = 0.0
    total_parquet_size = 0.0
    conversion_stats = []
    
    # Convert each file
    for csv_filename, config in FILE_CONFIGS.items():
        csv_path = data_dir / csv_filename
        
        if not csv_path.exists():
            logger.warning(f"Skipping (not found): {csv_filename}")
            continue
        
        # Output Parquet path
        parquet_filename = csv_filename.replace(".csv", ".parquet")
        parquet_path = data_dir / parquet_filename
        
        try:
            stats = convert_csv_to_parquet(csv_path, parquet_path, config)
            conversion_stats.append(stats)
            total_csv_size += stats["csv_size_mb"]
            total_parquet_size += stats["parquet_size_mb"]
            
            # Optionally remove CSV
            if not args.keep_csv:
                logger.info(f"  Removing original CSV: {csv_filename}")
                csv_path.unlink()
            
            logger.info("")
            
        except Exception as e:
            logger.error(f"Failed to convert {csv_filename}: {e}")
            # Clean up partial Parquet file
            if parquet_path.exists():
                parquet_path.unlink()
            continue
    
    # Print summary
    logger.info("=" * 70)
    logger.info("CONVERSION SUMMARY")
    logger.info("=" * 70)
    
    if conversion_stats:
        total_reduction = (1 - total_parquet_size / total_csv_size) * 100 if total_csv_size > 0 else 0
        avg_speedup = sum(s["speedup"] for s in conversion_stats) / len(conversion_stats)
        
        logger.info(f"Files converted: {len(conversion_stats)}")
        logger.info(f"Total CSV size: {total_csv_size:.2f} MB")
        logger.info(f"Total Parquet size: {total_parquet_size:.2f} MB")
        logger.info(f"Overall reduction: {total_reduction:.1f}%")
        logger.info(f"Average load speedup: {avg_speedup:.1f}x")
        
        logger.info("")
        logger.info("Per-file statistics:")
        logger.info("-" * 70)
        for stats in conversion_stats:
            logger.info(
                f"  {stats['file']:40s} "
                f"{stats['csv_size_mb']:>8.2f} MB → "
                f"{stats['parquet_size_mb']:>8.2f} MB "
                f"({stats['compression_ratio']:>5.1f}% reduction, "
                f"{stats['speedup']:.1f}x faster)"
            )
        
        logger.info("")
        logger.info("✓ Conversion complete!")
        
        if not args.keep_csv:
            logger.info("")
            logger.info("NOTE: Original CSV files have been removed.")
            logger.info("      To keep CSVs, use --keep-csv flag.")
    else:
        logger.warning("No files were converted.")
    
    # Update REQUIRED_FILES reference
    logger.info("")
    logger.info("IMPORTANT: Update src/data_processing.py to use Parquet files.")
    logger.info("           The module has been updated to auto-detect format.")


if __name__ == "__main__":
    main()
