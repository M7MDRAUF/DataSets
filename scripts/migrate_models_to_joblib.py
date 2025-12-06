#!/usr/bin/env python3
"""
CineMatch Model Migration Utility

Convert existing pickle models to optimized joblib format.
Reduces model file sizes by ~80% and enables memory-mapped loading.

Usage:
    python scripts/migrate_models_to_joblib.py
    python scripts/migrate_models_to_joblib.py --compress 5
    python scripts/migrate_models_to_joblib.py --dry-run

Author: CineMatch Development Team
Date: December 5, 2025
"""

import os
import sys
import time
import argparse
import pickle
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib


def get_file_size_mb(path: Path) -> float:
    """Get file size in MB."""
    return path.stat().st_size / (1024 * 1024)


def convert_model(
    input_path: Path,
    output_path: Path,
    compress: int = 3,
    dry_run: bool = False
) -> dict:
    """
    Convert a pickle model to joblib format.
    
    Args:
        input_path: Path to input .pkl file
        output_path: Path to output .joblib file
        compress: Compression level (0-9)
        dry_run: If True, don't actually write files
        
    Returns:
        Dict with conversion metrics
    """
    result = {
        'input_path': str(input_path),
        'output_path': str(output_path),
        'input_size_mb': get_file_size_mb(input_path),
        'output_size_mb': None,
        'compression_ratio': None,
        'load_time_pickle_s': None,
        'load_time_joblib_s': None,
        'speedup_factor': None,
        'success': False,
        'error': None
    }
    
    try:
        # Load with pickle
        print(f"  Loading {input_path.name}...", end='', flush=True)
        start = time.time()
        with open(input_path, 'rb') as f:
            model = pickle.load(f)
        result['load_time_pickle_s'] = time.time() - start
        print(f" {result['load_time_pickle_s']:.2f}s")
        
        if dry_run:
            print(f"  [DRY RUN] Would save to {output_path.name}")
            result['success'] = True
            return result
        
        # Save with joblib (compressed)
        print(f"  Saving {output_path.name} (compress={compress})...", end='', flush=True)
        start = time.time()
        joblib.dump(model, str(output_path), compress=compress)
        save_time = time.time() - start
        print(f" {save_time:.2f}s")
        
        result['output_size_mb'] = get_file_size_mb(output_path)
        result['compression_ratio'] = result['input_size_mb'] / result['output_size_mb']
        
        # Verify by loading with joblib
        print(f"  Verifying {output_path.name}...", end='', flush=True)
        start = time.time()
        _ = joblib.load(str(output_path), mmap_mode='r')
        result['load_time_joblib_s'] = time.time() - start
        print(f" {result['load_time_joblib_s']:.2f}s")
        
        result['speedup_factor'] = result['load_time_pickle_s'] / result['load_time_joblib_s']
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
        print(f" ERROR: {e}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Convert pickle models to joblib format')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory containing model files')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: same as models-dir)')
    parser.add_argument('--compress', type=int, default=3,
                       help='Compression level 0-9 (default: 3)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without writing files')
    parser.add_argument('--keep-original', action='store_true',
                       help='Keep original .pkl files')
    parser.add_argument('--model', type=str, default=None,
                       help='Convert specific model only')
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir) if args.output_dir else models_dir
    
    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find model files
    if args.model:
        pkl_files = [models_dir / args.model]
    else:
        pkl_files = list(models_dir.glob('*.pkl'))
    
    if not pkl_files:
        print("No .pkl files found to convert")
        sys.exit(0)
    
    print(f"\n{'='*60}")
    print(f"CineMatch Model Migration Utility")
    print(f"{'='*60}")
    print(f"Models directory: {models_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Compression level: {args.compress}")
    print(f"Files to convert: {len(pkl_files)}")
    print(f"Dry run: {args.dry_run}")
    print(f"{'='*60}\n")
    
    results = []
    total_input_size = 0
    total_output_size = 0
    
    for pkl_file in pkl_files:
        print(f"\nConverting: {pkl_file.name}")
        print(f"  Size: {get_file_size_mb(pkl_file):.1f}MB")
        
        output_path = output_dir / f"{pkl_file.stem}.joblib"
        
        result = convert_model(
            pkl_file, output_path,
            compress=args.compress,
            dry_run=args.dry_run
        )
        results.append(result)
        
        if result['success']:
            total_input_size += result['input_size_mb']
            if result['output_size_mb']:
                total_output_size += result['output_size_mb']
            
            # Remove original if requested
            if not args.keep_original and not args.dry_run and result['output_size_mb']:
                print(f"  Removing original {pkl_file.name}")
                pkl_file.unlink()
    
    # Summary
    print(f"\n{'='*60}")
    print("CONVERSION SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"Converted: {len(successful)}/{len(results)}")
    if failed:
        print(f"Failed: {len(failed)}")
        for r in failed:
            print(f"  - {r['input_path']}: {r['error']}")
    
    if total_output_size > 0:
        print(f"\nTotal input size:  {total_input_size:.1f}MB")
        print(f"Total output size: {total_output_size:.1f}MB")
        print(f"Space saved: {total_input_size - total_output_size:.1f}MB ({(1 - total_output_size/total_input_size)*100:.0f}%)")
    
    # Performance improvements
    if successful:
        avg_speedup = sum(r['speedup_factor'] or 1 for r in successful) / len(successful)
        print(f"\nAverage load speedup: {avg_speedup:.1f}x faster")
    
    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()
