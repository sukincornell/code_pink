#!/usr/bin/env python
"""Main runner script for the Breast Cancer Subtypes preprocessing pipeline.

This script provides a simple interface to run the entire pipeline:
1. Test S3 connection
2. Build manifest
3. Create subset with labels
4. Preprocess volumes
5. (Optional) Extract 2D slices
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from s3_utils import test_connection
from build_manifest import build_manifest
from make_subset import (
    create_subset_manifest,
    preprocess_subset,
    create_2d_slices
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(args):
    """Run the complete preprocessing pipeline."""

    start_time = datetime.now()

    print("\n" + "="*60)
    print("BREAST CANCER SUBTYPES - PREPROCESSING PIPELINE")
    print("="*60)

    # Step 1: Test S3 connection
    print("\n[Step 1/5] Testing S3 connection...")
    print("-" * 40)
    if not test_connection():
        print("[ERROR] Failed to connect to S3. Check your internet connection.")
        return False
    print("[OK] S3 connection successful!")

    # Step 2: Build manifest
    if not args.skip_manifest:
        print("\n[Step 2/5] Building manifest...")
        print("-" * 40)
        try:
            manifest = build_manifest(
                sample_size=args.sample_size,
                save_path=args.manifest_path
            )
            print("[OK] Manifest built successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to build manifest: {e}")
            return False
    else:
        print("\n[Step 2/5] Skipping manifest build (using existing)")

    # Step 3: Create subset with labels
    print("\n[Step 3/5] Creating subset with labels...")
    print("-" * 40)
    try:
        subset = create_subset_manifest(manifest_path=args.manifest_path)
        if subset is None:
            print("[ERROR] Failed to create subset")
            return False
        print("[OK] Subset created successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to create subset: {e}")
        return False

    # Step 4: Preprocess volumes
    print("\n[Step 4/5] Preprocessing volumes...")
    print("-" * 40)
    try:
        prep_df = preprocess_subset(
            subset_path=args.subset_path,
            do_n4=args.n4
        )
        print("[OK] Preprocessing completed successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to preprocess: {e}")
        return False

    # Step 5: Extract 2D slices (optional)
    if args.extract_slices:
        print("\n[Step 5/5] Extracting 2D slices...")
        print("-" * 40)
        try:
            slices_df = create_2d_slices(
                prep_index_path=args.prep_index_path,
                num_slices=args.num_slices
            )
            print("[OK] Slice extraction completed successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to extract slices: {e}")
            return False
    else:
        print("\n[Step 5/5] Skipping 2D slice extraction")

    # Summary
    elapsed = datetime.now() - start_time
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Total time: {elapsed}")
    print("\nOutput files:")
    print("  • outputs/series_manifest.csv - Full dataset manifest")
    print("  • outputs/subset_manifest.csv - Selected subset with labels")
    print("  • outputs/preprocessed_index.csv - Preprocessed volumes index")
    print("  • outputs/preprocessed/*.npy - Preprocessed 3D volumes")
    if args.extract_slices:
        print("  • outputs/slices_index.csv - 2D slices index")
        print("  • outputs/slices/*.npy - Extracted 2D slices")

    print("\nNext steps:")
    print("  1. Review the preprocessed data statistics")
    print("  2. Load data using the index CSV files")
    print("  3. Train your models!")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the Breast Cancer Subtypes preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with small sample
  python run_pipeline.py --sample 100

  # Full preprocessing with N4 bias correction
  python run_pipeline.py --n4

  # Extract 2D slices for 2D CNN training
  python run_pipeline.py --slices

  # Use existing manifest
  python run_pipeline.py --skip-manifest
        """
    )

    # Pipeline control
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Skip manifest building (use existing)"
    )

    # Manifest options
    parser.add_argument(
        "--sample", "--sample-size",
        dest="sample_size",
        type=int,
        default=None,
        help="Process only N files for testing (default: all)"
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Path to existing manifest CSV"
    )

    # Preprocessing options
    parser.add_argument(
        "--n4",
        action="store_true",
        help="Apply N4 bias field correction (slower but better quality)"
    )
    parser.add_argument(
        "--subset-path",
        type=str,
        default=None,
        help="Path to subset manifest CSV"
    )
    parser.add_argument(
        "--prep-index-path",
        type=str,
        default=None,
        help="Path to preprocessed index CSV"
    )

    # 2D slice extraction
    parser.add_argument(
        "--slices", "--extract-slices",
        dest="extract_slices",
        action="store_true",
        help="Extract 2D slices from volumes"
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=3,
        help="Number of 2D slices to extract per volume (default: 3)"
    )

    args = parser.parse_args()

    # Run pipeline
    success = run_pipeline(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()