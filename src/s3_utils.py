"""S3 utilities for accessing the public Duke Breast Cancer MRI dataset."""

import s3fs
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_fs() -> s3fs.S3FileSystem:
    """Get S3 filesystem with anonymous access for public bucket."""
    return s3fs.S3FileSystem(anon=True)


def ls(path: str, pattern: str = "*") -> List[str]:
    """List files in S3 path matching pattern.

    Args:
        path: S3 path to list
        pattern: Glob pattern to match (default: "*")

    Returns:
        List of S3 paths matching pattern
    """
    fs = get_fs()
    full_path = f"{path.rstrip('/')}/{pattern}"
    results = fs.glob(full_path)
    logger.info(f"Found {len(results)} files matching {full_path}")
    return results


def test_connection():
    """Test S3 connection to the public bucket."""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from config import S3_BUCKET, META_DIR

        # Test listing root
        root_files = ls(S3_BUCKET, "*")[:5]
        print(f"[OK] Connected to S3 bucket: {S3_BUCKET}")
        print(f"  Sample files: {root_files}")

        # Test listing metadata
        meta_files = ls(META_DIR, "*")
        print(f"[OK] Found {len(meta_files)} metadata files:")
        for f in meta_files:
            print(f"  - {f.split('/')[-1]}")

        return True
    except Exception as e:
        print(f"[ERROR] Failed to connect to S3: {e}")
        return False


if __name__ == "__main__":
    test_connection()