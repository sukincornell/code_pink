"""Build a manifest of all DICOM series in the dataset."""

import os
import sys
import pandas as pd
from tqdm import tqdm
from s3fs import S3FileSystem
import logging

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import S3_BUCKET, OUTPUTS_DIR
from dicom_utils import read_header_only
from s3_utils import get_fs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_manifest(sample_size: int = None, save_path: str = None):
    """Build manifest of all DICOM series in the dataset.

    Args:
        sample_size: If specified, only process this many DICOM files (for testing)
        save_path: Path to save manifest CSV (default: outputs/series_manifest.csv)

    Returns:
        DataFrame with series information
    """
    fs = get_fs()

    # Create outputs directory
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Find all DICOM files
    logger.info(f"Scanning for DICOM files in {S3_BUCKET}...")
    dcm_paths = fs.glob(f"{S3_BUCKET}/**/*.dcm")

    if sample_size:
        dcm_paths = dcm_paths[:sample_size]
        logger.info(f"Processing sample of {sample_size} files")
    else:
        logger.info(f"Found {len(dcm_paths)} DICOM files")

    # Read headers
    rows = []
    failed = 0
    for path in tqdm(dcm_paths, desc="Reading DICOM headers"):
        header = read_header_only(fs, path)
        if header:
            rows.append(header)
        else:
            failed += 1

    if failed > 0:
        logger.warning(f"Failed to read {failed} files")

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Group by series and get statistics
    logger.info("Grouping by series...")

    # Count instances per series
    series_counts = df.groupby(
        ["PatientID", "StudyInstanceUID", "SeriesInstanceUID"]
    ).size().reset_index(name="num_instances")

    # Get first file of each series (sorted by InstanceNumber) as representative
    df_sorted = df.sort_values("InstanceNumber", na_position="last")
    series_first = df_sorted.groupby(
        ["PatientID", "StudyInstanceUID", "SeriesInstanceUID"],
        as_index=False
    ).first()

    # Select relevant columns
    series_first = series_first[[
        "PatientID", "StudyInstanceUID", "SeriesInstanceUID",
        "SeriesDescription", "Modality", "S3Object",
        "PixelSpacing", "SliceThickness", "SpacingBetweenSlices"
    ]]

    # Merge with counts
    manifest = series_first.merge(
        series_counts,
        on=["PatientID", "StudyInstanceUID", "SeriesInstanceUID"],
        how="left"
    )

    # Add series folder path
    manifest["series_folder"] = manifest["S3Object"].apply(
        lambda x: x.rsplit("/", 1)[0] if x else None
    )

    # Save manifest
    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "series_manifest.csv")

    manifest.to_csv(save_path, index=False)
    logger.info(f"Saved manifest to {save_path}")

    # Print summary statistics
    print("\n" + "="*50)
    print("MANIFEST SUMMARY")
    print("="*50)
    print(f"Total patients: {manifest['PatientID'].nunique()}")
    print(f"Total studies: {manifest['StudyInstanceUID'].nunique()}")
    print(f"Total series: {len(manifest)}")
    print(f"\nModalities:")
    print(manifest["Modality"].value_counts())
    print(f"\nSeries with most slices:")
    top_series = manifest.nlargest(5, "num_instances")[
        ["PatientID", "SeriesDescription", "num_instances"]
    ]
    print(top_series.to_string(index=False))

    return manifest


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Build manifest of DICOM series")
    parser.add_argument(
        "--sample",
        type=int,
        help="Process only this many files (for testing)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for manifest CSV"
    )

    args = parser.parse_args()

    manifest = build_manifest(sample_size=args.sample, save_path=args.output)


if __name__ == "__main__":
    main()