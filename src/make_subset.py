"""Create a manageable subset of the data for development and testing."""

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from s3fs import S3FileSystem
from sklearn.model_selection import StratifiedGroupKFold
import logging

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import META_DIR, OUTPUTS_DIR, SUBSET_SIZE, N_FOLDS, RANDOM_SEED
from dicom_utils import load_series_3d
from preprocess import full_pipeline, extract_2d_slices
from s3_utils import get_fs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_metadata():
    """Load and merge metadata/labels from S3.

    Returns:
        DataFrame with patient labels
    """
    fs = get_fs()

    # List metadata files
    meta_files = fs.glob(f"{META_DIR}/*")
    logger.info(f"Found {len(meta_files)} metadata files:")
    for f in meta_files:
        logger.info(f"  - {f.split('/')[-1]}")

    # Find files with labels/clinical data
    label_files = []
    for f in meta_files:
        fname = f.lower()
        if any(keyword in fname for keyword in ["subtype", "label", "clinical", "outcome"]):
            label_files.append(f)

    if not label_files:
        logger.warning("No label files found. Using all metadata files.")
        label_files = meta_files

    # Read and concatenate metadata
    dfs = []
    for f in label_files:
        try:
            if f.lower().endswith(".csv"):
                df = pd.read_csv(fs.open(f, "rb"))
            elif f.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(fs.open(f, "rb"), engine="openpyxl")
            else:
                continue
            dfs.append(df)
            logger.info(f"  Loaded {f.split('/')[-1]}: {len(df)} rows, columns: {list(df.columns)[:5]}")
        except Exception as e:
            logger.warning(f"  Failed to load {f}: {e}")

    if not dfs:
        raise ValueError("Could not load any metadata files")

    # Combine all metadata
    if len(dfs) == 1:
        labels = dfs[0]
    else:
        # Try to merge on common columns
        labels = dfs[0]
        for df in dfs[1:]:
            common_cols = list(set(labels.columns) & set(df.columns))
            if common_cols:
                labels = labels.merge(df, on=common_cols, how="outer")
            else:
                logger.warning(f"No common columns to merge on")

    # Standardize column names
    column_mapping = {
        "patient_id": "PatientID",
        "Patient Id": "PatientID",
        "Patient_ID": "PatientID",
        "PatientId": "PatientID",
        "Subtype": "tumor_subtype",
        "TumorSubtype": "tumor_subtype",
        "Tumor_Subtype": "tumor_subtype",
        "Cancer_Type": "tumor_subtype",
        "Molecular_Subtype": "tumor_subtype",
    }

    for old, new in column_mapping.items():
        if old in labels.columns:
            labels = labels.rename(columns={old: new})

    # Check for required columns
    if "PatientID" not in labels.columns:
        logger.error(f"Available columns: {list(labels.columns)}")
        raise ValueError("PatientID column not found in metadata")

    # Find tumor subtype column
    subtype_cols = [col for col in labels.columns if "subtype" in col.lower() or "type" in col.lower()]
    if subtype_cols and "tumor_subtype" not in labels.columns:
        labels["tumor_subtype"] = labels[subtype_cols[0]]
        logger.info(f"Using {subtype_cols[0]} as tumor_subtype column")

    # Clean up
    labels = labels.dropna(subset=["PatientID"])
    labels = labels.drop_duplicates("PatientID")

    logger.info(f"Loaded metadata for {len(labels)} patients")
    if "tumor_subtype" in labels.columns:
        logger.info(f"Tumor subtypes: {labels['tumor_subtype'].value_counts().to_dict()}")

    return labels


def create_subset_manifest(manifest_path: str = None):
    """Create a subset of series with labels for development.

    Args:
        manifest_path: Path to series manifest CSV

    Returns:
        DataFrame with subset of series and labels
    """
    # Load series manifest
    if manifest_path is None:
        manifest_path = os.path.join(OUTPUTS_DIR, "series_manifest.csv")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            "Run 'python -m src.build_manifest' first."
        )

    manifest = pd.read_csv(manifest_path)
    logger.info(f"Loaded manifest with {len(manifest)} series")

    # Load metadata
    labels = load_metadata()

    # Merge manifest with labels
    df = manifest.merge(labels[["PatientID", "tumor_subtype"]], on="PatientID", how="inner")
    logger.info(f"Matched {len(df)} series with labels")

    if len(df) == 0:
        logger.error("No matching patients found between manifest and metadata")
        return None

    # Select densest series per patient (most slices)
    densest = df.sort_values("num_instances", ascending=False).groupby("PatientID").first().reset_index()
    logger.info(f"Selected densest series for {len(densest)} patients")

    # Create balanced subset
    if "tumor_subtype" in densest.columns:
        # Sample up to SUBSET_SIZE patients per subtype
        subset = densest.groupby("tumor_subtype", group_keys=False).apply(
            lambda x: x.head(SUBSET_SIZE)
        ).reset_index(drop=True)
    else:
        # Just take first SUBSET_SIZE * 3 patients
        subset = densest.head(SUBSET_SIZE * 3)

    # Save subset manifest
    subset_path = os.path.join(OUTPUTS_DIR, "subset_manifest.csv")
    subset.to_csv(subset_path, index=False)
    logger.info(f"Saved subset manifest to {subset_path}")

    # Print summary
    print("\n" + "="*50)
    print("SUBSET SUMMARY")
    print("="*50)
    print(f"Total patients: {subset['PatientID'].nunique()}")
    print(f"Total series: {len(subset)}")
    if "tumor_subtype" in subset.columns:
        print(f"\nTumor subtypes:")
        print(subset["tumor_subtype"].value_counts())

    return subset


def preprocess_subset(subset_path: str = None, do_n4: bool = False):
    """Preprocess subset of volumes and save as numpy arrays.

    Args:
        subset_path: Path to subset manifest CSV
        do_n4: Whether to apply N4 bias correction (compute-intensive)

    Returns:
        DataFrame with preprocessed file paths
    """
    # Load subset manifest
    if subset_path is None:
        subset_path = os.path.join(OUTPUTS_DIR, "subset_manifest.csv")

    subset = pd.read_csv(subset_path)

    # Create preprocessed directory
    prep_dir = os.path.join(OUTPUTS_DIR, "preprocessed")
    os.makedirs(prep_dir, exist_ok=True)

    records = []
    failed = []

    for idx, row in tqdm(subset.iterrows(), total=len(subset), desc="Preprocessing volumes"):
        try:
            # Load DICOM volume
            vol, meta = load_series_3d(row["S3Object"])

            # Apply preprocessing
            vol_processed = full_pipeline(
                vol,
                meta["spacing_mm"],
                do_n4=do_n4,
                normalize_method="zscore",
                out_mm=1.5,
                target=(128, 192, 192)
            )

            # Save preprocessed volume
            filename = f"{row['PatientID']}__{row['SeriesInstanceUID']}.npy"
            npy_path = os.path.join(prep_dir, filename)
            np.save(npy_path, vol_processed)

            # Record success
            record = {
                "PatientID": row["PatientID"],
                "SeriesInstanceUID": row["SeriesInstanceUID"],
                "tumor_subtype": row.get("tumor_subtype", "unknown"),
                "npy_path": npy_path,
                "shape": vol_processed.shape,
                "dtype": str(vol_processed.dtype),
                "min_val": float(vol_processed.min()),
                "max_val": float(vol_processed.max()),
            }
            records.append(record)

        except Exception as e:
            logger.warning(f"Failed to process {row['PatientID']}: {e}")
            failed.append(row["PatientID"])

    # Create preprocessed index
    prep_df = pd.DataFrame(records)

    # Add train/val/test splits
    if "tumor_subtype" in prep_df.columns and prep_df["tumor_subtype"].nunique() > 1:
        prep_df = add_splits(prep_df)

    # Save index
    index_path = os.path.join(OUTPUTS_DIR, "preprocessed_index.csv")
    prep_df.to_csv(index_path, index=False)

    # Print summary
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY")
    print("="*50)
    print(f"Successfully processed: {len(records)} volumes")
    print(f"Failed: {len(failed)} volumes")
    if records:
        print(f"Output shape: {records[0]['shape']}")
        print(f"Output dtype: {records[0]['dtype']}")
    print(f"Index saved to: {index_path}")

    return prep_df


def add_splits(df: pd.DataFrame, n_folds: int = N_FOLDS):
    """Add train/val/test splits to DataFrame, stratified by patient.

    Args:
        df: DataFrame with PatientID and tumor_subtype columns
        n_folds: Number of folds for cross-validation

    Returns:
        DataFrame with added 'split' and 'fold' columns
    """
    X = df.index.values
    y = df["tumor_subtype"].values
    groups = df["PatientID"].values

    # Create stratified group k-fold
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    # Assign folds
    df["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        df.loc[val_idx, "fold"] = fold

    # Assign train/val/test splits (fold 0 = test, fold 1 = val, rest = train)
    df["split"] = "train"
    df.loc[df["fold"] == 0, "split"] = "test"
    df.loc[df["fold"] == 1, "split"] = "val"

    logger.info(f"Split distribution:\n{df['split'].value_counts()}")

    return df


def create_2d_slices(prep_index_path: str = None, num_slices: int = 3):
    """Extract 2D slices from preprocessed 3D volumes.

    Args:
        prep_index_path: Path to preprocessed index CSV
        num_slices: Number of slices to extract per volume

    Returns:
        DataFrame with 2D slice paths
    """
    # Load preprocessed index
    if prep_index_path is None:
        prep_index_path = os.path.join(OUTPUTS_DIR, "preprocessed_index.csv")

    df = pd.read_csv(prep_index_path)

    # Create slices directory
    slices_dir = os.path.join(OUTPUTS_DIR, "slices")
    os.makedirs(slices_dir, exist_ok=True)

    records = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting 2D slices"):
        try:
            # Load volume
            vol = np.load(row["npy_path"])

            # Extract slices
            slices = extract_2d_slices(vol, num_slices=num_slices, method="center")

            # Save each slice
            for i, slice_2d in enumerate(slices):
                filename = f"{row['PatientID']}_slice{i:02d}.npy"
                slice_path = os.path.join(slices_dir, filename)
                np.save(slice_path, slice_2d)

                record = {
                    "PatientID": row["PatientID"],
                    "tumor_subtype": row.get("tumor_subtype", "unknown"),
                    "split": row.get("split", "train"),
                    "fold": row.get("fold", -1),
                    "slice_idx": i,
                    "npy_path": slice_path,
                    "shape": slice_2d.shape,
                }
                records.append(record)

        except Exception as e:
            logger.warning(f"Failed to extract slices from {row['PatientID']}: {e}")

    # Create slices index
    slices_df = pd.DataFrame(records)
    slices_path = os.path.join(OUTPUTS_DIR, "slices_index.csv")
    slices_df.to_csv(slices_path, index=False)

    logger.info(f"Extracted {len(slices_df)} slices, saved to {slices_path}")

    return slices_df


def main():
    """Main pipeline to create and preprocess subset."""
    import argparse

    parser = argparse.ArgumentParser(description="Create and preprocess data subset")
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Skip building manifest (use existing)"
    )
    parser.add_argument(
        "--n4",
        action="store_true",
        help="Apply N4 bias correction (compute-intensive)"
    )
    parser.add_argument(
        "--slices",
        action="store_true",
        help="Also extract 2D slices"
    )

    args = parser.parse_args()

    # Step 1: Build manifest (if not skipped)
    if not args.skip_manifest:
        logger.info("Step 1: Building manifest...")
        from build_manifest import build_manifest
        build_manifest(sample_size=5000)  # Sample for faster testing

    # Step 2: Create subset with labels
    logger.info("Step 2: Creating subset with labels...")
    subset = create_subset_manifest()

    if subset is None:
        logger.error("Failed to create subset")
        return

    # Step 3: Preprocess subset
    logger.info("Step 3: Preprocessing subset...")
    prep_df = preprocess_subset(do_n4=args.n4)

    # Step 4: Extract 2D slices (optional)
    if args.slices:
        logger.info("Step 4: Extracting 2D slices...")
        slices_df = create_2d_slices()

    print("\n" + "="*50)
    print("PIPELINE COMPLETE!")
    print("="*50)
    print(f"Next steps:")
    print(f"1. Check outputs/ directory for preprocessed data")
    print(f"2. Use preprocessed_index.csv for training")
    if args.slices:
        print(f"3. Use slices_index.csv for 2D models")


if __name__ == "__main__":
    main()