#!/usr/bin/env python
"""Run complete pipeline with controlled patient selection."""

import sys
import os
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from tqdm import tqdm
from s3_utils import get_fs
from dicom_utils import load_series_3d
from preprocess import full_pipeline, extract_2d_slices
from sklearn.model_selection import StratifiedGroupKFold
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("\n" + "="*60)
    print("COMPLETE BREAST CANCER PIPELINE")
    print("="*60)

    fs = get_fs()
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/preprocessed", exist_ok=True)
    os.makedirs("outputs/slices", exist_ok=True)

    # Step 1: Build a targeted manifest for first N patients
    print("\n[Step 1/5] Building manifest for first patients...")
    print("-" * 40)

    patients = fs.ls('s3://cu-qatar-hackathon-2025/Duke-Breast-Cancer-MRI/')
    # Filter to actual patient folders
    patients = [p for p in patients if 'Breast_MRI_' in p][:20]  # First 20 patients

    print(f"Processing {len(patients)} patients...")

    manifest_rows = []
    for patient_path in tqdm(patients, desc="Scanning patients"):
        patient_id = patient_path.split('/')[-1]

        # Get all studies for this patient
        studies = fs.ls(patient_path)

        for study_path in studies[:1]:  # Take first study only for speed
            # Get all series in this study
            series_list = fs.ls(study_path)

            for series_path in series_list[:2]:  # Take first 2 series per study
                # Get DICOM files
                files = fs.ls(series_path)
                dcm_files = [f for f in files if f.endswith('.dcm')]

                if dcm_files:
                    manifest_rows.append({
                        'PatientID': patient_id,
                        'StudyPath': study_path,
                        'SeriesPath': series_path,
                        'SeriesName': series_path.split('/')[-1],
                        'NumInstances': len(dcm_files),
                        'SampleDICOM': dcm_files[0]
                    })

    manifest_df = pd.DataFrame(manifest_rows)

    # Select densest series per patient
    manifest_df = manifest_df.sort_values('NumInstances', ascending=False).groupby('PatientID').first().reset_index()

    manifest_df.to_csv('outputs/manifest.csv', index=False)
    print(f"[OK] Created manifest with {len(manifest_df)} series from {manifest_df['PatientID'].nunique()} patients")

    # Step 2: Load metadata
    print("\n[Step 2/5] Loading metadata...")
    print("-" * 40)

    meta_files = fs.glob('s3://cu-qatar-hackathon-2025/Duke-Breast-Cancer-MRI/a_metadata/*.xlsx')

    # Try Clinical_and_Other_Features.xlsx first
    clinical_file = [f for f in meta_files if 'Clinical' in f][0]
    print(f"Loading {clinical_file.split('/')[-1]}...")

    with fs.open(clinical_file, 'rb') as f:
        # Skip the first row which contains category headers
        clinical_df = pd.read_excel(f, engine='openpyxl', skiprows=1)

    print(f"[OK] Loaded clinical data with {len(clinical_df)} rows")

    # Map patient IDs - the column is 'Patient ID' in this dataset
    if 'Patient ID' in clinical_df.columns:
        clinical_df['PatientID'] = clinical_df['Patient ID']
    else:
        print("  [WARNING] No Patient ID column found")
        clinical_df['PatientID'] = 'Unknown'

    # Use 'Mol Subtype' column for tumor subtype
    if 'Mol Subtype' in clinical_df.columns:
        clinical_df['tumor_subtype'] = clinical_df['Mol Subtype']
        print(f"  Using 'Mol Subtype' as tumor subtype")
        # Clean up the subtypes
        clinical_df['tumor_subtype'] = clinical_df['tumor_subtype'].fillna('Unknown')
        subtype_counts = clinical_df['tumor_subtype'].value_counts()
        print(f"  Subtype distribution: {subtype_counts.to_dict()}")
    else:
        # If no subtype, create synthetic labels for testing
        print("  [WARNING] No subtype found, creating synthetic labels for testing")
        np.random.seed(42)
        clinical_df['tumor_subtype'] = np.random.choice(['TypeA', 'TypeB', 'TypeC'], len(clinical_df))

    # Merge with manifest
    merged_df = manifest_df.merge(clinical_df[['PatientID', 'tumor_subtype']], on='PatientID', how='left')

    # Fill missing subtypes
    merged_df['tumor_subtype'] = merged_df['tumor_subtype'].fillna('Unknown')

    merged_df.to_csv('outputs/manifest_with_labels.csv', index=False)
    print(f"[OK] Merged labels: {merged_df['tumor_subtype'].value_counts().to_dict()}")

    # Step 3: Preprocess volumes
    print("\n[Step 3/5] Preprocessing volumes...")
    print("-" * 40)

    prep_records = []
    failed = []

    # Process up to 10 volumes for demonstration
    to_process = merged_df.head(10)

    for idx, row in tqdm(to_process.iterrows(), total=len(to_process), desc="Processing"):
        try:
            # Load volume
            vol, meta = load_series_3d(row['SampleDICOM'])

            # Preprocess
            processed = full_pipeline(
                vol,
                meta['spacing_mm'],
                do_n4=False,  # Skip N4 for speed
                normalize_method='zscore',
                out_mm=1.5,
                target=(128, 192, 192)
            )

            # Save
            filename = f"{row['PatientID']}_preprocessed.npy"
            np.save(f'outputs/preprocessed/{filename}', processed)

            prep_records.append({
                'PatientID': row['PatientID'],
                'tumor_subtype': row['tumor_subtype'],
                'npy_path': f'outputs/preprocessed/{filename}',
                'shape': processed.shape
            })

        except Exception as e:
            logger.warning(f"Failed to process {row['PatientID']}: {e}")
            failed.append(row['PatientID'])
            continue

    prep_df = pd.DataFrame(prep_records)

    # Step 4: Add train/val/test splits
    print("\n[Step 4/5] Creating train/val/test splits...")
    print("-" * 40)

    if len(prep_df) > 0:
        # Add splits
        X = prep_df.index.values
        y = prep_df['tumor_subtype'].values
        groups = prep_df['PatientID'].values

        # Simple split for small dataset
        n = len(prep_df)
        train_n = int(0.6 * n)
        val_n = int(0.2 * n)

        prep_df['split'] = 'test'
        prep_df.iloc[:train_n, prep_df.columns.get_loc('split')] = 'train'
        prep_df.iloc[train_n:train_n+val_n, prep_df.columns.get_loc('split')] = 'val'

        prep_df.to_csv('outputs/preprocessed_index.csv', index=False)
        print(f"[OK] Split distribution: {prep_df['split'].value_counts().to_dict()}")

        # Step 5: Extract 2D slices
        print("\n[Step 5/5] Extracting 2D slices...")
        print("-" * 40)

        slice_records = []
        for idx, row in prep_df.iterrows():
            vol = np.load(row['npy_path'])

            # Extract 5 center slices
            slices = extract_2d_slices(vol, num_slices=5, method='center')

            for i, slice_2d in enumerate(slices):
                filename = f"{row['PatientID']}_slice{i}.npy"
                np.save(f'outputs/slices/{filename}', slice_2d)

                slice_records.append({
                    'PatientID': row['PatientID'],
                    'tumor_subtype': row['tumor_subtype'],
                    'split': row['split'],
                    'slice_idx': i,
                    'npy_path': f'outputs/slices/{filename}'
                })

        slices_df = pd.DataFrame(slice_records)
        slices_df.to_csv('outputs/slices_index.csv', index=False)
        print(f"[OK] Extracted {len(slices_df)} slices")

    # Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"[OK] Processed {len(prep_df)} volumes successfully")
    if failed:
        print(f"[WARNING] Failed: {failed}")

    print("\nOutput files:")
    print("  • outputs/manifest.csv")
    print("  • outputs/manifest_with_labels.csv")
    print("  • outputs/preprocessed_index.csv")
    print("  • outputs/slices_index.csv")
    print("  • outputs/preprocessed/*.npy")
    print("  • outputs/slices/*.npy")

    print("\nRun 'python analyze_results.py' to see detailed statistics")

if __name__ == "__main__":
    main()