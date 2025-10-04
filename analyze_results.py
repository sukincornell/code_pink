#!/usr/bin/env python
"""Analyze the results of the preprocessing pipeline."""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def analyze_results():
    """Analyze preprocessed data and generate statistics."""

    print("\n" + "="*60)
    print("PREPROCESSING RESULTS ANALYSIS")
    print("="*60)

    # Check what files were created
    outputs_dir = Path("outputs")

    # 1. Check manifest
    if (outputs_dir / "series_manifest.csv").exists():
        manifest = pd.read_csv(outputs_dir / "series_manifest.csv")
        print("\n DATASET MANIFEST")
        print("-" * 40)
        print(f"Total patients: {manifest['PatientID'].nunique()}")
        print(f"Total studies: {manifest['StudyInstanceUID'].nunique()}")
        print(f"Total series: {len(manifest)}")
        print(f"\nModalities distribution:")
        print(manifest["Modality"].value_counts().head())
        print(f"\nSeries size statistics:")
        print(f"  Mean slices per series: {manifest['num_instances'].mean():.1f}")
        print(f"  Median slices per series: {manifest['num_instances'].median():.0f}")
        print(f"  Max slices per series: {manifest['num_instances'].max()}")

    # 2. Check subset with labels
    if (outputs_dir / "subset_manifest.csv").exists():
        subset = pd.read_csv(outputs_dir / "subset_manifest.csv")
        print("\n LABELED SUBSET")
        print("-" * 40)
        print(f"Selected patients: {subset['PatientID'].nunique()}")
        print(f"Selected series: {len(subset)}")

        if "tumor_subtype" in subset.columns:
            print(f"\nTumor subtype distribution:")
            print(subset["tumor_subtype"].value_counts())
        else:
            print("[WARNING]  No tumor subtype labels found")

    # 3. Check preprocessed volumes
    if (outputs_dir / "preprocessed_index.csv").exists():
        prep_df = pd.read_csv(outputs_dir / "preprocessed_index.csv")
        print("\n PREPROCESSED VOLUMES")
        print("-" * 40)
        print(f"Total preprocessed volumes: {len(prep_df)}")

        if "split" in prep_df.columns:
            print(f"\nTrain/Val/Test split:")
            print(prep_df["split"].value_counts())

        if "tumor_subtype" in prep_df.columns:
            print(f"\nSubtypes per split:")
            print(prep_df.groupby(["split", "tumor_subtype"]).size().unstack(fill_value=0))

        # Check actual files
        prep_dir = outputs_dir / "preprocessed"
        if prep_dir.exists():
            npy_files = list(prep_dir.glob("*.npy"))
            if npy_files:
                # Sample one file to check
                sample_vol = np.load(npy_files[0])
                print(f"\nVolume characteristics:")
                print(f"  Shape: {sample_vol.shape}")
                print(f"  Dtype: {sample_vol.dtype}")
                print(f"  Size per volume: {sample_vol.nbytes / 1024 / 1024:.1f} MB")

                # Check value statistics across a few volumes
                stats = []
                for f in npy_files[:min(5, len(npy_files))]:
                    v = np.load(f)
                    stats.append({
                        'min': v.min(),
                        'max': v.max(),
                        'mean': v.mean(),
                        'std': v.std()
                    })

                stats_df = pd.DataFrame(stats)
                print(f"\nIntensity statistics (first {len(stats)} volumes):")
                print(f"  Mean range: [{stats_df['min'].mean():.2f}, {stats_df['max'].mean():.2f}]")
                print(f"  Mean μ: {stats_df['mean'].mean():.2f} ± {stats_df['mean'].std():.2f}")
                print(f"  Mean σ: {stats_df['std'].mean():.2f} ± {stats_df['std'].std():.2f}")

    # 4. Check 2D slices
    if (outputs_dir / "slices_index.csv").exists():
        slices_df = pd.read_csv(outputs_dir / "slices_index.csv")
        print("\n 2D SLICES")
        print("-" * 40)
        print(f"Total 2D slices extracted: {len(slices_df)}")
        print(f"Slices per patient: {len(slices_df) / slices_df['PatientID'].nunique():.1f}")

        if "split" in slices_df.columns:
            print(f"\nSlices per split:")
            print(slices_df["split"].value_counts())

        # Check actual slice files
        slices_dir = outputs_dir / "slices"
        if slices_dir.exists():
            slice_files = list(slices_dir.glob("*.npy"))
            if slice_files:
                sample_slice = np.load(slice_files[0])
                print(f"\n2D slice characteristics:")
                print(f"  Shape: {sample_slice.shape}")
                print(f"  Size per slice: {sample_slice.nbytes / 1024:.1f} KB")

    # 5. Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Calculate total disk usage
    total_size = 0
    for root, dirs, files in os.walk(outputs_dir):
        for file in files:
            if file.endswith('.npy'):
                total_size += os.path.getsize(os.path.join(root, file))

    print(f"Total disk usage: {total_size / 1024 / 1024:.1f} MB")

    # Check if ready for training
    ready = True
    requirements = []

    if not (outputs_dir / "preprocessed_index.csv").exists():
        ready = False
        requirements.append("[FAILED] No preprocessed volumes found")
    else:
        requirements.append("[DONE] Preprocessed volumes ready")

    if (outputs_dir / "subset_manifest.csv").exists():
        subset = pd.read_csv(outputs_dir / "subset_manifest.csv")
        if "tumor_subtype" not in subset.columns:
            ready = False
            requirements.append("[FAILED] No tumor subtype labels")
        else:
            requirements.append("[DONE] Labels available")

    if (outputs_dir / "preprocessed_index.csv").exists():
        prep_df = pd.read_csv(outputs_dir / "preprocessed_index.csv")
        if "split" in prep_df.columns:
            requirements.append("[DONE] Train/val/test splits created")
        else:
            requirements.append("[WARNING]  No train/val/test splits")

    print("\nReadiness checklist:")
    for req in requirements:
        print(f"  {req}")

    if ready:
        print("\n[SUCCESS] Data is ready for model training!")
    else:
        print("\n[WARNING]  Some issues need to be addressed before training")

    return ready

if __name__ == "__main__":
    analyze_results()