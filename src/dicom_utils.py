"""DICOM utilities for loading and processing MRI volumes."""

import numpy as np
import pydicom
from s3fs import S3FileSystem
from typing import Tuple, Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_header_only(fs: S3FileSystem, s3_path: str) -> Dict:
    """Read DICOM header without pixel data for fast metadata extraction.

    Args:
        fs: S3 filesystem object
        s3_path: Path to DICOM file in S3

    Returns:
        Dictionary of DICOM metadata
    """
    try:
        with fs.open(s3_path, "rb") as f:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)

        return {
            "PatientID": getattr(ds, "PatientID", None),
            "StudyInstanceUID": getattr(ds, "StudyInstanceUID", None),
            "SeriesInstanceUID": getattr(ds, "SeriesInstanceUID", None),
            "SeriesDescription": getattr(ds, "SeriesDescription", None),
            "Modality": getattr(ds, "Modality", None),
            "InstanceNumber": getattr(ds, "InstanceNumber", None),
            "ImagePositionPatient": getattr(ds, "ImagePositionPatient", None),
            "PixelSpacing": getattr(ds, "PixelSpacing", None),
            "SliceThickness": getattr(ds, "SliceThickness", None),
            "SpacingBetweenSlices": getattr(ds, "SpacingBetweenSlices", None),
            "S3Object": s3_path,
        }
    except Exception as e:
        logger.warning(f"Failed to read header from {s3_path}: {e}")
        return None


def load_series_3d(series_rep_file: str) -> Tuple[np.ndarray, Dict]:
    """Load all DICOM slices from a series and stack into 3D volume.

    Args:
        series_rep_file: Path to any DICOM file in the series

    Returns:
        Tuple of (volume as float32 numpy array (Z,H,W), metadata dict)
    """
    fs = S3FileSystem(anon=True)

    # Assume all slices are in the same folder
    folder = series_rep_file.rsplit("/", 1)[0]
    dcm_files = fs.glob(f"{folder}/*.dcm")

    if not dcm_files:
        raise ValueError(f"No DICOM files found in {folder}")

    logger.info(f"Loading {len(dcm_files)} slices from {folder}")

    # Read all DICOM files with headers
    headers = []
    for f in dcm_files:
        try:
            with fs.open(f, "rb") as fh:
                ds = pydicom.dcmread(fh, force=True)
                headers.append((f, ds))
        except Exception as e:
            logger.warning(f"Failed to read {f}: {e}")
            continue

    if not headers:
        raise ValueError(f"Could not read any DICOM files from {folder}")

    # Sort slices by ImagePositionPatient[2] (z-coordinate) or InstanceNumber
    def slice_key(item):
        ds = item[1]
        ipp = getattr(ds, "ImagePositionPatient", None)
        if isinstance(ipp, (list, tuple)) and len(ipp) >= 3:
            return float(ipp[2])
        else:
            # Fallback to InstanceNumber
            return float(getattr(ds, "InstanceNumber", 0))

    headers.sort(key=slice_key)

    # Extract pixel arrays and apply rescale slope/intercept
    arrays = []
    for f, ds in headers:
        try:
            arr = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            inter = float(getattr(ds, "RescaleIntercept", 0.0))
            arr = arr * slope + inter
            arrays.append(arr)
        except Exception as e:
            logger.warning(f"Failed to extract pixels from {f}: {e}")
            continue

    if not arrays:
        raise ValueError(f"Could not extract pixels from any DICOM files")

    # Stack into 3D volume (Z, H, W)
    vol = np.stack(arrays, axis=0)

    # Extract spacing information
    first_ds = headers[0][1]
    px_spacing = getattr(first_ds, "PixelSpacing", [1.0, 1.0])

    # Try to get z-spacing from various DICOM tags
    try:
        spacing_z = float(getattr(first_ds, "SpacingBetweenSlices",
                                 getattr(first_ds, "SliceThickness", 1.0)))
    except:
        # Calculate from ImagePositionPatient if available
        if len(headers) > 1:
            ipp1 = getattr(headers[0][1], "ImagePositionPatient", None)
            ipp2 = getattr(headers[1][1], "ImagePositionPatient", None)
            if ipp1 and ipp2:
                spacing_z = abs(float(ipp2[2]) - float(ipp1[2]))
            else:
                spacing_z = 1.0
        else:
            spacing_z = 1.0

    spacing = (spacing_z, float(px_spacing[0]), float(px_spacing[1]))  # (z, y, x) in mm

    meta = {
        "shape": vol.shape,
        "spacing_mm": spacing,
        "series_folder": folder,
        "series_uid": getattr(first_ds, "SeriesInstanceUID", None),
        "series_desc": getattr(first_ds, "SeriesDescription", None),
        "patient_id": getattr(first_ds, "PatientID", None),
        "modality": getattr(first_ds, "Modality", None),
        "num_slices": len(arrays),
    }

    logger.info(f"Loaded volume: shape={vol.shape}, spacing={spacing}")

    return vol, meta


def test_load_single_file():
    """Test loading a single DICOM series (for debugging)."""
    from s3_utils import ls
    from config import S3_BUCKET

    # Find a sample DICOM file
    dcm_files = ls(S3_BUCKET, "**/*.dcm")[:100]  # Get first 100 for testing
    if dcm_files:
        sample_file = dcm_files[0]
        print(f"Testing with: {sample_file}")
        try:
            vol, meta = load_series_3d(sample_file)
            print(f"[OK] Successfully loaded volume:")
            for k, v in meta.items():
                print(f"  {k}: {v}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load: {e}")
            return False
    else:
        print("[ERROR] No DICOM files found")
        return False


if __name__ == "__main__":
    test_load_single_file()