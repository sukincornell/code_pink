"""Preprocessing utilities for MRI volumes."""

import numpy as np
import SimpleITK as sitk
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def robust_clip_zscore(vol: np.ndarray, lower_q: float = 1, upper_q: float = 99) -> np.ndarray:
    """Apply robust intensity normalization with percentile clipping and z-score.

    Args:
        vol: Input volume
        lower_q: Lower percentile for clipping (default: 1)
        upper_q: Upper percentile for clipping (default: 99)

    Returns:
        Normalized volume with mean=0, std=1
    """
    # Calculate percentiles
    lo = np.percentile(vol, lower_q)
    hi = np.percentile(vol, upper_q)

    # Clip intensities
    vol = np.clip(vol, lo, hi)

    # Z-score normalization
    mean = vol.mean()
    std = vol.std() + 1e-6  # Add small epsilon to prevent division by zero
    vol = (vol - mean) / std

    return vol


def minmax_scale(vol: np.ndarray, lower_q: float = 1, upper_q: float = 99) -> np.ndarray:
    """Apply min-max scaling to [0, 1] range with percentile clipping.

    Args:
        vol: Input volume
        lower_q: Lower percentile for clipping (default: 1)
        upper_q: Upper percentile for clipping (default: 99)

    Returns:
        Scaled volume in range [0, 1]
    """
    # Calculate percentiles
    lo = np.percentile(vol, lower_q)
    hi = np.percentile(vol, upper_q)

    # Clip and scale
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo + 1e-6)

    return vol


def n4_bias_correct(vol: np.ndarray, spacing_mm: Tuple[float, float, float]) -> np.ndarray:
    """Apply N4 bias field correction to MRI volume.

    Args:
        vol: Input volume (Z, Y, X)
        spacing_mm: Voxel spacing in mm (Z, Y, X)

    Returns:
        Bias-corrected volume
    """
    try:
        # Convert to SimpleITK image
        img = sitk.GetImageFromArray(vol)
        # SimpleITK uses (X, Y, Z) ordering for spacing
        img.SetSpacing((spacing_mm[2], spacing_mm[1], spacing_mm[0]))

        # Create mask using Otsu thresholding
        mask = sitk.OtsuThreshold(img, 0, 1)

        # Configure N4 bias correction
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])

        # Apply correction
        corrected = corrector.Execute(img, mask)

        # Convert back to numpy
        return sitk.GetArrayFromImage(corrected)

    except Exception as e:
        logger.warning(f"N4 bias correction failed: {e}. Returning original volume.")
        return vol


def resample_isotropic(vol: np.ndarray,
                       spacing_mm: Tuple[float, float, float],
                       out_mm: float = 1.5,
                       interpolator: str = "linear") -> np.ndarray:
    """Resample volume to isotropic spacing.

    Args:
        vol: Input volume (Z, Y, X)
        spacing_mm: Original voxel spacing in mm (Z, Y, X)
        out_mm: Target isotropic spacing in mm (default: 1.5)
        interpolator: Interpolation method ("linear" or "nearest")

    Returns:
        Resampled volume with isotropic spacing
    """
    # Convert to SimpleITK image
    img = sitk.GetImageFromArray(vol)
    # SimpleITK uses (X, Y, Z) ordering
    img.SetSpacing((spacing_mm[2], spacing_mm[1], spacing_mm[0]))

    # Calculate new size
    new_spacing = (out_mm, out_mm, out_mm)
    original_size = img.GetSize()
    original_spacing = img.GetSpacing()

    new_size = [
        int(round(original_size[i] * original_spacing[i] / new_spacing[i]))
        for i in range(3)
    ]

    # Choose interpolator
    if interpolator == "nearest":
        interp = sitk.sitkNearestNeighbor
    else:
        interp = sitk.sitkLinear

    # Resample
    resampled = sitk.Resample(
        img,
        new_size,
        sitk.Transform(),
        interp,
        img.GetOrigin(),
        new_spacing,
        img.GetDirection(),
        0.0,
        img.GetPixelID(),
    )

    # Convert back to numpy (Z, Y, X)
    return sitk.GetArrayFromImage(resampled)


def center_crop_or_pad(vol: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    """Center crop or pad volume to target size.

    Args:
        vol: Input volume (Z, Y, X)
        target: Target size (Z, Y, X)

    Returns:
        Cropped or padded volume
    """
    z, y, x = vol.shape
    tz, ty, tx = target

    # Create output volume (zero-padded)
    out = np.zeros(target, dtype=vol.dtype)

    # Calculate padding/cropping indices
    # For padding
    z0 = max(0, (tz - z) // 2)
    y0 = max(0, (ty - y) // 2)
    x0 = max(0, (tx - x) // 2)

    # For cropping
    zs = max(0, (z - tz) // 2)
    ys = max(0, (y - ty) // 2)
    xs = max(0, (x - tx) // 2)

    # Copy data
    out[z0:z0 + min(z, tz),
        y0:y0 + min(y, ty),
        x0:x0 + min(x, tx)] = vol[zs:zs + min(z, tz),
                                  ys:ys + min(y, ty),
                                  xs:xs + min(x, tx)]

    return out


def full_pipeline(vol: np.ndarray,
                 spacing_mm: Tuple[float, float, float],
                 do_n4: bool = True,
                 normalize_method: str = "zscore",
                 out_mm: float = 1.5,
                 target: Tuple[int, int, int] = (128, 192, 192)) -> np.ndarray:
    """Complete preprocessing pipeline for MRI volume.

    Args:
        vol: Input volume
        spacing_mm: Original voxel spacing (Z, Y, X)
        do_n4: Whether to apply N4 bias correction (compute-intensive)
        normalize_method: "zscore" or "minmax"
        out_mm: Target isotropic spacing in mm
        target: Target volume size (Z, Y, X)

    Returns:
        Preprocessed volume as float32
    """
    logger.info(f"Preprocessing volume: shape={vol.shape}, spacing={spacing_mm}")

    # Step 1: Initial intensity normalization
    if normalize_method == "zscore":
        vol = robust_clip_zscore(vol)
    else:
        vol = minmax_scale(vol)

    # Step 2: N4 bias correction (optional, compute-intensive)
    if do_n4:
        try:
            vol = n4_bias_correct(vol, spacing_mm)
            logger.info("Applied N4 bias correction")
        except Exception as e:
            logger.warning(f"N4 correction failed: {e}")

    # Step 3: Resample to isotropic spacing
    vol = resample_isotropic(vol, spacing_mm, out_mm=out_mm)
    logger.info(f"Resampled to {out_mm}mm isotropic: shape={vol.shape}")

    # Step 4: Center crop or pad to target size
    vol = center_crop_or_pad(vol, target=target)
    logger.info(f"Cropped/padded to target: shape={vol.shape}")

    # Convert to float32 for efficiency
    return vol.astype(np.float32)


def extract_2d_slices(vol: np.ndarray,
                      num_slices: int = 3,
                      method: str = "center") -> np.ndarray:
    """Extract 2D slices from 3D volume for 2D models.

    Args:
        vol: 3D volume (Z, Y, X)
        num_slices: Number of slices to extract
        method: "center" for middle slices, "uniform" for evenly spaced

    Returns:
        Array of 2D slices (num_slices, Y, X)
    """
    z = vol.shape[0]

    if method == "center":
        # Extract center slices
        mid = z // 2
        start = max(0, mid - num_slices // 2)
        end = min(z, start + num_slices)
        indices = list(range(start, end))
    else:
        # Extract uniformly spaced slices
        indices = np.linspace(0, z - 1, num_slices, dtype=int)

    return vol[indices]


if __name__ == "__main__":
    # Test preprocessing on synthetic data
    print("Testing preprocessing pipeline...")

    # Create synthetic volume
    vol = np.random.randn(100, 256, 256).astype(np.float32)
    spacing = (2.0, 1.0, 1.0)  # Non-isotropic spacing

    # Run pipeline
    processed = full_pipeline(
        vol, spacing,
        do_n4=False,  # Skip N4 for synthetic data
        normalize_method="zscore",
        out_mm=1.5,
        target=(128, 192, 192)
    )

    print(f"[OK] Input shape: {vol.shape}")
    print(f"[OK] Output shape: {processed.shape}")
    print(f"[OK] Output dtype: {processed.dtype}")
    print(f"[OK] Output range: [{processed.min():.2f}, {processed.max():.2f}]")