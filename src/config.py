"""Configuration file for the Breast Cancer Subtypes challenge."""

# S3 paths (public bucket, no credentials needed)
S3_BUCKET = "s3://cu-qatar-hackathon-2025/Duke-Breast-Cancer-MRI"
META_DIR = "s3://cu-qatar-hackathon-2025/Duke-Breast-Cancer-MRI/a_metadata"

# Local paths
DATA_DIR = "data"
OUTPUTS_DIR = "outputs"

# Preprocessing parameters
TARGET_SPACING_MM = 1.5  # Isotropic spacing in mm
TARGET_SHAPE = (128, 192, 192)  # (Z, Y, X) for 3D volumes
CLIP_PERCENTILES = (1, 99)  # For robust intensity normalization

# Training parameters
RANDOM_SEED = 42
N_FOLDS = 5
SUBSET_SIZE = 50  # Number of patients per class for initial subset