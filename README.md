# code_pink - Breast Cancer Tumor Subtypes Classification
Cornell Precision Medicine Hackathon at Doha

This project provides a complete preprocessing pipeline for the Duke Breast Cancer MRI dataset.

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -y -n bc_subtypes python=3.10
conda activate bc_subtypes

# Install dependencies
pip install -r requirements.txt

# For GPU support (optional):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Test Connection

```bash
# Test S3 connection
python -m src.s3_utils
```

### 3. Run Pipeline

```bash
# Quick test with small sample (recommended first)
python run_pipeline.py --sample 100

# Full preprocessing
python run_pipeline.py

# With N4 bias correction (better quality, slower)
python run_pipeline.py --n4

# Also extract 2D slices
python run_pipeline.py --slices
```

## Project Structure

```
bc_subtypes/
├── data/               # Local cache (created when needed)
├── outputs/            # Preprocessed data
│   ├── series_manifest.csv      # All DICOM series
│   ├── subset_manifest.csv      # Selected subset with labels
│   ├── preprocessed_index.csv   # Preprocessed volumes index
│   ├── preprocessed/*.npy       # 3D volumes (128×192×192)
│   └── slices/*.npy             # 2D slices (if extracted)
├── src/
│   ├── config.py       # Configuration
│   ├── s3_utils.py     # S3 access utilities
│   ├── dicom_utils.py  # DICOM loading
│   ├── preprocess.py   # Preprocessing pipeline
│   ├── build_manifest.py    # Manifest builder
│   └── make_subset.py       # Subset creation
├── requirements.txt
├── run_pipeline.py     # Main runner
└── README.md
```

## Pipeline Steps

1. **Build Manifest**: Scan S3 bucket and catalog all DICOM series
2. **Load Metadata**: Load tumor subtype labels from metadata files
3. **Create Subset**: Select balanced subset for development
4. **Preprocess Volumes**:
   - Robust intensity normalization (percentile clipping + z-score)
   - Optional N4 bias field correction
   - Resample to 1.5mm isotropic spacing
   - Center crop/pad to 128×192×192
5. **Extract Slices** (optional): Extract center slices for 2D models

## Key Features

- **Direct S3 access** - No need to download entire dataset
- **Robust preprocessing** - Handles MRI intensity variations
- **Patient-level splits** - Prevents data leakage
- **Memory efficient** - Processes one volume at a time
- **Resumable** - Can skip completed steps

## Data Access

The dataset is publicly available at:
- Main data: `s3://cu-qatar-hackathon-2025/Duke-Breast-Cancer-MRI/`
- Metadata: `s3://cu-qatar-hackathon-2025/Duke-Breast-Cancer-MRI/a_metadata/`

No AWS credentials needed - the bucket is public.

## For SageMaker

When running on SageMaker during the hackathon:

1. Use the same environment setup
2. Start with `ml.g4dn.xlarge` instance
3. Data access works the same (direct from S3)
4. For faster training, cache subset to local SageMaker storage

## Troubleshooting

### DICOM Compression Issues
If you see "transfer syntax not supported":
```bash
# Install additional codecs
conda install -c conda-forge gdcm
```

### Memory Issues
- Reduce batch processing size with `--sample` flag
- Process smaller target dimensions in `config.py`
- Use `--skip-manifest` if manifest already exists

### S3 Access Issues
- Check internet connection
- Verify the bucket is still public
- Try with smaller sample size first

## Next Steps

After preprocessing:

1. **Load preprocessed data**:
```python
import pandas as pd
import numpy as np

# Load index
df = pd.read_csv("outputs/preprocessed_index.csv")

# Load a volume
vol = np.load(df.iloc[0]["npy_path"])
print(f"Shape: {vol.shape}, Range: [{vol.min():.2f}, {vol.max():.2f}]")
```

2. **Train models** using the preprocessed volumes
3. **Scale up** by increasing subset size and resolution

## Support

For hackathon support, refer to the official challenge page:
https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/
