# Breast Cancer Subtype Classification - Hackathon Guide

## Quick Start for Hackathon Day

### 1. Clone and Setup (5 minutes)
```bash
git clone https://github.com/sukincornell/code_pink.git
cd code_pink

# Create environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install openpyxl  # For Excel metadata files
```

### 2. Test Connection (1 minute)
```bash
python -m src.s3_utils
```

### 3. Run Pipeline (10-30 minutes depending on scale)
```bash
# Quick test with 10 patients
python run_full_pipeline.py

# Or use the main runner for more control
python run_pipeline.py --sample 100 --slices
```

## Dataset Information

- **Source**: Duke Breast Cancer MRI (TCIA)
- **Access**: Public S3 bucket (no credentials needed)
- **Size**: ~200GB full dataset
- **Patients**: 924 total
- **Molecular Subtypes**:
  - 0 = Luminal-like (595 patients)
  - 1 = ER/PR+, HER2+ (104 patients)
  - 2 = HER2 (59 patients)
  - 3 = Triple Negative (164 patients)

## 🗂️ Output Files

After running the pipeline:

```
outputs/
├── manifest.csv                 # All series catalog
├── manifest_with_labels.csv     # With molecular subtypes
├── preprocessed_index.csv       # 3D volumes index
├── slices_index.csv            # 2D slices index
├── preprocessed/               # 3D volumes (128×192×192)
│   └── *.npy
└── slices/                     # 2D slices (192×192)
    └── *.npy
```

## Training a Model

### Load Data
```python
import pandas as pd
import numpy as np

# For 3D CNN
df = pd.read_csv('outputs/preprocessed_index.csv')
train_df = df[df['split'] == 'train']

X_train = np.array([np.load(f) for f in train_df['npy_path']])
y_train = train_df['tumor_subtype'].values

print(f"Training data shape: {X_train.shape}")
print(f"Labels: {np.unique(y_train)}")
```

### Simple 3D CNN Example
```python
import torch
import torch.nn as nn

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 48 * 48, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
```

## 🏃‍♂️ Scaling Up

### Process More Patients
Edit `run_full_pipeline.py` line 39:
```python
patients = [p for p in patients if 'Breast_MRI_' in p][:100]  # Change 20 to 100+
```

### Use Higher Resolution
Edit `src/config.py`:
```python
TARGET_SHAPE = (160, 224, 224)  # Increase from (128, 192, 192)
```

### Enable N4 Bias Correction
In preprocessing calls:
```python
processed = full_pipeline(vol, spacing, do_n4=True, ...)  # Set True
```

## Tips for Competition

1. **Start Small**: Test with 10-20 patients first
2. **Use 2D Slices**: Faster to train initially
3. **Data Augmentation**: Rotation, flipping, intensity shifts
4. **Ensemble Models**: Train multiple architectures
5. **Cross-Validation**: Use the 5 folds properly
6. **Class Imbalance**: Use weighted loss or oversampling

## Troubleshooting

### S3 Connection Issues
- Check internet connection
- Verify bucket is still public
- Try smaller sample size

### Memory Issues
- Reduce batch size
- Process fewer patients at once
- Use 2D slices instead of 3D volumes

### DICOM Errors
```bash
# Install additional codecs if needed
conda install -c conda-forge gdcm
```

## Performance Baselines

Expected results with basic models:
- Random: 25% accuracy (4 classes)
- 2D CNN: 40-50% accuracy
- 3D CNN: 50-60% accuracy
- Pre-trained + Fine-tuning: 60-70% accuracy
- Ensemble: 65-75% accuracy

## Winning Strategy

1. **Preprocess Early**: Run pipeline before hackathon if possible
2. **Multiple Models**: Train 2D and 3D models in parallel
3. **Feature Engineering**: Extract radiomic features
4. **Use Metadata**: Incorporate clinical features if available
5. **Optimize Inference**: Focus on accuracy AND speed

## Team Collaboration

```bash
# Share preprocessed data
tar -czf preprocessed_data.tar.gz outputs/

# Upload to shared location
# Team members can skip preprocessing and start training immediately
```

## Success Tips

Remember:
- Pipeline handles all DICOM complexity
- Data is pre-normalized and ready
- Focus on model architecture and training
- Collaborate with your team effectively