#!/bin/bash

# Breast Cancer Subtypes - Environment Setup Script

echo "======================================"
echo "Breast Cancer Subtypes Pipeline Setup"
echo "======================================"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed. Please install Miniconda or Anaconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo ""
echo "Creating conda environment 'bc_subtypes' with Python 3.10..."
conda create -y -n bc_subtypes python=3.10

# Activate environment
echo ""
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate bc_subtypes

# Install requirements
echo ""
echo "Installing required packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Optional: Install PyTorch for GPU
echo ""
read -p "Do you want to install PyTorch with CUDA support? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing PyTorch with CUDA 11.8 support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch CPU version..."
    pip install torch torchvision
fi

# Test installation
echo ""
echo "Testing installation..."
python -c "
import sys
import pydicom
import s3fs
import SimpleITK
import numpy as np
import pandas as pd
print('✓ All core packages imported successfully!')
print(f'✓ Python version: {sys.version}')
print(f'✓ NumPy version: {np.__version__}')
print(f'✓ Pandas version: {pd.__version__}')
print(f'✓ PyDICOM version: {pydicom.__version__}')
"

# Test S3 connection
echo ""
echo "Testing S3 connection..."
python -m src.s3_utils

echo ""
echo "======================================"
echo "Setup complete!"
echo "======================================"
echo ""
echo "To activate the environment in the future:"
echo "  conda activate bc_subtypes"
echo ""
echo "To run a quick test:"
echo "  python run_pipeline.py --sample 100"
echo ""
echo "For full preprocessing:"
echo "  python run_pipeline.py"