#!/bin/bash

# rPPG Training Environment Setup Script
# For A6000 Server with Python 3.12.3

echo "=============================="
echo "rPPG Training Environment Setup"
echo "=============================="

# 创建目录结构
echo "Creating directory structure..."
mkdir -p raw_data
mkdir -p data
mkdir -p checkpoints
mkdir -p logs

echo "✅ Directory structure created"
echo ""

# 创建 conda 环境
echo "Creating conda environment..."
conda env create -f environment.yml

# 激活环境
echo ""
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rppg_training

# 验证安装
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
python -c "import cv2, gdown; print(f'OpenCV: {cv2.__version__}'); print('gdown: installed')"

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "Directory structure:"
echo "  raw_data/        - For raw UBFC dataset"
echo "  data/            - For preprocessed data"
echo "  checkpoints/     - For trained models"
echo "  logs/            - For training logs"
echo ""
echo "Next steps:"
echo "  1. Download dataset:    bash download_ubfc.sh"
echo "  2. Preprocess data:     python preprocess_data.py --dataset ubfc --raw_data raw_data --output data"
echo "  3. Start training:      bash run_training.sh"
echo ""
echo "Or run all steps at once:"
echo "  bash run_all.sh"
