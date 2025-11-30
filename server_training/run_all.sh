#!/bin/bash
################################################################################
# rPPG Full Pipeline - One-Click Execution
# Runs complete workflow: Download → Preprocess → Train
################################################################################

set -e  # Exit on error

echo "========================================"
echo "rPPG Full Pipeline Executor"
echo "========================================"
echo ""
echo "This script will:"
echo "  1. Download UBFC-rPPG dataset (if not exists)"
echo "  2. Preprocess data (if not exists)"
echo "  3. Train the model"
echo ""
echo "Estimated total time: 4-6 hours"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate conda environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rppg_training

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate 'rppg_training' environment"
    echo ""
    echo "Please run setup first:"
    echo "  bash setup_env.sh"
    exit 1
fi

echo "✅ Environment activated: rppg_training"
echo ""

################################################################################
# Step 1: Download Dataset
################################################################################
echo "========================================"
echo "Step 1/3: Download UBFC-rPPG Dataset"
echo "========================================"
echo ""

if [ -d "raw_data/UBFC-rPPG" ] && [ "$(find raw_data/UBFC-rPPG -type d -name 'subject*' | wc -l)" -ge 40 ]; then
    echo "✅ UBFC-rPPG dataset already exists"
    echo "   Location: raw_data/UBFC-rPPG"
    echo "   Subjects: $(find raw_data/UBFC-rPPG -type d -name 'subject*' | wc -l)"
    echo "   Skipping download..."
else
    echo "📥 Downloading UBFC-rPPG dataset..."
    echo "   Estimated time: 30-60 minutes"
    echo ""

    bash download_ubfc.sh

    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Error: Dataset download failed"
        echo "Please check the error messages above and try again"
        exit 1
    fi

    echo ""
    echo "✅ Dataset download complete!"
fi

echo ""

################################################################################
# Step 2: Validate Raw Data
################################################################################
echo "========================================"
echo "Step 2/3: Validate & Preprocess Data"
echo "========================================"
echo ""

# Validate raw data first
echo "🔍 Validating raw dataset..."
python validate_data.py --mode raw

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  Warning: Raw data validation failed"
    echo "   Continuing anyway, but there may be issues during preprocessing..."
fi

echo ""

# Preprocess data
if [ -f "data/ubfc_processed.pt" ]; then
    echo "✅ Preprocessed data already exists"
    echo "   Location: data/ubfc_processed.pt"
    echo "   Size: $(du -h data/ubfc_processed.pt | cut -f1)"
    echo "   Skipping preprocessing..."

    # Validate preprocessed data
    echo ""
    echo "🔍 Validating preprocessed data..."
    python validate_data.py --mode preprocessed

else
    echo "⚙️  Preprocessing dataset..."
    echo "   Estimated time: 2-3 hours (CPU intensive)"
    echo ""

    python preprocess_data.py \
        --dataset ubfc \
        --raw_data raw_data \
        --output data \
        --window_size 8 \
        --stride 1

    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Error: Data preprocessing failed"
        echo "Please check the error messages above"
        exit 1
    fi

    echo ""
    echo "✅ Data preprocessing complete!"
    echo "   Output: data/ubfc_processed.pt"
    echo "   Size: $(du -h data/ubfc_processed.pt | cut -f1)"

    # Validate preprocessed data
    echo ""
    echo "🔍 Validating preprocessed data..."
    python validate_data.py --mode preprocessed
fi

echo ""

################################################################################
# Step 3: Train Model
################################################################################
echo "========================================"
echo "Step 3/3: Train Model"
echo "========================================"
echo ""

echo "🚀 Starting model training..."
echo "   Estimated time: 1.5-2 hours (GPU)"
echo ""

bash run_training.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Error: Training failed"
    echo "Please check the error messages and logs above"
    exit 1
fi

################################################################################
# Summary
################################################################################
echo ""
echo "========================================"
echo "✅ Full Pipeline Complete!"
echo "========================================"
echo ""
echo "Outputs:"
echo "  📊 Raw Data:        raw_data/UBFC-rPPG/"
echo "  💾 Processed Data:  data/ubfc_processed.pt"
echo "  🎯 Trained Model:   checkpoints/best_model.pth"
echo "  📝 Training Logs:   logs/"
echo ""
echo "Model Info:"
if [ -f "checkpoints/best_model.pth" ]; then
    echo "  Size: $(du -h checkpoints/best_model.pth | cut -f1)"
    echo "  Location: $(pwd)/checkpoints/best_model.pth"
fi

if [ -f "checkpoints/train_history.json" ]; then
    echo ""
    echo "Training History:"
    python -c "
import json
with open('checkpoints/train_history.json', 'r') as f:
    history = json.load(f)
    print(f\"  Best Validation Loss: {min(history.get('val_loss', [999])):.4f}\")
    print(f\"  Total Epochs: {len(history.get('train_loss', []))}\")
" 2>/dev/null || echo "  (Unable to parse training history)"
fi

echo ""
echo "Next steps:"
echo "  1. Download model: scp username@server:$(pwd)/checkpoints/best_model.pth ."
echo "  2. Convert to ONNX format"
echo "  3. Deploy to STM32N6 using X-CUBE-AI"
echo ""
