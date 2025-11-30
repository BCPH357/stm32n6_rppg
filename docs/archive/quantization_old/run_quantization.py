# -*- coding: utf-8 -*-
"""
一鍵運行完整量化流程
避免編碼和subprocess問題
"""

import torch
import torch.onnx
import os
import sys
import numpy as np

# 設置輸出編碼
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*70)
print("rPPG Multi-ROI INT8 Quantization - Complete Workflow")
print("="*70)

# ========== STEP 1: Export FP32 ONNX ==========
print("\n[STEP 1/3] Exporting FP32 ONNX Model...")
print("-"*70)

try:
    from server_training.model import UltraLightRPPG

    checkpoint_path = '../webapp/models/best_model.pth'
    fp32_output = 'models/rppg_fp32.onnx'

    print(f"Loading model from: {checkpoint_path}")
    model = UltraLightRPPG(window_size=8, num_rois=3)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"  Loaded from epoch: {epoch}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Test forward pass
    dummy_input = torch.randn(1, 8, 3, 36, 36, 3)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"  Test output: {output.item():.2f} BPM")

    # Export
    os.makedirs('models', exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        fp32_output,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        verbose=False
    )

    file_size = os.path.getsize(fp32_output) / 1024
    print(f"  Exported: {fp32_output} ({file_size:.1f} KB)")
    print("[OK] STEP 1 completed")

except Exception as e:
    print(f"[ERROR] STEP 1 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ========== STEP 2: Create Synthetic Calibration Data ==========
print("\n[STEP 2/3] Creating Synthetic Calibration Data...")
print("-"*70)

try:
    # 創建合成校準數據（因為沒有真實訓練數據）
    print("  Note: Using synthetic data (no training data available)")
    print("  Creating 100 synthetic samples...")

    calibration_data = {
        'samples': torch.randn(100, 8, 3, 36, 36, 3),  # 100 samples
        'labels': torch.FloatTensor(np.random.uniform(60, 120, 100)),  # Random HR 60-120
        'num_samples': 100
    }

    cal_file = 'calibration_data_synthetic.pt'
    torch.save(calibration_data, cal_file)

    file_size = os.path.getsize(cal_file) / (1024**2)
    print(f"  Created: {cal_file} ({file_size:.1f} MB)")
    print("[OK] STEP 2 completed")

except Exception as e:
    print(f"[ERROR] STEP 2 failed: {e}")
    sys.exit(1)

# ========== STEP 3: INT8 Quantization ==========
print("\n[STEP 3/3] Running INT8 Quantization...")
print("-"*70)

try:
    from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader
    import onnxruntime as ort

    # Calibration data reader
    class SyntheticCalibrationDataReader(CalibrationDataReader):
        def __init__(self, data_path):
            data = torch.load(data_path)
            self.samples = data['samples'].numpy().astype(np.float32)
            self.num_samples = data['num_samples']
            self.current_idx = 0
            print(f"  Loaded {self.num_samples} calibration samples")

        def get_next(self):
            if self.current_idx >= self.num_samples:
                return None
            sample = self.samples[self.current_idx:self.current_idx+1]
            self.current_idx += 1
            return {'input': sample}

        def rewind(self):
            self.current_idx = 0

    calibration_reader = SyntheticCalibrationDataReader('calibration_data_synthetic.pt')

    int8_output = 'models/rppg_int8_qdq.onnx'

    print("  Quantizing to INT8 (QDQ format)...")
    quantize_static(
        fp32_output,
        int8_output,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        optimize_model=False,
        use_external_data_format=False
    )

    fp32_size = os.path.getsize(fp32_output) / 1024
    int8_size = os.path.getsize(int8_output) / 1024
    ratio = fp32_size / int8_size

    print(f"  FP32 model: {fp32_size:.1f} KB")
    print(f"  INT8 model: {int8_size:.1f} KB")
    print(f"  Compression: {ratio:.2f}x")
    print("[OK] STEP 3 completed")

except Exception as e:
    print(f"[ERROR] STEP 3 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ========== Summary ==========
print("\n" + "="*70)
print("Quantization Workflow Completed Successfully!")
print("="*70)
print(f"\nGenerated files:")
print(f"  1. FP32 ONNX:  {fp32_output}")
print(f"  2. INT8 ONNX:  {int8_output} <- Ready for X-CUBE-AI")
print(f"\nNext steps:")
print(f"  1. Use {int8_output} in STM32CubeMX")
print(f"  2. Configure X-CUBE-AI (Optimization: O1 or O2)")
print(f"  3. Generate code and deploy to STM32N6")
print(f"\nRefer to: ../stm32n6_deployment/deployment_guide.md")
print("="*70)
