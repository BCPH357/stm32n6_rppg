# -*- coding: utf-8 -*-
"""
只執行 INT8 量化步驟
前提: rppg_fp32.onnx 和 calibration_data_synthetic.pt 已存在
"""

import sys
import os
import numpy as np
import torch

# 設置編碼
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print("="*70)
print("INT8 Quantization Only")
print("="*70)

# 檢查輸入文件
fp32_model = 'models/rppg_fp32.onnx'
cal_data = 'calibration_data_synthetic.pt'

if not os.path.exists(fp32_model):
    print(f"ERROR: {fp32_model} not found!")
    sys.exit(1)

if not os.path.exists(cal_data):
    print(f"ERROR: {cal_data} not found!")
    sys.exit(1)

print(f"FP32 Model: {fp32_model} ({os.path.getsize(fp32_model)/1024:.1f} KB)")
print(f"Calibration: {cal_data} ({os.path.getsize(cal_data)/(1024**2):.1f} MB)")

# 導入量化工具
try:
    print("\nImporting ONNX Runtime quantization tools...")
    from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader
    print("  OK: Import successful")
except Exception as e:
    print(f"  ERROR: Failed to import - {e}")
    sys.exit(1)

# 校準數據讀取器
class SimpleCalibrationReader(CalibrationDataReader):
    def __init__(self, data_file):
        data = torch.load(data_file)
        self.samples = data['samples'].numpy().astype(np.float32)
        self.num_samples = data['num_samples']
        self.current_idx = 0
        print(f"  Loaded {self.num_samples} samples")

    def get_next(self):
        if self.current_idx >= self.num_samples:
            return None
        sample = self.samples[self.current_idx:self.current_idx+1]
        self.current_idx += 1
        if self.current_idx % 20 == 0:
            print(f"    Calibrating... {self.current_idx}/{self.num_samples}")
        return {'input': sample}

    def rewind(self):
        self.current_idx = 0

# 創建讀取器
print("\nCreating calibration data reader...")
try:
    reader = SimpleCalibrationReader(cal_data)
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

# 執行量化
int8_output = 'models/rppg_int8_qdq.onnx'
print(f"\nQuantizing to INT8...")
print(f"  Output: {int8_output}")
print(f"  Format: QDQ (Quantize-DeQuantize)")
print(f"  Mode: per-channel")

try:
    quantize_static(
        fp32_model,
        int8_output,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        optimize_model=False,
        use_external_data_format=False
    )
    print("  OK: Quantization completed")
except Exception as e:
    print(f"  ERROR: Quantization failed - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 比較大小
fp32_size = os.path.getsize(fp32_model) / 1024
int8_size = os.path.getsize(int8_output) / 1024
ratio = fp32_size / int8_size

print("\n" + "="*70)
print("Quantization Results")
print("="*70)
print(f"FP32 Model:  {fp32_size:.1f} KB")
print(f"INT8 Model:  {int8_size:.1f} KB")
print(f"Compression: {ratio:.2f}x")
print("\n" + "="*70)
print("SUCCESS! INT8 model ready for X-CUBE-AI")
print("="*70)
print(f"\nOutput file: {int8_output}")
print("\nNext steps:")
print("  1. Use this ONNX in STM32CubeMX X-CUBE-AI")
print("  2. Set Optimization: O1 or O2 (avoid O3)")
print("  3. Generate code and deploy to STM32N6")
print("="*70)
