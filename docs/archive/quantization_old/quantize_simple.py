# -*- coding: utf-8 -*-
"""
最簡化的量化腳本 - 不使用校準數據
使用 ONNX Runtime 的內建校準
"""

import sys
import os

print("Simple INT8 Quantization")
print("="*60)

# 檢查文件
fp32 = 'models/rppg_fp32.onnx'
int8 = 'models/rppg_int8_simple.onnx'

if not os.path.exists(fp32):
    print(f"Error: {fp32} not found")
    sys.exit(1)

print(f"Input: {fp32}")

# 嘗試導入
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    print("Using dynamic quantization (no calibration needed)...")

    # 動態量化 - 不需要校準數據
    quantize_dynamic(
        model_input=fp32,
        model_output=int8,
        weight_type=QuantType.QInt8
    )

    print(f"\nSuccess!")
    print(f"Output: {int8}")
    print(f"Size: {os.path.getsize(int8)/1024:.1f} KB")
    print("\nNote: This is dynamic quantization.")
    print("For static (better accuracy), use X-CUBE-AI's built-in quantization.")

except Exception as e:
    print(f"\nQuantization failed: {e}")
    print("\nRecommendation: Use FP32 ONNX directly with X-CUBE-AI")
    print("X-CUBE-AI will handle INT8 quantization automatically.")
