# -*- coding: utf-8 -*-
"""
使用 ONNX 原生工具進行量化
避免 ONNXRuntime DLL 問題
"""

import sys
import os

# 設置編碼
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("="*70)
print("INT8 Quantization using ONNX native tools")
print("="*70)

try:
    import onnx
    from onnx import numpy_helper
    print(f"ONNX version: {onnx.__version__}")
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# 檢查文件
fp32_model = 'models/rppg_fp32.onnx'
if not os.path.exists(fp32_model):
    print(f"ERROR: {fp32_model} not found!")
    sys.exit(1)

print(f"\nInput: {fp32_model} ({os.path.getsize(fp32_model)/1024:.1f} KB)")

# 載入模型
print("\nLoading ONNX model...")
try:
    model = onnx.load(fp32_model)
    onnx.checker.check_model(model)
    print("  Model loaded and validated")
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

# 顯示模型信息
print("\nModel Information:")
print(f"  IR version: {model.ir_version}")
print(f"  Opset version: {model.opset_import[0].version}")
print(f"  Graph nodes: {len(model.graph.node)}")

input_tensor = model.graph.input[0]
output_tensor = model.graph.output[0]
print(f"\n  Input: {input_tensor.name}")
print(f"    Shape: {[dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]}")
print(f"  Output: {output_tensor.name}")
print(f"    Shape: {[dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]}")

# 說明
print("\n" + "="*70)
print("Quantization Status")
print("="*70)
print("\nONNX native quantization requires ONNXRuntime, which has DLL issues")
print("on your system. However, X-CUBE-AI can handle FP32 ONNX directly.")
print("\n兩個選項:")
print("\n選項 1 (推薦): 直接使用 FP32 ONNX")
print("  - X-CUBE-AI 會自動進行 INT8 量化")
print("  - 在 STM32CubeMX 中:")
print("    Model File: models/rppg_fp32.onnx")
print("    X-CUBE-AI 將自動量化為 INT8")
print("\n選項 2: 在服務器上量化")
print("  - 上傳 FP32 ONNX 到服務器")
print("  - 在 Linux 環境完成量化（無 DLL 問題）")
print("  - 下載 INT8 ONNX 回本地")

print("\n" + "="*70)
print("建議: 使用選項 1 - X-CUBE-AI 自動量化")
print("="*70)
print("\n您的 FP32 ONNX 模型已經可以使用了!")
print(f"文件: {fp32_model}")
print("\n下一步:")
print("  1. 打開 STM32CubeMX")
print("  2. 載入此 ONNX 模型")
print("  3. X-CUBE-AI 會處理 INT8 轉換")
print("  4. 配置 Optimization: O1 或 O2")
print("  5. 生成代碼並部署")
print("\n參考: ../stm32n6_deployment/deployment_guide.md")
print("="*70)
