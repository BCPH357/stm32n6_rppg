"""
服務器端：量化 4D ONNX 模型為 INT8

用法 (在服務器上執行):
    cd /mnt/data_8T/ChenPinHao/server_training/
    conda activate rppg_training  # 或你的環境
    pip install onnxruntime onnx  # 如果還沒安裝
    python quantize_4d_model.py

輸入:
    models/rppg_4d_fp32.onnx

輸出:
    models/rppg_4d_int8.onnx  - INT8 量化模型 (用於 STM32)
"""

import onnx
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
from onnxruntime.quantization import CalibrationDataReader
import numpy as np
import torch
from pathlib import Path


class CalibrationDataReaderCustom(CalibrationDataReader):
    """校準數據讀取器"""
    def __init__(self, num_samples=200):
        self.num_samples = num_samples
        self.current_idx = 0

        # 生成隨機校準數據（模擬真實輸入分布）
        # 真實場景：應該從預處理好的數據集中採樣
        np.random.seed(42)
        self.calibration_data = []

        print(f"   Generating {num_samples} calibration samples...")
        for i in range(num_samples):
            # 4D 輸入: (1, 72, 36, 36)
            # 模擬正常化後的圖像數據 [0, 1]
            sample = np.random.rand(1, 72, 36, 36).astype(np.float32)
            self.calibration_data.append({'input': sample})

        print(f"   [OK] Calibration data ready: {len(self.calibration_data)} samples")

    def get_next(self):
        if self.current_idx >= self.num_samples:
            return None
        data = self.calibration_data[self.current_idx]
        self.current_idx += 1
        return data


def quantize_model_dynamic(input_path, output_path):
    """動態量化 (簡單但精度稍差)"""
    print("\n[Method 1] Dynamic Quantization...")

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8
    )

    print(f"   [OK] Saved: {output_path}")


def quantize_model_static(input_path, output_path, num_calibration_samples=200):
    """靜態量化 (需要校準數據，精度更好)"""
    print("\n[Method 2] Static Quantization (Recommended)...")

    # 創建校準數據讀取器
    calibration_data_reader = CalibrationDataReaderCustom(num_calibration_samples)

    # 靜態量化
    quantize_static(
        model_input=str(input_path),
        model_output=str(output_path),
        calibration_data_reader=calibration_data_reader
    )

    print(f"   [OK] Saved: {output_path}")


def verify_quantized_model(fp32_path, int8_path):
    """驗證量化模型精度"""
    print("\n[Verification] Comparing FP32 vs INT8...")

    import onnxruntime as ort

    # 載入模型
    sess_fp32 = ort.InferenceSession(str(fp32_path))
    sess_int8 = ort.InferenceSession(str(int8_path))

    # 測試數據
    num_tests = 10
    errors = []

    for i in range(num_tests):
        x = np.random.rand(1, 72, 36, 36).astype(np.float32)

        y_fp32 = sess_fp32.run(None, {'input': x})[0][0, 0]
        y_int8 = sess_int8.run(None, {'input': x})[0][0, 0]

        error = abs(y_fp32 - y_int8)
        errors.append(error)

    mean_error = np.mean(errors)
    max_error = np.max(errors)

    print(f"   Mean Absolute Error: {mean_error:.4f} BPM")
    print(f"   Max Absolute Error:  {max_error:.4f} BPM")

    if mean_error < 5.0:
        print(f"   [OK] Quantization quality: GOOD (MAE < 5 BPM)")
    elif mean_error < 10.0:
        print(f"   [WARNING] Quantization quality: ACCEPTABLE (MAE < 10 BPM)")
    else:
        print(f"   [ERROR] Quantization quality: POOR (MAE >= 10 BPM)")

    return mean_error, max_error


def main():
    print("="*70)
    print("Quantize 4D Model to INT8 for STM32")
    print("="*70)

    # 路徑
    models_dir = Path("models")
    fp32_path = models_dir / "rppg_4d_fp32.onnx"
    int8_dynamic_path = models_dir / "rppg_4d_int8_dynamic.onnx"
    int8_static_path = models_dir / "rppg_4d_int8.onnx"

    # 檢查輸入
    if not fp32_path.exists():
        print(f"\n[ERROR] FP32 model not found: {fp32_path}")
        print("   Please run convert_to_4d_for_stm32.py first")
        return

    print(f"\n[Input] {fp32_path}")
    print(f"   Size: {fp32_path.stat().st_size / 1024:.2f} KB")

    # 方法 1: 動態量化（快速但精度稍差）
    # quantize_model_dynamic(fp32_path, int8_dynamic_path)

    # 方法 2: 靜態量化（推薦，精度更好）
    quantize_model_static(fp32_path, int8_static_path, num_calibration_samples=200)

    # 驗證
    mean_err, max_err = verify_quantized_model(fp32_path, int8_static_path)

    # 完成
    print("\n" + "="*70)
    print("[SUCCESS] Quantization Complete!")
    print("="*70)

    print(f"\nModel sizes:")
    print(f"   FP32: {fp32_path.stat().st_size / 1024:.2f} KB")
    print(f"   INT8: {int8_static_path.stat().st_size / 1024:.2f} KB")
    compression_ratio = fp32_path.stat().st_size / int8_static_path.stat().st_size
    print(f"   Compression: {compression_ratio:.2f}x")

    print(f"\nQuantization error:")
    print(f"   Mean: {mean_err:.2f} BPM")
    print(f"   Max:  {max_err:.2f} BPM")

    print(f"\nDownload to local:")
    print(f"   scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/models/rppg_4d_int8.onnx D:\\MIAT\\rppg\\")

    print(f"\nImport into STM32CubeMX:")
    print(f"   1. Open X-CUBE-AI")
    print(f"   2. Add network: rppg_4d_int8.onnx")
    print(f"   3. Set Optimization: Time (O2) or Default (O1)")
    print(f"   4. AVOID: Balanced (O3) - causes buffer overlap issues")
    print(f"   5. Analyze and generate code")
    print("="*70)


if __name__ == "__main__":
    main()
