"""
服務器端：量化 4D ONNX 模型為 INT8（改進版）

基於之前成功的量化方案：
1. 使用真實訓練數據進行校準（非隨機數據）
2. 分層採樣確保各心率範圍都有代表
3. QDQ 格式量化（更精確）
4. Per-channel 量化

用法 (在服務器上執行):
    cd /mnt/data_8T/ChenPinHao/server_training/
    conda activate rppg_training
    python quantize_4d_model_v2.py

輸入:
    models/rppg_4d_fp32.onnx
    data/ubfc_processed.pt (用於校準數據)

輸出:
    models/rppg_4d_int8_qdq.onnx  - INT8 量化模型 (用於 STM32)
"""

import onnx
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader
import onnxruntime as ort
import numpy as np
import torch
from pathlib import Path


class RPPG4DCalibrationDataReader(CalibrationDataReader):
    """
    4D rPPG 校準數據讀取器

    使用真實訓練數據，並進行分層採樣確保涵蓋各心率範圍
    """
    def __init__(self, data_path, num_samples=200):
        """
        初始化校準數據讀取器

        Args:
            data_path: ubfc_processed.pt 路徑
            num_samples: 校準樣本數量
        """
        print(f"\n[Calibration Data Reader]")
        print(f"   Loading from: {data_path}")

        # 載入完整數據
        data = torch.load(data_path)
        samples_6d = data['samples']  # (N, 8, 3, 36, 36, 3)
        labels = data['labels']       # (N,)

        print(f"   Total samples: {len(samples_6d)}")
        print(f"   HR range: {labels.min():.2f} - {labels.max():.2f} BPM")

        # 分層採樣（確保各心率範圍都有代表）
        print(f"   Stratified sampling across HR ranges...")
        hr_bins = np.digitize(labels.numpy(), bins=[40, 60, 80, 100, 120, 160])

        selected_indices = []
        bin_names = ['40-60', '60-80', '80-100', '100-120', '120-160', '>160']

        for bin_id in range(1, 7):
            bin_indices = np.where(hr_bins == bin_id)[0]
            if len(bin_indices) > 0:
                # 每個 bin 取約 30-35 samples
                n = min(35, len(bin_indices))
                selected = np.random.choice(bin_indices, n, replace=False)
                selected_indices.extend(selected.tolist())
                print(f"     Bin {bin_names[bin_id-1]:>10s} BPM: {len(bin_indices):4d} available, {n:3d} selected")

        # 限制總數
        if len(selected_indices) > num_samples:
            selected_indices = np.random.choice(
                selected_indices, num_samples, replace=False
            ).tolist()

        # 選取樣本
        calibration_samples_6d = samples_6d[selected_indices]
        calibration_labels = labels[selected_indices]

        # 轉換為 4D
        print(f"   Converting 6D to 4D...")
        N, T, ROI, H, W, C = calibration_samples_6d.shape
        calibration_samples_4d = calibration_samples_6d.permute(0, 1, 2, 5, 3, 4)  # (N, T, ROI, C, H, W)
        calibration_samples_4d = calibration_samples_4d.reshape(N, T*ROI*C, H, W)  # (N, 72, 36, 36)

        self.samples = calibration_samples_4d.numpy().astype(np.float32)
        self.labels = calibration_labels.numpy()
        self.num_samples = len(selected_indices)
        self.current_idx = 0

        print(f"   [OK] Prepared {self.num_samples} calibration samples (4D)")
        print(f"   4D shape: {self.samples.shape}")
        print(f"   HR range: {self.labels.min():.2f} - {self.labels.max():.2f} BPM")
        print(f"   HR mean: {self.labels.mean():.2f} BPM")
        print(f"   HR std: {self.labels.std():.2f} BPM")

    def get_next(self):
        """
        獲取下一個校準樣本

        Returns:
            dict: {input_name: numpy_array} 或 None（無更多數據）
        """
        if self.current_idx >= self.num_samples:
            return None

        # 返回字典 {input_name: numpy_array}
        sample = self.samples[self.current_idx:self.current_idx+1]
        self.current_idx += 1

        return {'input': sample}

    def rewind(self):
        """重置讀取索引"""
        self.current_idx = 0


def quantize_model_qdq(fp32_path, int8_path, data_path, num_calibration_samples=200):
    """
    使用 QDQ 格式進行 INT8 量化（推薦方法）

    QDQ 優點：
    - 更高精度（保留更多動態範圍）
    - X-CUBE-AI 完全支持
    - Per-channel 量化
    """
    print("\n[Quantization Method] QDQ (Quantize-DeQuantize) - Recommended")
    print("   Format: QDQ")
    print("   Weight type: INT8")
    print("   Activation type: INT8")
    print("   Per-channel: True")

    # 創建校準數據讀取器
    calibration_data_reader = RPPG4DCalibrationDataReader(data_path, num_calibration_samples)

    # QDQ 格式量化
    print(f"\n   Running quantization...")
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=calibration_data_reader,
        quant_format=QuantFormat.QDQ,        # QDQ 格式
        per_channel=True,                    # Per-channel 量化
        weight_type=QuantType.QInt8,         # INT8 權重
        activation_type=QuantType.QInt8,     # INT8 激活
        use_external_data_format=False
    )

    print(f"   [OK] Quantization completed")


def verify_quantized_model(fp32_path, int8_path, data_path):
    """驗證量化模型精度（簡化版，只測試幾個樣本）"""
    print("\n[Verification] Quick accuracy test...")

    # 載入模型
    sess_fp32 = ort.InferenceSession(str(fp32_path))
    sess_int8 = ort.InferenceSession(str(int8_path))

    # 載入測試數據（取後 20% 作為測試集）
    data = torch.load(data_path)
    total = data['num_samples']
    split_idx = int(total * 0.8)
    test_samples_6d = data['samples'][split_idx:split_idx+100]  # 只測試 100 樣本
    test_labels = data['labels'][split_idx:split_idx+100]

    # 轉換為 4D
    N, T, ROI, H, W, C = test_samples_6d.shape
    test_samples_4d = test_samples_6d.permute(0, 1, 2, 5, 3, 4)
    test_samples_4d = test_samples_4d.reshape(N, T*ROI*C, H, W)
    test_samples_4d = test_samples_4d.numpy().astype(np.float32)

    # 批次推論 (STM32N6 模型要求 batch=1)
    batch_size = 1  # Fixed batch size for STM32N6 compatibility
    preds_fp32 = []
    preds_int8 = []

    for i in range(0, len(test_samples_4d), batch_size):
        batch = test_samples_4d[i:i+batch_size]

        y_fp32 = sess_fp32.run(None, {'input': batch})[0]
        y_int8 = sess_int8.run(None, {'input': batch})[0]

        preds_fp32.append(y_fp32)
        preds_int8.append(y_int8)

    preds_fp32 = np.concatenate(preds_fp32).flatten()
    preds_int8 = np.concatenate(preds_int8).flatten()
    labels_np = test_labels.numpy()

    # 計算誤差
    mae_fp32 = np.mean(np.abs(preds_fp32 - labels_np))
    mae_int8 = np.mean(np.abs(preds_int8 - labels_np))
    mae_degradation = mae_int8 - mae_fp32

    rmse_fp32 = np.sqrt(np.mean((preds_fp32 - labels_np)**2))
    rmse_int8 = np.sqrt(np.mean((preds_int8 - labels_np)**2))

    print(f"   Test samples: {len(labels_np)}")
    print(f"   FP32 MAE:  {mae_fp32:.2f} BPM")
    print(f"   INT8 MAE:  {mae_int8:.2f} BPM")
    print(f"   Degradation: {mae_degradation:.2f} BPM")
    print(f"   FP32 RMSE: {rmse_fp32:.2f} BPM")
    print(f"   INT8 RMSE: {rmse_int8:.2f} BPM")

    # 評估
    if mae_degradation < 1.0:
        quality = "EXCELLENT"
    elif mae_degradation < 2.0:
        quality = "GOOD"
    elif mae_degradation < 3.0:
        quality = "ACCEPTABLE"
    else:
        quality = "POOR"

    print(f"   Quality: {quality}")

    return mae_fp32, mae_int8, mae_degradation


def main():
    print("="*70)
    print("Quantize 4D Model to INT8 (Improved Version)")
    print("="*70)

    # 路徑
    models_dir = Path("models")
    data_path = Path("data/ubfc_processed.pt")
    fp32_path = models_dir / "rppg_4d_fp32.onnx"
    int8_path = models_dir / "rppg_4d_int8_qdq.onnx"

    # 檢查輸入
    if not fp32_path.exists():
        print(f"\n[ERROR] FP32 model not found: {fp32_path}")
        print("   Please run convert_to_4d_for_stm32.py first")
        return

    if not data_path.exists():
        print(f"\n[ERROR] Data not found: {data_path}")
        return

    print(f"\n[Input]")
    print(f"   FP32 model: {fp32_path}")
    print(f"   Data: {data_path}")
    print(f"   FP32 size: {fp32_path.stat().st_size / 1024:.2f} KB")

    # 執行量化
    quantize_model_qdq(fp32_path, int8_path, data_path, num_calibration_samples=200)

    # 驗證
    mae_fp32, mae_int8, degradation = verify_quantized_model(fp32_path, int8_path, data_path)

    # 完成
    print("\n" + "="*70)
    print("[SUCCESS] Quantization Complete!")
    print("="*70)

    print(f"\nModel sizes:")
    print(f"   FP32: {fp32_path.stat().st_size / 1024:.2f} KB")
    print(f"   INT8: {int8_path.stat().st_size / 1024:.2f} KB")
    compression_ratio = fp32_path.stat().st_size / int8_path.stat().st_size
    print(f"   Compression: {compression_ratio:.2f}x")

    print(f"\nAccuracy (Quick test on 100 samples):")
    print(f"   FP32 MAE: {mae_fp32:.2f} BPM")
    print(f"   INT8 MAE: {mae_int8:.2f} BPM")
    print(f"   Degradation: {degradation:.2f} BPM")

    print(f"\nOutput: {int8_path}")
    print(f"\nNext steps:")
    print(f"   1. Run full evaluation:")
    print(f"      python evaluate_quantized_model.py")
    print(f"   2. Download to local:")
    print(f"      scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/models/rppg_4d_int8_qdq.onnx D:\\MIAT\\rppg\\")
    print(f"   3. Import to STM32CubeMX (use O1 or O2, avoid O3)")
    print("="*70)


if __name__ == "__main__":
    main()
