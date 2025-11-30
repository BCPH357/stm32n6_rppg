"""
評估量化後的 4D ONNX 模型性能

用法 (在服務器上執行):
    cd /mnt/data_8T/ChenPinHao/server_training/
    conda activate rppg_training
    python evaluate_quantized_model.py

功能:
    1. 載入預處理好的測試數據
    2. 對比 FP32 vs INT8 模型輸出
    3. 計算性能指標: MAE, RMSE, MAPE, R²
    4. 統計預測分布: Min, Max, Mean, Std
"""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm


def load_test_data(data_path, max_samples=None):
    """載入測試數據"""
    print("\n" + "="*70)
    print("[Step 1] Loading Test Data")
    print("="*70)

    data = torch.load(data_path)

    # 提取測試集
    if 'test_samples' in data and 'test_labels' in data:
        inputs = data['test_samples']
        labels = data['test_labels']
    else:
        # 如果沒有預先分割，使用後 20% 作為測試集
        total = data['num_samples']
        split_idx = int(total * 0.8)
        inputs = data['samples'][split_idx:]
        labels = data['labels'][split_idx:]

    if max_samples is not None:
        inputs = inputs[:max_samples]
        labels = labels[:max_samples]

    print(f"   Test samples: {len(inputs)}")
    print(f"   Input shape: {inputs.shape}")
    print(f"   Label shape: {labels.shape}")
    print(f"   Label range: [{labels.min():.2f}, {labels.max():.2f}] BPM")
    print(f"   Label mean: {labels.mean():.2f} BPM")
    print(f"   Label std: {labels.std():.2f} BPM")

    return inputs, labels


def convert_6d_to_4d(inputs_6d):
    """將 6D 輸入轉換為 4D"""
    # 輸入: (N, 8, 3, 36, 36, 3)
    # 輸出: (N, 72, 36, 36)
    N, T, ROI, H, W, C = inputs_6d.shape

    # (N, T, ROI, H, W, C) -> (N, T, ROI, C, H, W) -> (N, T*ROI*C, H, W)
    inputs_4d = inputs_6d.permute(0, 1, 2, 5, 3, 4)  # (N, T, ROI, C, H, W)
    inputs_4d = inputs_4d.reshape(N, T*ROI*C, H, W)  # (N, 72, 36, 36)

    return inputs_4d


def evaluate_model(onnx_path, inputs_4d, labels, model_name="Model"):
    """評估 ONNX 模型"""
    print(f"\n[Evaluating {model_name}]")

    # 載入 ONNX 模型
    session = ort.InferenceSession(str(onnx_path))

    # 批次推論
    batch_size = 32
    predictions = []

    num_batches = (len(inputs_4d) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc=f"   Inference ({model_name})"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(inputs_4d))

        batch = inputs_4d[start_idx:end_idx].numpy()

        # ONNX 推論
        outputs = session.run(None, {'input': batch})[0]
        predictions.append(outputs)

    predictions = np.concatenate(predictions, axis=0).flatten()
    labels_np = labels.numpy()

    return predictions, labels_np


def calculate_metrics(predictions, labels):
    """計算性能指標"""
    print("\n" + "="*70)
    print("[Step 3] Calculating Metrics")
    print("="*70)

    # 基本統計
    pred_min = predictions.min()
    pred_max = predictions.max()
    pred_mean = predictions.mean()
    pred_std = predictions.std()

    label_min = labels.min()
    label_max = labels.max()
    label_mean = labels.mean()
    label_std = labels.std()

    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(predictions - labels))

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean((predictions - labels) ** 2))

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((predictions - labels) / labels)) * 100

    # R² (Coefficient of Determination)
    ss_res = np.sum((labels - predictions) ** 2)
    ss_tot = np.sum((labels - label_mean) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Pearson Correlation
    correlation = np.corrcoef(predictions, labels)[0, 1]

    return {
        'pred_min': pred_min,
        'pred_max': pred_max,
        'pred_mean': pred_mean,
        'pred_std': pred_std,
        'label_min': label_min,
        'label_max': label_max,
        'label_mean': label_mean,
        'label_std': label_std,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'correlation': correlation
    }


def print_metrics(metrics_fp32, metrics_int8):
    """打印指標對比"""
    print("\n" + "="*70)
    print("Performance Metrics")
    print("="*70)

    print("\n[1] Prediction Statistics")
    print("-" * 70)
    print(f"{'Metric':<20} {'FP32':>15} {'INT8':>15} {'Diff':>15}")
    print("-" * 70)
    print(f"{'Min BPM':<20} {metrics_fp32['pred_min']:>15.2f} {metrics_int8['pred_min']:>15.2f} {metrics_int8['pred_min']-metrics_fp32['pred_min']:>15.2f}")
    print(f"{'Max BPM':<20} {metrics_fp32['pred_max']:>15.2f} {metrics_int8['pred_max']:>15.2f} {metrics_int8['pred_max']-metrics_fp32['pred_max']:>15.2f}")
    print(f"{'Mean BPM':<20} {metrics_fp32['pred_mean']:>15.2f} {metrics_int8['pred_mean']:>15.2f} {metrics_int8['pred_mean']-metrics_fp32['pred_mean']:>15.2f}")
    print(f"{'Std BPM':<20} {metrics_fp32['pred_std']:>15.2f} {metrics_int8['pred_std']:>15.2f} {metrics_int8['pred_std']-metrics_fp32['pred_std']:>15.2f}")

    print("\n[2] Ground Truth Statistics")
    print("-" * 70)
    print(f"{'Min BPM':<20} {metrics_fp32['label_min']:>15.2f}")
    print(f"{'Max BPM':<20} {metrics_fp32['label_max']:>15.2f}")
    print(f"{'Mean BPM':<20} {metrics_fp32['label_mean']:>15.2f}")
    print(f"{'Std BPM':<20} {metrics_fp32['label_std']:>15.2f}")

    print("\n[3] Error Metrics")
    print("-" * 70)
    print(f"{'Metric':<20} {'FP32':>15} {'INT8':>15} {'Degradation':>15}")
    print("-" * 70)
    print(f"{'MAE (BPM)':<20} {metrics_fp32['mae']:>15.2f} {metrics_int8['mae']:>15.2f} {metrics_int8['mae']-metrics_fp32['mae']:>+15.2f}")
    print(f"{'RMSE (BPM)':<20} {metrics_fp32['rmse']:>15.2f} {metrics_int8['rmse']:>15.2f} {metrics_int8['rmse']-metrics_fp32['rmse']:>+15.2f}")
    print(f"{'MAPE (%)':<20} {metrics_fp32['mape']:>15.2f} {metrics_int8['mape']:>15.2f} {metrics_int8['mape']-metrics_fp32['mape']:>+15.2f}")

    print("\n[4] Correlation Metrics")
    print("-" * 70)
    print(f"{'Metric':<20} {'FP32':>15} {'INT8':>15} {'Diff':>15}")
    print("-" * 70)
    print(f"{'R² Score':<20} {metrics_fp32['r2']:>15.4f} {metrics_int8['r2']:>15.4f} {metrics_int8['r2']-metrics_fp32['r2']:>+15.4f}")
    print(f"{'Correlation':<20} {metrics_fp32['correlation']:>15.4f} {metrics_int8['correlation']:>15.4f} {metrics_int8['correlation']-metrics_fp32['correlation']:>+15.4f}")

    # 評估量化質量
    print("\n[5] Quantization Quality Assessment")
    print("-" * 70)

    mae_degradation = metrics_int8['mae'] - metrics_fp32['mae']

    if mae_degradation < 1.0:
        quality = "EXCELLENT"
        color = "\033[92m"  # Green
    elif mae_degradation < 2.0:
        quality = "GOOD"
        color = "\033[92m"  # Green
    elif mae_degradation < 3.0:
        quality = "ACCEPTABLE"
        color = "\033[93m"  # Yellow
    elif mae_degradation < 5.0:
        quality = "FAIR"
        color = "\033[93m"  # Yellow
    else:
        quality = "POOR"
        color = "\033[91m"  # Red

    reset = "\033[0m"

    print(f"MAE Degradation: {color}{mae_degradation:.2f} BPM{reset}")
    print(f"Quality Rating: {color}{quality}{reset}")
    print()

    if mae_degradation < 3.0:
        print("[OK] Quantization is suitable for deployment")
    else:
        print("[WARNING] Consider re-quantizing with more calibration data")




def main():
    print("="*70)
    print("Evaluate Quantized 4D rPPG Model")
    print("="*70)

    # 路徑配置
    data_path = Path("data/ubfc_processed.pt")
    fp32_path = Path("models/rppg_4d_fp32.onnx")
    int8_path = Path("models/rppg_4d_int8.onnx")

    # 檢查文件
    if not data_path.exists():
        print(f"\n[ERROR] Data not found: {data_path}")
        return

    if not fp32_path.exists():
        print(f"\n[ERROR] FP32 model not found: {fp32_path}")
        return

    if not int8_path.exists():
        print(f"\n[ERROR] INT8 model not found: {int8_path}")
        return

    # Step 1: 載入測試數據
    inputs_6d, labels = load_test_data(data_path, max_samples=None)

    # Step 2: 轉換為 4D
    print("\n" + "="*70)
    print("[Step 2] Converting 6D to 4D Input")
    print("="*70)
    inputs_4d = convert_6d_to_4d(inputs_6d)
    print(f"   6D shape: {inputs_6d.shape}")
    print(f"   4D shape: {inputs_4d.shape}")

    # Step 3: 評估 FP32 模型
    predictions_fp32, labels_np = evaluate_model(fp32_path, inputs_4d, labels, "FP32")

    # Step 4: 評估 INT8 模型
    predictions_int8, _ = evaluate_model(int8_path, inputs_4d, labels, "INT8")

    # Step 5: 計算指標
    metrics_fp32 = calculate_metrics(predictions_fp32, labels_np)
    metrics_int8 = calculate_metrics(predictions_int8, labels_np)

    # Step 6: 打印結果
    print_metrics(metrics_fp32, metrics_int8)

    # 完成
    print("\n" + "="*70)
    print("[SUCCESS] Evaluation Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
