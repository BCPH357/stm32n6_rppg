"""
TFLite 量化模型驗證腳本

驗證 TFLite INT8 量化模型的精度損失：
- 對比 PyTorch 原始模型與 TFLite 量化模型
- 計算 MAE, RMSE, MAPE, R²
- 評估量化質量等級

執行方法:
    python validate_tflite.py

輸出:
    驗證報告（MAE, RMSE, 質量評級）
"""

import torch
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 導入模型定義
from model_split import SpatialCNN, TemporalFusion, CombinedModel


def load_test_data(data_path, num_samples=200):
    """
    載入測試數據（與訓練/校準數據不同）

    Args:
        data_path: ubfc_processed.pt 路徑
        num_samples: 測試樣本數

    Returns:
        test_samples: (num_samples, 8, 3, 36, 36, 3)
        test_labels: (num_samples,)
    """
    print(f"\n[載入測試數據]")
    data = torch.load(data_path)
    samples = data['samples']
    labels = data['labels']

    # 使用後面的樣本作為測試集（避免與校準數據重疊）
    total = len(samples)
    test_indices = torch.arange(total - num_samples, total)

    test_samples = samples[test_indices]
    test_labels = labels[test_indices]

    print(f"  測試樣本數: {len(test_samples)}")
    print(f"  HR 範圍: [{test_labels.min():.1f}, {test_labels.max():.1f}] BPM")

    return test_samples, test_labels


def run_pytorch_inference(test_samples, test_labels):
    """
    運行 PyTorch 原始模型推論（作為參考）

    Args:
        test_samples: (N, 8, 3, 36, 36, 3)
        test_labels: (N,)

    Returns:
        predictions: (N,) PyTorch 預測結果
    """
    print(f"\n{'='*70}")
    print(f"PyTorch 原始模型推論（參考基準）")
    print(f"{'='*70}")

    # 載入組合模型
    spatial_cnn = SpatialCNN()
    temporal_fusion = TemporalFusion(window_size=8, num_rois=3)

    spatial_checkpoint = torch.load("checkpoints/spatial_cnn.pth", map_location='cpu')
    temporal_checkpoint = torch.load("checkpoints/temporal_fusion.pth", map_location='cpu')

    spatial_cnn.load_state_dict(spatial_checkpoint['model_state_dict'])
    temporal_fusion.load_state_dict(temporal_checkpoint['model_state_dict'])

    combined_model = CombinedModel(spatial_cnn, temporal_fusion)
    combined_model.eval()

    # 推論
    predictions = []

    with torch.no_grad():
        for sample in test_samples:
            sample = sample.unsqueeze(0)  # (1, 8, 3, 36, 36, 3)
            hr = combined_model(sample)
            predictions.append(hr.item())

    predictions = np.array(predictions)

    # 計算誤差
    mae = mean_absolute_error(test_labels.numpy(), predictions)
    rmse = np.sqrt(mean_squared_error(test_labels.numpy(), predictions))
    mape = np.mean(np.abs((test_labels.numpy() - predictions) / test_labels.numpy())) * 100
    r2 = r2_score(test_labels.numpy(), predictions)

    print(f"\n[PyTorch 模型性能]")
    print(f"  MAE:  {mae:.4f} BPM")
    print(f"  RMSE: {rmse:.4f} BPM")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")

    return predictions


def run_tflite_inference(test_samples, test_labels):
    """
    運行 TFLite 量化模型推論

    Args:
        test_samples: (N, 8, 3, 36, 36, 3)
        test_labels: (N,)

    Returns:
        predictions: (N,) TFLite 預測結果
    """
    print(f"\n{'='*70}")
    print(f"TFLite INT8 量化模型推論")
    print(f"{'='*70}")

    # 載入 TFLite 模型
    spatial_path = "models/spatial_cnn_int8.tflite"
    temporal_path = "models/temporal_fusion_int8.tflite"

    if not Path(spatial_path).exists() or not Path(temporal_path).exists():
        print(f"[ERROR] TFLite 模型不存在，請先運行 export_tflite_split.py")
        return None

    # 初始化解釋器
    spatial_interpreter = tf.lite.Interpreter(model_path=spatial_path)
    temporal_interpreter = tf.lite.Interpreter(model_path=temporal_path)

    spatial_interpreter.allocate_tensors()
    temporal_interpreter.allocate_tensors()

    # 獲取輸入輸出詳情
    spatial_input_details = spatial_interpreter.get_input_details()
    spatial_output_details = spatial_interpreter.get_output_details()

    temporal_input_details = temporal_interpreter.get_input_details()
    temporal_output_details = temporal_interpreter.get_output_details()

    print(f"\n[Spatial CNN TFLite]")
    print(f"  輸入: {spatial_input_details[0]['shape']} {spatial_input_details[0]['dtype']}")
    print(f"  輸出: {spatial_output_details[0]['shape']} {spatial_output_details[0]['dtype']}")

    print(f"\n[Temporal Fusion TFLite]")
    print(f"  輸入: {temporal_input_details[0]['shape']} {temporal_input_details[0]['dtype']}")
    print(f"  輸出: {temporal_output_details[0]['shape']} {temporal_output_details[0]['dtype']}")

    # 推論
    predictions = []

    for idx, sample in enumerate(test_samples):
        # sample: (8, 3, 36, 36, 3)

        # Step 1: 運行 Spatial CNN (24 次)
        features_list = []

        for t in range(8):
            for roi in range(3):
                patch = sample[t, roi, :, :, :]  # (36, 36, 3) HWC

                # 轉換為 numpy 並保持 NHWC 格式（TFLite 格式）
                if isinstance(patch, torch.Tensor):
                    patch = patch.numpy()

                # 添加 batch 維度: (36, 36, 3) → (1, 36, 36, 3) NHWC
                patch = np.expand_dims(patch, axis=0).astype(np.float32)

                # 量化輸入
                spatial_input_scale, spatial_input_zero_point = spatial_input_details[0]['quantization']
                patch_quant = (patch / spatial_input_scale + spatial_input_zero_point).astype(np.int8)

                # 推論
                spatial_interpreter.set_tensor(spatial_input_details[0]['index'], patch_quant)
                spatial_interpreter.invoke()

                # 獲取輸出
                feat_quant = spatial_interpreter.get_tensor(spatial_output_details[0]['index'])

                # 反量化
                spatial_output_scale, spatial_output_zero_point = spatial_output_details[0]['quantization']
                feat = (feat_quant.astype(np.float32) - spatial_output_zero_point) * spatial_output_scale

                features_list.append(feat)

        # 堆疊特徵 (1, 24, 16)
        features = np.concatenate(features_list, axis=0)  # (24, 16)
        features = np.expand_dims(features, axis=0)  # (1, 24, 16)

        # Step 2: 運行 Temporal Fusion
        temporal_interpreter.set_tensor(temporal_input_details[0]['index'], features.astype(np.float32))
        temporal_interpreter.invoke()

        hr = temporal_interpreter.get_tensor(temporal_output_details[0]['index'])
        predictions.append(hr[0, 0])

        if (idx + 1) % 50 == 0:
            print(f"  已處理: {idx + 1}/{len(test_samples)} 樣本")

    predictions = np.array(predictions)

    # 計算誤差
    mae = mean_absolute_error(test_labels.numpy(), predictions)
    rmse = np.sqrt(mean_squared_error(test_labels.numpy(), predictions))
    mape = np.mean(np.abs((test_labels.numpy() - predictions) / test_labels.numpy())) * 100
    r2 = r2_score(test_labels.numpy(), predictions)

    print(f"\n[TFLite 模型性能]")
    print(f"  MAE:  {mae:.4f} BPM")
    print(f"  RMSE: {rmse:.4f} BPM")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")

    return predictions


def compare_predictions(pytorch_preds, tflite_preds, test_labels):
    """
    對比 PyTorch 和 TFLite 預測結果

    Args:
        pytorch_preds: PyTorch 預測
        tflite_preds: TFLite 預測
        test_labels: 真實標籤
    """
    print(f"\n{'='*70}")
    print(f"量化精度分析")
    print(f"{'='*70}")

    # PyTorch vs Ground Truth
    pytorch_mae = mean_absolute_error(test_labels.numpy(), pytorch_preds)

    # TFLite vs Ground Truth
    tflite_mae = mean_absolute_error(test_labels.numpy(), tflite_preds)

    # TFLite vs PyTorch
    quantization_error = mean_absolute_error(pytorch_preds, tflite_preds)

    mae_increase = tflite_mae - pytorch_mae

    print(f"\n[MAE 對比]")
    print(f"  PyTorch (FP32):        {pytorch_mae:.4f} BPM")
    print(f"  TFLite (INT8):         {tflite_mae:.4f} BPM")
    print(f"  MAE 增加:              {mae_increase:.4f} BPM")
    print(f"  量化誤差 (vs PyTorch): {quantization_error:.4f} BPM")

    # 質量評級
    print(f"\n[量化質量評級]")
    if mae_increase < 0.5:
        quality = "EXCELLENT"
        print(f"  質量: {quality} (MAE 增加 < 0.5 BPM)")
    elif mae_increase < 1.5:
        quality = "GOOD"
        print(f"  質量: {quality} (MAE 增加 < 1.5 BPM)")
    elif mae_increase < 3.0:
        quality = "ACCEPTABLE"
        print(f"  質量: {quality} (MAE 增加 < 3.0 BPM)")
    else:
        quality = "POOR"
        print(f"  質量: {quality} (MAE 增加 >= 3.0 BPM)")

    # 顯示樣本對比
    print(f"\n[樣本預測對比（前 10 個）]")
    print(f"  {'Index':>6} | {'Ground Truth':>12} | {'PyTorch':>10} | {'TFLite':>10} | {'Diff':>8}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

    for i in range(min(10, len(test_labels))):
        gt = test_labels[i].item()
        pt = pytorch_preds[i]
        tfl = tflite_preds[i]
        diff = abs(pt - tfl)
        print(f"  {i:6d} | {gt:12.2f} | {pt:10.2f} | {tfl:10.2f} | {diff:8.4f}")

    # 分布分析
    print(f"\n[誤差分布分析]")
    diff_array = np.abs(pytorch_preds - tflite_preds)
    print(f"  最小差異: {diff_array.min():.4f} BPM")
    print(f"  最大差異: {diff_array.max():.4f} BPM")
    print(f"  平均差異: {diff_array.mean():.4f} BPM")
    print(f"  標準差:   {diff_array.std():.4f} BPM")

    # 百分位數
    p50 = np.percentile(diff_array, 50)
    p95 = np.percentile(diff_array, 95)
    p99 = np.percentile(diff_array, 99)

    print(f"\n[差異百分位數]")
    print(f"  P50: {p50:.4f} BPM")
    print(f"  P95: {p95:.4f} BPM")
    print(f"  P99: {p99:.4f} BPM")

    return quality, mae_increase


def main():
    print(f"{'='*70}")
    print(f"TFLite 量化模型驗證")
    print(f"{'='*70}")

    # 載入測試數據
    test_samples, test_labels = load_test_data(
        "data/ubfc_processed.pt",
        num_samples=200
    )

    # 運行 PyTorch 推論
    pytorch_preds = run_pytorch_inference(test_samples, test_labels)

    # 運行 TFLite 推論
    tflite_preds = run_tflite_inference(test_samples, test_labels)

    if tflite_preds is None:
        print(f"\n[ERROR] TFLite 推論失敗")
        return

    # 對比分析
    quality, mae_increase = compare_predictions(pytorch_preds, tflite_preds, test_labels)

    # 總結
    print(f"\n{'='*70}")
    print(f"[驗證完成]")
    print(f"{'='*70}")
    print(f"\n量化質量: {quality}")
    print(f"MAE 增加: {mae_increase:.4f} BPM")

    if quality in ["EXCELLENT", "GOOD"]:
        print(f"\n[OK] 量化精度良好，可以部署到 STM32N6")
    elif quality == "ACCEPTABLE":
        print(f"\n[WARNING] 量化精度可接受，建議進一步優化")
    else:
        print(f"\n[ERROR] 量化精度不佳，需要重新調整量化策略")

    print(f"\n下一步:")
    print(f"  1. 下載 .tflite 模型到本地")
    print(f"  2. 在 STM32CubeMX 中導入模型")
    print(f"  3. 驗證 STM32N6 推論結果")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
