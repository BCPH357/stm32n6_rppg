"""
量化精度驗證工具
對比 FP32 vs INT8 ONNX 模型的預測精度

使用方式:
    conda activate zerodce_tf
    python verify_quantization.py
"""

import torch
import onnxruntime as ort
import numpy as np
import os
import sys


def verify_quantization_accuracy(
    fp32_onnx_path='models/rppg_fp32.onnx',
    int8_onnx_path='models/rppg_int8_qdq.onnx',
    test_data_path='../data/ubfc_processed.pt',
    num_samples=500
):
    """
    驗證量化後的準確度損失

    指標：
    - MAE 差異
    - RMSE 差異
    - 輸出分布差異

    Args:
        fp32_onnx_path: FP32 ONNX 模型路徑
        int8_onnx_path: INT8 ONNX 模型路徑
        test_data_path: 測試數據路徑
        num_samples: 驗證樣本數量

    Returns:
        dict: 驗證結果統計
    """
    print("="*70)
    print("Verifying INT8 Quantization Accuracy")
    print("="*70)

    # [1/5] 檢查文件
    print(f"\n[1/5] Checking files...")
    if not os.path.exists(fp32_onnx_path):
        print(f"❌ Error: FP32 model not found: {fp32_onnx_path}")
        return None

    if not os.path.exists(int8_onnx_path):
        print(f"❌ Error: INT8 model not found: {int8_onnx_path}")
        return None

    if not os.path.exists(test_data_path):
        print(f"❌ Error: Test data not found: {test_data_path}")
        return None

    print(f"   ✅ All files found")

    # [2/5] 載入測試數據
    print(f"\n[2/5] Loading test data...")
    data = torch.load(test_data_path)
    samples = data['samples'].numpy()  # (N, 8, 3, 36, 36, 3)
    labels = data['labels'].numpy()    # (N,)

    # 限制樣本數量
    if num_samples > len(samples):
        num_samples = len(samples)

    samples = samples[:num_samples]
    labels = labels[:num_samples]

    print(f"   Using {num_samples} test samples")
    print(f"   HR range: {labels.min():.2f} - {labels.max():.2f} BPM")

    # [3/5] FP32 模型推論
    print(f"\n[3/5] Running FP32 model inference...")
    try:
        sess_fp32 = ort.InferenceSession(fp32_onnx_path)
        input_name = sess_fp32.get_inputs()[0].name

        preds_fp32 = []
        for i in range(len(samples)):
            if (i + 1) % 100 == 0:
                print(f"   Progress: {i+1}/{len(samples)}")

            input_data = {input_name: samples[i:i+1].astype(np.float32)}
            output = sess_fp32.run(None, input_data)[0]
            preds_fp32.append(output[0, 0])

        preds_fp32 = np.array(preds_fp32)
        print(f"   ✅ FP32 inference completed")
    except Exception as e:
        print(f"❌ Error during FP32 inference: {e}")
        return None

    # [4/5] INT8 模型推論
    print(f"\n[4/5] Running INT8 model inference...")
    try:
        sess_int8 = ort.InferenceSession(int8_onnx_path)
        input_name = sess_int8.get_inputs()[0].name

        preds_int8 = []
        for i in range(len(samples)):
            if (i + 1) % 100 == 0:
                print(f"   Progress: {i+1}/{len(samples)}")

            input_data = {input_name: samples[i:i+1].astype(np.float32)}
            output = sess_int8.run(None, input_data)[0]
            preds_int8.append(output[0, 0])

        preds_int8 = np.array(preds_int8)
        print(f"   ✅ INT8 inference completed")
    except Exception as e:
        print(f"❌ Error during INT8 inference: {e}")
        return None

    # [5/5] 計算精度指標
    print(f"\n[5/5] Computing accuracy metrics...")

    # Ground truth 指標
    mae_fp32 = np.mean(np.abs(preds_fp32 - labels))
    mae_int8 = np.mean(np.abs(preds_int8 - labels))

    rmse_fp32 = np.sqrt(np.mean((preds_fp32 - labels)**2))
    rmse_int8 = np.sqrt(np.mean((preds_int8 - labels)**2))

    # FP32 vs INT8 輸出差異
    output_diff = np.abs(preds_fp32 - preds_int8)
    output_diff_mean = output_diff.mean()
    output_diff_max = output_diff.max()
    output_diff_std = output_diff.std()

    # 相對誤差
    mae_increase = mae_int8 - mae_fp32
    mae_increase_pct = (mae_int8 / mae_fp32 - 1) * 100
    rmse_increase = rmse_int8 - rmse_fp32
    rmse_increase_pct = (rmse_int8 / rmse_fp32 - 1) * 100

    # 結果報告
    print("\n" + "="*70)
    print("Quantization Accuracy Verification Results")
    print("="*70)

    print(f"\n📊 FP32 Model Performance:")
    print(f"   MAE:  {mae_fp32:.4f} BPM")
    print(f"   RMSE: {rmse_fp32:.4f} BPM")

    print(f"\n📊 INT8 Model Performance:")
    print(f"   MAE:  {mae_int8:.4f} BPM")
    print(f"   RMSE: {rmse_int8:.4f} BPM")

    print(f"\n📈 Quantization Impact:")
    print(f"   MAE increase:  {mae_increase:+.4f} BPM ({mae_increase_pct:+.2f}%)")
    print(f"   RMSE increase: {rmse_increase:+.4f} BPM ({rmse_increase_pct:+.2f}%)")

    print(f"\n🔍 Output Difference (FP32 vs INT8):")
    print(f"   Mean:  {output_diff_mean:.4f} BPM")
    print(f"   Max:   {output_diff_max:.4f} BPM")
    print(f"   Std:   {output_diff_std:.4f} BPM")

    # 可接受性判斷
    print("\n" + "="*70)
    acceptable_threshold = 2.0  # MAE 增加 < 2 BPM
    if mae_increase < acceptable_threshold:
        print(f"✅ Quantization ACCEPTABLE")
        print(f"   MAE increase ({mae_increase:.2f} BPM) < threshold ({acceptable_threshold} BPM)")
        print(f"   INT8 model is ready for deployment!")
    else:
        print(f"⚠️  Quantization DEGRADATION SIGNIFICANT")
        print(f"   MAE increase ({mae_increase:.2f} BPM) >= threshold ({acceptable_threshold} BPM)")
        print(f"   Consider:")
        print(f"   - Increasing calibration samples (currently {num_samples})")
        print(f"   - Using Quantization-Aware Training (QAT)")
        print(f"   - Adjusting quantization parameters")

    print("="*70)

    # 返回統計結果
    results = {
        'mae_fp32': mae_fp32,
        'mae_int8': mae_int8,
        'mae_increase': mae_increase,
        'mae_increase_pct': mae_increase_pct,
        'rmse_fp32': rmse_fp32,
        'rmse_int8': rmse_int8,
        'rmse_increase': rmse_increase,
        'rmse_increase_pct': rmse_increase_pct,
        'output_diff_mean': output_diff_mean,
        'output_diff_max': output_diff_max,
        'output_diff_std': output_diff_std,
        'acceptable': mae_increase < acceptable_threshold,
        'num_samples': num_samples
    }

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Verify INT8 quantization accuracy')
    parser.add_argument('--fp32', type=str, default='models/rppg_fp32.onnx',
                        help='FP32 ONNX model path')
    parser.add_argument('--int8', type=str, default='models/rppg_int8_qdq.onnx',
                        help='INT8 ONNX model path')
    parser.add_argument('--data', type=str, default='../data/ubfc_processed.pt',
                        help='Test data path')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of test samples')

    args = parser.parse_args()

    results = verify_quantization_accuracy(
        fp32_onnx_path=args.fp32,
        int8_onnx_path=args.int8,
        test_data_path=args.data,
        num_samples=args.num_samples
    )

    # 根據結果決定退出碼
    if results is None:
        sys.exit(1)  # 錯誤
    elif results['acceptable']:
        sys.exit(0)  # 成功
    else:
        sys.exit(2)  # 精度不足
