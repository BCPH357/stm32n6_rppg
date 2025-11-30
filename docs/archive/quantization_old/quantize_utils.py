"""
校準數據集準備工具
用於 ONNX Runtime INT8 量化的校準數據準備

使用方式:
    conda activate zerodce_tf
    python quantize_utils.py
"""

import torch
import numpy as np
import os
import sys

# 添加父目錄到路徑以導入 model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def prepare_calibration_dataset(
    data_path='../data/ubfc_processed.pt',
    output_path='calibration_data.pt',
    num_samples=200
):
    """
    準備校準數據集

    最佳實踐：
    - 使用訓練集的子集（非測試集！）
    - 100-200 samples 足夠（約 2-4 個 batch）
    - 必須涵蓋各種場景（不同受試者、光照、膚色）

    Args:
        data_path: 預處理數據路徑
        output_path: 校準數據保存路徑
        num_samples: 校準樣本數量
    """
    print("="*70)
    print("Preparing Calibration Dataset for INT8 Quantization")
    print("="*70)

    # 載入完整數據集
    print(f"\n[1/4] Loading data from: {data_path}")
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        print(f"   Please ensure training data is available.")
        return False

    data = torch.load(data_path)
    samples = data['samples']  # (N, 8, 3, 36, 36, 3)
    labels = data['labels']    # (N,)

    print(f"   Total samples: {len(samples)}")
    print(f"   Sample shape: {samples.shape}")
    print(f"   Label range: {labels.min():.2f} - {labels.max():.2f} BPM")

    # 分層採樣（確保各心率範圍都有代表）
    print(f"\n[2/4] Stratified sampling across HR ranges...")
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
            print(f"   Bin {bin_names[bin_id-1]:>10s} BPM: {len(bin_indices):4d} available, {n:3d} selected")

    # 限制總數
    if len(selected_indices) > num_samples:
        print(f"\n   Limiting to {num_samples} samples (from {len(selected_indices)})")
        selected_indices = np.random.choice(
            selected_indices, num_samples, replace=False
        ).tolist()

    # 創建校準數據集
    print(f"\n[3/4] Creating calibration dataset...")
    calibration_samples = samples[selected_indices]
    calibration_labels = labels[selected_indices]

    calibration_data = {
        'samples': calibration_samples,
        'labels': calibration_labels,
        'num_samples': len(selected_indices),
        'hr_range': (calibration_labels.min().item(), calibration_labels.max().item()),
        'hr_mean': calibration_labels.mean().item(),
        'hr_std': calibration_labels.std().item()
    }

    # 保存
    print(f"\n[4/4] Saving calibration data to: {output_path}")
    torch.save(calibration_data, output_path)

    # 統計報告
    print("\n" + "="*70)
    print("Calibration Dataset Summary")
    print("="*70)
    print(f"Total samples:  {len(selected_indices)}")
    print(f"Sample shape:   {calibration_samples.shape}")
    print(f"HR range:       {calibration_labels.min():.2f} - {calibration_labels.max():.2f} BPM")
    print(f"HR mean:        {calibration_labels.mean():.2f} BPM")
    print(f"HR std:         {calibration_labels.std():.2f} BPM")
    print(f"File size:      {os.path.getsize(output_path) / (1024**2):.2f} MB")
    print("="*70)
    print("✅ Calibration dataset prepared successfully!")
    print("="*70)

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Prepare calibration dataset for INT8 quantization')
    parser.add_argument('--data', type=str, default='../data/ubfc_processed.pt',
                        help='Path to preprocessed training data')
    parser.add_argument('--output', type=str, default='calibration_data.pt',
                        help='Output path for calibration data')
    parser.add_argument('--num_samples', type=int, default=200,
                        help='Number of calibration samples')

    args = parser.parse_args()

    success = prepare_calibration_dataset(
        data_path=args.data,
        output_path=args.output,
        num_samples=args.num_samples
    )

    sys.exit(0 if success else 1)
