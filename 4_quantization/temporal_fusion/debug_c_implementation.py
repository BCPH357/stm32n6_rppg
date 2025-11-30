"""
調試 C 實現 - 逐層對比 PyTorch 和 C 的中間結果

執行方法:
    python debug_c_implementation.py
"""

import torch
import numpy as np
from model_split import TemporalFusion


def main():
    print("="*70)
    print("調試 C 實現 - 逐層對比")
    print("="*70)

    # 載入 PyTorch 模型
    checkpoint = torch.load("checkpoints/temporal_fusion.pth", map_location='cpu')
    model = TemporalFusion(window_size=8, num_rois=3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 生成一個簡單的測試輸入
    np.random.seed(42)
    test_input = np.random.randn(1, 24, 16).astype(np.float32)
    test_input_tensor = torch.from_numpy(test_input)

    print(f"\n[輸入數據]")
    print(f"  Shape: {test_input.shape}")
    print(f"  前 3 個特徵向量:")
    for i in range(3):
        print(f"    [{i}]: {test_input[0, i, :5]}... (顯示前 5 個值)")

    # -------------------------------------------------------------------------
    # Step 1: Reshape
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Step 1: Reshape (24, 16) → (8, 3, 16) → (8, 48) → (48, 8)")
    print(f"{'='*70}")

    x = test_input_tensor
    B, TxROI, F = x.shape
    print(f"  原始輸入: {x.shape}")

    # PyTorch reshape
    x = x.view(B, 8, 3, F)
    print(f"  view(B, 8, 3, 16): {x.shape}")
    print(f"    x[0, 0, 0, :5] = {x[0, 0, 0, :5]}")
    print(f"    x[0, 0, 1, :5] = {x[0, 0, 1, :5]}")
    print(f"    x[0, 0, 2, :5] = {x[0, 0, 2, :5]}")

    x = x.view(B, 8, 48)
    print(f"  view(B, 8, 48): {x.shape}")
    print(f"    x[0, 0, :10] = {x[0, 0, :10]}")

    x = x.transpose(1, 2)
    print(f"  transpose(1, 2): {x.shape}")
    print(f"    x[0, :5, 0] = {x[0, :5, 0]}")
    print(f"    x[0, :5, 1] = {x[0, :5, 1]}")

    # 保存 reshape 後的結果
    reshaped_pt = x.clone()

    # 生成 C 代碼用的 reshape 邏輯
    print(f"\n  [生成 C 代碼參考]")
    print(f"  // 輸入: features[24][16]")
    print(f"  // 輸出: reshaped[48][8]")
    print(f"  //")
    print(f"  // 映射關係:")
    for t in range(2):  # 只顯示前 2 個時間步
        for roi in range(3):
            for f in range(3):  # 只顯示前 3 個特徵
                src_idx = t * 3 + roi
                dst_ch = roi * 16 + f
                src_val = test_input[0, src_idx, f]
                dst_val = reshaped_pt[0, dst_ch, t].item()
                print(f"  features[{src_idx:2d}][{f:2d}] = {src_val:8.4f} → reshaped[{dst_ch:2d}][{t}] = {dst_val:8.4f}")

    # -------------------------------------------------------------------------
    # Step 2: Conv1D Layer 1
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Step 2: Conv1D(48 → 32, kernel=3) + ReLU")
    print(f"{'='*70}")

    conv1_weight = model.temporal[0].weight.data  # (32, 48, 3)
    conv1_bias = model.temporal[0].bias.data      # (32,)

    print(f"  權重 shape: {conv1_weight.shape}")
    print(f"  偏置 shape: {conv1_bias.shape}")
    print(f"  權重範圍: [{conv1_weight.min():.6f}, {conv1_weight.max():.6f}]")
    print(f"  偏置範圍: [{conv1_bias.min():.6f}, {conv1_bias.max():.6f}]")

    # PyTorch Conv1D
    conv1_out = model.temporal[0](reshaped_pt)
    conv1_out = torch.relu(conv1_out)

    print(f"  輸出 shape: {conv1_out.shape}")
    print(f"  輸出範圍: [{conv1_out.min():.6f}, {conv1_out.max():.6f}]")
    print(f"  第 1 個輸出通道 (前 8 個值): {conv1_out[0, 0, :]}")
    print(f"  第 2 個輸出通道 (前 8 個值): {conv1_out[0, 1, :]}")

    # 手動計算第一個輸出 (驗證 Conv1D 邏輯)
    print(f"\n  [手動計算驗證 - 第 1 個輸出通道, 時間步 0]")
    oc = 0
    t = 0
    padding = 1

    manual_sum = conv1_bias[oc].item()
    for ic in range(48):
        for k in range(3):
            t_in = t - padding + k  # -1, 0, 1
            if 0 <= t_in < 8:
                val = reshaped_pt[0, ic, t_in].item() * conv1_weight[oc, ic, k].item()
                manual_sum += val
                if ic < 2 and k < 3:  # 只顯示前幾個
                    print(f"    reshaped[{ic}][{t_in}] * weight[{oc}][{ic}][{k}] = {reshaped_pt[0, ic, t_in].item():.6f} * {conv1_weight[oc, ic, k].item():.6f} = {val:.6f}")

    manual_result = max(0, manual_sum)
    pytorch_result = conv1_out[0, oc, t].item()

    print(f"  手動計算結果: {manual_result:.6f}")
    print(f"  PyTorch 結果:  {pytorch_result:.6f}")
    print(f"  差異: {abs(manual_result - pytorch_result):.8f}")

    # -------------------------------------------------------------------------
    # 完整推論
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"完整推論對比")
    print(f"{'='*70}")

    with torch.no_grad():
        hr_pytorch = model(test_input_tensor)

    print(f"  PyTorch HR: {hr_pytorch.item():.6f} BPM")

    # 保存測試數據為 C 格式
    print(f"\n[保存測試數據供 C 代碼使用]")
    output_file = "test_data_for_c.txt"

    with open(output_file, 'w') as f:
        f.write(f"// 測試輸入數據\n")
        f.write(f"const float test_input[24][16] = {{\n")
        for i in range(24):
            f.write(f"    {{")
            for j in range(16):
                f.write(f"{test_input[0, i, j]:.8f}f")
                if j < 15:
                    f.write(", ")
            f.write(f"}}")
            if i < 23:
                f.write(",")
            f.write(f"\n")
        f.write(f"}};\n\n")
        f.write(f"// 預期輸出\n")
        f.write(f"// PyTorch HR: {hr_pytorch.item():.8f} BPM\n")

    print(f"  已保存到: {output_file}")
    print(f"  預期輸出: {hr_pytorch.item():.6f} BPM")


if __name__ == "__main__":
    main()
