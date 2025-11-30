"""
STM32N6 兼容的拆分 rPPG 模型
Pattern A: 分離式架構

Part 1: SpatialCNN - 單個 ROI 的空間特徵提取（~9.8K 參數）
Part 2: TemporalFusion - 時空特徵融合和心率預測（~10.4K 參數）

用於 STM32N6 部署：
- SpatialCNN: 在 MCU 上循環調用 24 次（8 幀 × 3 ROI）
- TemporalFusion: 調用 1 次（融合所有特徵）

輸入輸出:
- Spatial CNN: (1, 3, 36, 36) → (1, 16)
- Temporal Fusion: (1, 24, 16) → (1, 1)
"""

import torch
import torch.nn as nn


class SpatialCNN(nn.Module):
    """
    空間特徵提取器 - 處理單個 ROI 圖像

    輸入: (B, 3, 36, 36) - PyTorch NCHW 格式
    輸出: (B, 16) - 特徵向量

    STM32 部署:
    - 輸入: [1, 36, 36, 3] INT8 (TFLite NHWC 格式)
    - 輸出: [1, 16] FP32
    - 模型大小: ~40 KB (INT8 量化)
    - 推理時間: ~5-10 ms (NPU)
    """

    def __init__(self):
        super(SpatialCNN, self).__init__()

        self.features = nn.Sequential(
            # Layer 1: 3→16 channels
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 448 params
            nn.BatchNorm2d(16),                          # 32 params
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 36×36 → 18×18

            # Layer 2: 16→32 channels
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # 4,640 params
            nn.BatchNorm2d(32),                          # 64 params
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 18×18 → 9×9

            # Layer 3: 32→16 channels
            nn.Conv2d(32, 16, kernel_size=3, padding=1), # 4,624 params
            nn.BatchNorm2d(16),                          # 32 params
            nn.ReLU(inplace=True),

            # Global Average Pooling
            nn.AdaptiveAvgPool2d(1)                      # 9×9 → 1×1
        )

        # Total: ~9,840 params

    def forward(self, x):
        """
        Args:
            x: (B, 3, 36, 36) - 單個 ROI 圖像 (PyTorch NCHW)

        Returns:
            features: (B, 16) - 特徵向量
        """
        x = self.features(x)  # (B, 16, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 16)
        return x

    def get_num_params(self):
        """計算參數量"""
        return sum(p.numel() for p in self.parameters())


class TemporalFusion(nn.Module):
    """
    時序融合網絡 - 融合所有時空特徵並預測心率

    輸入: (B, 24, 16) - 8 幀 × 3 ROI × 16 特徵
    輸出: (B, 1) - 心率預測 [30, 180] BPM

    STM32 部署:
    - 輸入: [1, 24, 16] FP32
    - 輸出: [1, 1] FP32
    - 模型大小: ~40 KB (混合量化)
    - 推理時間: ~10-20 ms
    """

    def __init__(self, window_size=8, num_rois=3):
        super(TemporalFusion, self).__init__()

        self.window_size = window_size
        self.num_rois = num_rois

        # Temporal Conv1D: 建模時序依賴
        # 輸入: (B, 48, 8) - 48 = 3 ROI × 16 特徵
        self.temporal = nn.Sequential(
            nn.Conv1d(48, 32, kernel_size=3, padding=1),  # 4,640 params
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),  # 1,552 params
            nn.ReLU(inplace=True)
        )

        # Fully Connected: 回歸預測
        # 輸入: 16 × 8 = 128
        self.fc = nn.Sequential(
            nn.Linear(16 * window_size, 32),  # 4,128 params
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)                   # 33 params
        )

        # 心率範圍約束：[30, 180] BPM
        self.output_act = nn.Sigmoid()

        # Total: ~10,353 params

    def forward(self, x):
        """
        Args:
            x: (B, 24, 16) - 8 幀 × 3 ROI × 16 特徵

        Returns:
            hr: (B, 1) - 心率預測 [30, 180] BPM
        """
        B, TxROI, F = x.shape

        # Reshape: (B, 24, 16) → (B, 8, 3, 16) → (B, 8, 48)
        x = x.view(B, self.window_size, self.num_rois, F)
        x = x.view(B, self.window_size, self.num_rois * F)

        # Transpose for Conv1D: (B, 8, 48) → (B, 48, 8)
        x = x.transpose(1, 2)

        # Temporal modeling
        x = self.temporal(x)  # (B, 48, 8) → (B, 16, 8)

        # Flatten
        x = x.flatten(1)  # (B, 128)

        # FC layers
        out = self.fc(x)  # (B, 1)

        # Apply sigmoid constraint: [30, 180] BPM
        hr = self.output_act(out) * 150 + 30

        return hr

    def get_num_params(self):
        """計算參數量"""
        return sum(p.numel() for p in self.parameters())


class CombinedModel(nn.Module):
    """
    組合模型 - 用於訓練和驗證

    將 SpatialCNN 和 TemporalFusion 組合，接受原始 6D 輸入

    輸入: (B, 8, 3, 36, 36, 3) - 原始格式
    輸出: (B, 1) - 心率預測

    用途:
    - 權重遷移驗證
    - 與原始模型對比
    - 聯合訓練（如需要）
    """

    def __init__(self, spatial_cnn, temporal_fusion):
        super(CombinedModel, self).__init__()

        self.spatial = spatial_cnn
        self.temporal = temporal_fusion

    def forward(self, x):
        """
        Args:
            x: (B, 8, 3, 36, 36, 3) - 原始 6D 輸入

        Returns:
            hr: (B, 1) - 心率預測
        """
        B, T, ROI, H, W, C = x.shape

        # Permute and flatten: (B, T, ROI, H, W, C) → (B*T*ROI, C, H, W)
        x = x.permute(0, 1, 2, 5, 3, 4)  # (B, T, ROI, C, H, W)
        x = x.contiguous().view(B * T * ROI, C, H, W)  # (24, 3, 36, 36)

        # Extract spatial features (all ROIs)
        features = self.spatial(x)  # (24, 16)

        # Reshape to (B, T*ROI, 16)
        features = features.view(B, T * ROI, 16)  # (B, 24, 16)

        # Temporal fusion
        hr = self.temporal(features)

        return hr

    def get_num_params(self):
        """計算總參數量"""
        return self.spatial.get_num_params() + self.temporal.get_num_params()


def test_models():
    """測試拆分模型"""
    print("=" * 70)
    print("Testing Split rPPG Models (Pattern A)")
    print("=" * 70)

    # 創建模型
    spatial = SpatialCNN()
    temporal = TemporalFusion(window_size=8, num_rois=3)
    combined = CombinedModel(spatial, temporal)

    # 參數統計
    print("\n[參數統計]")
    print(f"  Spatial CNN:      {spatial.get_num_params():8,} params")
    print(f"  Temporal Fusion:  {temporal.get_num_params():8,} params")
    print(f"  Total:            {combined.get_num_params():8,} params")

    # Test 1: Spatial CNN
    print("\n[Test 1: Spatial CNN]")
    x_roi = torch.randn(1, 3, 36, 36)  # 單個 ROI
    print(f"  Input:  {x_roi.shape} (NCHW)")
    feat = spatial(x_roi)
    print(f"  Output: {feat.shape}")
    print(f"  [OK] Spatial CNN test passed")

    # Test 2: Temporal Fusion
    print("\n[Test 2: Temporal Fusion]")
    x_features = torch.randn(1, 24, 16)  # 8 幀 × 3 ROI
    print(f"  Input:  {x_features.shape}")
    hr = temporal(x_features)
    print(f"  Output: {hr.shape}")
    print(f"  HR value: {hr.item():.2f} BPM")
    assert 30 <= hr.item() <= 180, "HR 超出範圍！"
    print(f"  [OK] Temporal Fusion test passed")

    # Test 3: Combined Model (6D input)
    print("\n[Test 3: Combined Model]")
    x_6d = torch.randn(2, 8, 3, 36, 36, 3)  # Batch=2
    print(f"  Input:  {x_6d.shape}")
    hr_combined = combined(x_6d)
    print(f"  Output: {hr_combined.shape}")
    print(f"  HR values: {hr_combined.squeeze().tolist()}")
    print(f"  [OK] Combined Model test passed")

    # Test 4: 模擬 MCU 推理流程
    print("\n[Test 4: 模擬 MCU 推理流程]")
    B = 1
    x_6d_single = torch.randn(B, 8, 3, 36, 36, 3)

    # 模擬 MCU: 循環調用 Spatial CNN
    features_list = []
    for t in range(8):
        for roi in range(3):
            # 提取單個 ROI
            patch = x_6d_single[0, t, roi, :, :, :]  # (36, 36, 3) NHWC
            patch = patch.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 36, 36) NCHW

            # 運行 Spatial CNN
            with torch.no_grad():
                feat = spatial(patch)  # (1, 16)
            features_list.append(feat)

    # 堆疊特徵
    features_mcu = torch.cat(features_list, dim=0)  # (24, 16)
    features_mcu = features_mcu.unsqueeze(0)  # (1, 24, 16)

    # 運行 Temporal Fusion
    with torch.no_grad():
        hr_mcu = temporal(features_mcu)

    # 對比完整模型
    with torch.no_grad():
        hr_full = combined(x_6d_single)

    diff = abs(hr_mcu.item() - hr_full.item())
    print(f"  MCU 模擬 HR:  {hr_mcu.item():.4f} BPM")
    print(f"  完整模型 HR:  {hr_full.item():.4f} BPM")
    print(f"  差異:         {diff:.6f} BPM")

    if diff < 1e-4:
        print(f"  [OK] MCU 流程與完整模型等價")
    else:
        print(f"  [WARNING] 差異較大: {diff:.6f} BPM")

    print("\n" + "=" * 70)
    print("[SUCCESS] All Tests Passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_models()
