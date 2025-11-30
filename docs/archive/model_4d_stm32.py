"""
STM32-Compatible Multi-ROI rPPG Model (4D Input)

修改點:
- 原始輸入: (B, 8, 3, 36, 36, 3) = 6D 張量 ❌ X-CUBE-AI 不支持
- 新輸入: (B, 72, 36, 36) = 4D 張量 ✅ X-CUBE-AI 支持
  - 72 = 8 (時間步) × 3 (ROI) × 3 (RGB通道)
  - 合併 T, ROI, C 三個維度到 channel 維度

架構:
1. 將 4D 輸入 reshape 回 6D (模擬原始邏輯)
2. Shared CNN for each ROI
3. ROI feature fusion
4. Temporal Conv1D modeling
5. Fully connected output

參數量: ~20K (與原始模型相同)
"""

import torch
import torch.nn as nn


class UltraLightRPPG_4D(nn.Module):
    """
    STM32 兼容版本的 Multi-ROI rPPG 模型
    - 接受 4D 輸入: (B, T*ROI*C=72, H=36, W=36)
    - 內部轉換為 6D 處理: (B, T=8, ROI=3, H=36, W=36, C=3)
    - 輸出: (B, 1) - 心率 (BPM)
    """
    def __init__(self, window_size=8, num_rois=3, in_channels=3):
        super(UltraLightRPPG_4D, self).__init__()

        self.window_size = window_size
        self.num_rois = num_rois
        self.in_channels = in_channels
        self.merged_channels = window_size * num_rois * in_channels  # 72

        # Shared 空間特徵提取器（2D CNN，每個 ROI 共享權重）
        self.spatial = nn.Sequential(
            # Layer 1: 3 -> 16
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 36x36 -> 18x18

            # Layer 2: 16 -> 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 18x18 -> 9x9

            # Layer 3: 32 -> 16
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Global Average Pooling
            nn.AdaptiveAvgPool2d(1)  # 9x9 -> 1x1
        )

        # 時序建模（1D Conv）
        self.temporal = nn.Sequential(
            nn.Conv1d(48, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # 輸出層
        self.fc = nn.Sequential(
            nn.Linear(16 * window_size, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

        # 心率範圍約束：[30, 180] BPM
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (B, T*ROI*C, H, W) = (B, 72, 36, 36)  [4D 輸入]

        Returns:
            hr: (B, 1)
        """
        B = x.shape[0]
        H, W = x.shape[2], x.shape[3]

        # Step 1: 將 4D 輸入 reshape 為 6D (B, T, ROI, C, H, W)
        # (B, 72, 36, 36) -> (B, 8, 3, 3, 36, 36) -> (B, 8, 3, 36, 36, 3)
        x = x.view(B, self.window_size, self.num_rois, self.in_channels, H, W)
        x = x.permute(0, 1, 2, 4, 5, 3)  # (B, T, ROI, H, W, C)

        T, ROI, C = x.shape[1], x.shape[2], x.shape[5]

        # Step 2: Reshape 為 (B*T*ROI, C, H, W) 以便 CNN 處理
        x = x.permute(0, 1, 2, 5, 3, 4)  # (B, T, ROI, C, H, W)
        x = x.contiguous().view(B * T * ROI, C, H, W)

        # Step 3: 空間特徵提取（所有 ROI 共享 CNN）
        spatial_feats = self.spatial(x)  # (B*T*ROI, 16, 1, 1)
        spatial_feats = spatial_feats.squeeze(-1).squeeze(-1)  # (B*T*ROI, 16)

        # Step 4: Reshape 回 (B, T, ROI, 16)
        spatial_feats = spatial_feats.view(B, T, ROI, 16)

        # Step 5: ROI 特徵融合（在 ROI 維度上拼接）
        fused_feats = spatial_feats.reshape(B, T, ROI * 16)  # (B, T, 48)

        # Step 6: 轉置以適應 Conv1d: (B, T, 48) -> (B, 48, T)
        fused_feats = fused_feats.transpose(1, 2)  # (B, 48, 8)

        # Step 7: 時序建模
        temporal_feats = self.temporal(fused_feats)  # (B, 16, 8)

        # Step 8: 展平
        temporal_feats = temporal_feats.flatten(1)  # (B, 128)

        # Step 9: 全連接層輸出
        out = self.fc(temporal_feats)  # (B, 1)

        # Step 10: 應用 Sigmoid 並縮放到 [30, 180] BPM
        hr = self.output_act(out) * 150 + 30

        return hr

    def get_num_params(self):
        """計算參數量"""
        return sum(p.numel() for p in self.parameters())


def test_model():
    """測試模型"""
    print("="*70)
    print("Testing UltraLightRPPG_4D Model (STM32-Compatible)")
    print("="*70)

    model = UltraLightRPPG_4D(window_size=8, num_rois=3, in_channels=3)

    # 計算參數量
    total_params = model.get_num_params()
    print(f"\n[OK] Total Parameters: {total_params:,}")
    print(f"   Target: < 500K params -> {'PASS' if total_params < 500000 else 'FAIL'}")

    # 測試前向傳播 (4D 輸入)
    x_4d = torch.randn(2, 72, 36, 36)  # Batch=2, Channels=72, H=36, W=36
    print(f"\n[INPUT] 4D Input shape (STM32-compatible):")
    print(f"   {x_4d.shape} = (Batch, TxROIxC, H, W)")
    print(f"   Batch: {x_4d.shape[0]}")
    print(f"   Channels (TxROIxC): {x_4d.shape[1]} = 8x3x3")
    print(f"   Height: {x_4d.shape[2]}")
    print(f"   Width: {x_4d.shape[3]}")

    y = model(x_4d)
    print(f"\n[OUTPUT] Output shape:")
    print(f"   {y.shape} = (Batch, 1)")
    print(f"   Heart Rate Range: [{y.min().item():.2f}, {y.max().item():.2f}] BPM")

    # 驗證輸出範圍
    assert y.min() >= 30 and y.max() <= 180, "Heart rate out of range!"
    print(f"   [OK] Heart rate within [30, 180] BPM")

    print(f"\n[OK] Model test passed!")

    # 顯示模型結構
    print("\n" + "="*70)
    print("Model Architecture")
    print("="*70)
    print(model)

    # 詳細參數統計
    print("\n" + "="*70)
    print("Parameter Breakdown")
    print("="*70)
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"{name:15s}: {num_params:8,} params")

    print("\n" + "="*70)
    print("[OK] Ready for ONNX export and X-CUBE-AI deployment!")
    print("="*70)


if __name__ == "__main__":
    test_model()
