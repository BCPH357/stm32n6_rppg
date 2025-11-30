"""
Ultra-Lightweight Multi-ROI rPPG Model
參數量: ~60K-100K
輸入: (B, 8, 3, 36, 36, 3) - 8 frames × 3 ROIs × 36×36×3
輸出: (B, 1)

架構:
1. Shared CNN for each ROI region (forehead, left cheek, right cheek)
2. ROI feature fusion (concatenation)
3. Temporal Conv1D modeling
4. Fully connected output layer
"""

import torch
import torch.nn as nn


class UltraLightRPPG(nn.Module):
    """
    超輕量 Multi-ROI rPPG 模型
    - Shared 2D CNN 提取每個 ROI 的空間特徵
    - ROI 特徵融合（拼接）
    - 1D Conv 進行時序建模
    - 全連接層輸出 BVP
    """
    def __init__(self, window_size=8, num_rois=3):
        super(UltraLightRPPG, self).__init__()

        self.window_size = window_size
        self.num_rois = num_rois

        # Shared 空間特徵提取器（2D CNN，每個 ROI 共享權重）
        self.spatial = nn.Sequential(
            # Layer 1: 3 -> 16
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 448 params
            nn.BatchNorm2d(16),                          # 32 params
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 36x36 -> 18x18

            # Layer 2: 16 -> 32
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # 4,640 params
            nn.BatchNorm2d(32),                          # 64 params
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 18x18 -> 9x9

            # Layer 3: 32 -> 16
            nn.Conv2d(32, 16, kernel_size=3, padding=1), # 4,624 params
            nn.BatchNorm2d(16),                          # 32 params
            nn.ReLU(inplace=True),

            # Global Average Pooling
            nn.AdaptiveAvgPool2d(1)                      # 9x9 -> 1x1
        )
        # Total spatial params: ~9,840

        # ROI 特徵提取後每個 ROI 產生 16 維特徵
        # 融合後: 16 × 3 = 48 維

        # 時序建模（1D Conv）
        # 輸入: (B, 48, T=8)
        self.temporal = nn.Sequential(
            nn.Conv1d(48, 32, kernel_size=3, padding=1), # 4,640 params
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 16, kernel_size=3, padding=1), # 1,552 params
            nn.ReLU(inplace=True)
        )
        # Total temporal params: ~6,192

        # 輸出層
        # 輸入: 16 × 8 = 128
        self.fc = nn.Sequential(
            nn.Linear(16 * window_size, 32),             # 4,128 params
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)                             # 33 params
        )
        # Total fc params: ~4,161

        # 心率範圍約束：[30, 180] BPM
        self.output_act = nn.Sigmoid()

        # Total params: ~20,193 (spatial + temporal + fc)

    def forward(self, x):
        """
        Args:
            x: (B, T, ROI, H, W, C) = (B, 8, 3, 36, 36, 3)

        Returns:
            bvp: (B, 1)
        """
        B, T, ROI, H, W, C = x.shape

        # 將 (B, T, ROI, H, W, C) reshape 為 (B*T*ROI, C, H, W) 以便 CNN 處理
        # 注意: PyTorch CNN 需要 (N, C, H, W) 格式
        x = x.permute(0, 1, 2, 5, 3, 4)  # (B, T, ROI, C, H, W)
        x = x.contiguous().view(B * T * ROI, C, H, W)  # (B*T*ROI, 3, 36, 36)

        # 空間特徵提取（所有 ROI 共享 CNN）
        spatial_feats = self.spatial(x)  # (B*T*ROI, 16, 1, 1)
        spatial_feats = spatial_feats.squeeze(-1).squeeze(-1)  # (B*T*ROI, 16)

        # Reshape 回 (B, T, ROI, 16)
        spatial_feats = spatial_feats.view(B, T, ROI, 16)

        # ROI 特徵融合（在 ROI 維度上拼接）
        # (B, T, ROI, 16) -> (B, T, ROI*16) = (B, T, 48)
        fused_feats = spatial_feats.reshape(B, T, ROI * 16)

        # 轉置以適應 Conv1d: (B, T, 48) -> (B, 48, T)
        fused_feats = fused_feats.transpose(1, 2)  # (B, 48, T=8)

        # 時序建模
        temporal_feats = self.temporal(fused_feats)  # (B, 16, T=8)

        # 展平
        temporal_feats = temporal_feats.flatten(1)  # (B, 16*8=128)

        # 全連接層輸出
        out = self.fc(temporal_feats)  # (B, 1)

        # 應用 Sigmoid 並縮放到 [30, 180] BPM
        hr = self.output_act(out) * 150 + 30  # (B, 1)

        return hr

    def get_num_params(self):
        """計算參數量"""
        return sum(p.numel() for p in self.parameters())


def test_model():
    """測試模型"""
    print("="*60)
    print("Testing UltraLightRPPG Multi-ROI Model")
    print("="*60)

    model = UltraLightRPPG(window_size=8, num_rois=3)

    # 計算參數量
    total_params = model.get_num_params()
    print(f"\nTotal Parameters: {total_params:,}")

    # 測試前向傳播
    x = torch.randn(2, 8, 3, 36, 36, 3)  # Batch=2, T=8, ROI=3, H=36, W=36, C=3
    print(f"\nInput shape: {x.shape}")
    print(f"  Batch: {x.shape[0]}")
    print(f"  Time: {x.shape[1]}")
    print(f"  ROI: {x.shape[2]}")
    print(f"  Height: {x.shape[3]}")
    print(f"  Width: {x.shape[4]}")
    print(f"  Channels: {x.shape[5]}")

    y = model(x)
    print(f"\nOutput shape: {y.shape}")
    print(f"  Expected: (2, 1)")

    print(f"\n✅ Model test passed!")

    # 顯示模型結構
    print("\n" + "="*60)
    print("Model Architecture")
    print("="*60)
    print(model)

    # 詳細參數統計
    print("\n" + "="*60)
    print("Parameter Breakdown")
    print("="*60)
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"{name:15s}: {num_params:8,} params")


if __name__ == "__main__":
    test_model()
