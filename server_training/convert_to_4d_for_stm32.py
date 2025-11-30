"""
服務器端：將 6D 模型轉換為 4D ONNX (STM32 兼容)

用法 (在服務器上執行):
    cd /mnt/data_8T/ChenPinHao/server_training/
    python convert_to_4d_for_stm32.py

輸出:
    models/rppg_4d_fp32.onnx  - FP32 ONNX (用於 INT8 量化)
"""

import torch
import torch.onnx
import onnx
import numpy as np
from pathlib import Path

# 導入模型
from model import UltraLightRPPG as Model_6D


class UltraLightRPPG_4D(torch.nn.Module):
    """STM32 兼容的 4D 輸入版本"""
    def __init__(self, model_6d):
        super().__init__()

        # 直接複製所有層（權重共享）
        self.spatial = model_6d.spatial
        self.temporal = model_6d.temporal
        self.fc = model_6d.fc
        self.output_act = model_6d.output_act

        self.window_size = model_6d.window_size
        self.num_rois = model_6d.num_rois
        self.in_channels = 3

    def forward(self, x):
        """
        Args:
            x: (B, 72, 36, 36) - 4D 輸入
        Returns:
            hr: (B, 1)
        """
        B, _, H, W = x.shape
        T, ROI, C = 8, 3, 3

        # Reshape: (B, 72, 36, 36) -> (B, 8, 3, 3, 36, 36) -> (B, 8, 3, 36, 36, 3)
        x = x.view(B, T, ROI, C, H, W)
        x = x.permute(0, 1, 2, 4, 5, 3)  # (B, T, ROI, H, W, C)

        # 原始 6D 模型邏輯
        x = x.permute(0, 1, 2, 5, 3, 4)  # (B, T, ROI, C, H, W)
        x = x.contiguous().view(B * T * ROI, C, H, W)

        spatial_feats = self.spatial(x)
        spatial_feats = spatial_feats.squeeze(-1).squeeze(-1)
        spatial_feats = spatial_feats.view(B, T, ROI, 16)

        fused_feats = spatial_feats.reshape(B, T, ROI * 16)
        fused_feats = fused_feats.transpose(1, 2)

        temporal_feats = self.temporal(fused_feats)
        temporal_feats = temporal_feats.flatten(1)

        out = self.fc(temporal_feats)
        hr = self.output_act(out) * 150 + 30

        return hr


def main():
    print("="*70)
    print("Convert 6D Model to 4D ONNX for STM32 (Server-Side)")
    print("="*70)

    # 路徑
    checkpoint_path = Path("checkpoints/best_model.pth")
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    onnx_path = output_dir / "rppg_4d_fp32.onnx"

    # 載入 6D 模型
    print("\n[Step 1] Loading trained 6D model...")
    model_6d = Model_6D(window_size=8, num_rois=3)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model_6d.load_state_dict(checkpoint['model_state_dict'])
    else:
        model_6d.load_state_dict(checkpoint)

    model_6d.eval()
    print(f"   [OK] Loaded: {checkpoint_path}")
    print(f"   Parameters: {sum(p.numel() for p in model_6d.parameters()):,}")

    # 創建 4D 模型（共享權重）
    print("\n[Step 2] Creating 4D model (shared weights)...")
    model_4d = UltraLightRPPG_4D(model_6d)
    model_4d.eval()
    print(f"   [OK] 4D model created")

    # 驗證等價性
    print("\n[Step 3] Verifying equivalence...")
    x_6d = torch.randn(1, 8, 3, 36, 36, 3)
    x_4d = x_6d.permute(0, 1, 2, 5, 3, 4).reshape(1, 72, 36, 36)

    with torch.no_grad():
        y_6d = model_6d(x_6d).item()
        y_4d = model_4d(x_4d).item()

    diff = abs(y_6d - y_4d)
    print(f"   6D output: {y_6d:.4f} BPM")
    print(f"   4D output: {y_4d:.4f} BPM")
    print(f"   Difference: {diff:.6f} BPM")
    assert diff < 1e-5, f"Models not equivalent! Diff: {diff}"
    print(f"   [OK] Models are equivalent")

    # 導出 ONNX
    print("\n[Step 4] Exporting to ONNX...")
    dummy_input = torch.randn(1, 72, 36, 36)

    torch.onnx.export(
        model_4d,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch'},
            'output': {0: 'batch'}
        }
    )

    print(f"   [OK] Exported: {onnx_path}")

    # 驗證 ONNX
    print("\n[Step 5] Verifying ONNX...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # 檢查 shape
    input_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [d.dim_value for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]
    print(f"   Input shape: {input_shape} (batch=0 means dynamic)")
    print(f"   Output shape: {output_shape}")
    print(f"   [OK] ONNX validation passed")

    # 完成
    print("\n" + "="*70)
    print("[SUCCESS] Conversion Complete!")
    print("="*70)
    print(f"\nOutput file: {onnx_path}")
    print(f"Size: {onnx_path.stat().st_size / 1024:.2f} KB")
    print(f"\nNext steps:")
    print(f"  1. Quantize to INT8:")
    print(f"     cd /mnt/data_8T/ChenPinHao/server_training/")
    print(f"     python quantize_4d_model.py")
    print(f"  2. Download quantized model:")
    print(f"     scp miat@140.115.53.67:/mnt/data_8T/ChenPinHao/server_training/models/rppg_4d_int8.onnx D:\\MIAT\\rppg\\")
    print(f"  3. Import into STM32CubeMX (use O1 or O2, avoid O3)")
    print("="*70)


if __name__ == "__main__":
    main()
