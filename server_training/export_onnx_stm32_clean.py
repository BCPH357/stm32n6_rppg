"""
STM32N6 兼容 ONNX 導出腳本 - 從頭避免 6D 張量

策略: 修改 forward() 以避免產生任何 >4D 的中間張量

用法 (在服務器上執行):
    cd /mnt/data_8T/ChenPinHao/server_training/
    python export_onnx_stm32_clean.py

輸出:
    models/rppg_stm32_clean_fp32.onnx  - 完全避免 6D 張量的 ONNX
"""

import torch
import torch.nn as nn
import onnx
from pathlib import Path

# 導入原始模型
from model import UltraLightRPPG as Model_6D


class UltraLightRPPG_STM32Clean(nn.Module):
    """
    STM32 兼容版本 - 完全避免 6D 張量的 forward()

    與 model_4d_stm32.py 的差異:
    - model_4d_stm32.py: 內部有 6D reshape (違規)
    - 本版本: 使用替代邏輯避免 6D 張量
    """

    def __init__(self, model_6d):
        super().__init__()

        # 複製權重
        self.spatial = model_6d.spatial
        self.temporal = model_6d.temporal
        self.fc = model_6d.fc
        self.output_act = model_6d.output_act

        self.window_size = model_6d.window_size
        self.num_rois = model_6d.num_rois
        self.in_channels = 3

    def forward(self, x):
        """
        STM32-Clean Forward Pass - 避免所有 6D 張量

        輸入: (B, 72, 36, 36)
        策略: 使用 reshape 和 permute 的組合，但保持所有中間張量 ≤4D

        Args:
            x: (B, 72, H, W) where 72 = T*ROI*C = 8*3*3

        Returns:
            hr: (B, 1)
        """
        B, merged_C, H, W = x.shape  # (B, 72, 36, 36)

        # 關鍵: 不要 reshape 到 6D！
        # 原始: x.view(B, 8, 3, 3, 36, 36)  <- 6D 違規
        # 替代: 使用 4D reshape 序列

        # Step 1: 拆分 72 channels → 24 個 (T*ROI) 組，每組 3 channels
        # (B, 72, H, W) → (B*24, 3, H, W)
        x = x.view(B * 24, 3, H, W)  # ✅ 4D (24 = 8*3)

        # Step 2: 通過 shared spatial CNN
        spatial_feats = self.spatial(x)  # (B*24, 16, 1, 1)
        spatial_feats = spatial_feats.squeeze(-1).squeeze(-1)  # (B*24, 16)

        # Step 3: Reshape 回 (B, T*ROI, 16) = (B, 24, 16)
        spatial_feats = spatial_feats.view(B, 24, 16)  # ✅ 3D

        # Step 4: Reshape 為 (B, T, ROI*16) = (B, 8, 48)
        # 原理: 24 個時間步 * ROI → 8 個時間步，每個有 3*16=48 features
        spatial_feats = spatial_feats.view(B, 8, 3 * 16)  # ✅ 3D: (B, 8, 48)

        # Step 5: Transpose 為 Conv1d 格式: (B, 48, 8)
        fused_feats = spatial_feats.transpose(1, 2)  # ✅ 3D: (B, 48, 8)

        # Step 6: Temporal modeling
        temporal_feats = self.temporal(fused_feats)  # ✅ 3D: (B, 16, 8)

        # Step 7: Flatten
        temporal_feats = temporal_feats.flatten(1)  # ✅ 2D: (B, 128)

        # Step 8: FC layers
        out = self.fc(temporal_feats)  # ✅ 2D: (B, 1)

        # Step 9: Output activation
        hr = self.output_act(out) * 150 + 30  # ✅ 2D: (B, 1)

        return hr


def main():
    print("="*70)
    print("Export STM32-Clean ONNX (No 6D Tensors)")
    print("="*70)

    # 路徑
    checkpoint_path = Path("checkpoints/best_model.pth")
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    onnx_path = output_dir / "rppg_stm32_clean_fp32.onnx"

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

    # 創建 STM32-Clean 模型
    print("\n[Step 2] Creating STM32-Clean model (no 6D tensors)...")
    model_stm32 = UltraLightRPPG_STM32Clean(model_6d)
    model_stm32.eval()
    print(f"   [OK] STM32-Clean model created")

    # 驗證等價性
    print("\n[Step 3] Verifying equivalence...")
    x_6d = torch.randn(1, 8, 3, 36, 36, 3)
    x_4d = x_6d.permute(0, 1, 2, 5, 3, 4).reshape(1, 72, 36, 36)

    with torch.no_grad():
        y_6d = model_6d(x_6d).item()
        y_stm32 = model_stm32(x_4d).item()

    diff = abs(y_6d - y_stm32)
    print(f"   6D model output: {y_6d:.4f} BPM")
    print(f"   STM32 model output: {y_stm32:.4f} BPM")
    print(f"   Difference: {diff:.6f} BPM")

    if diff < 1e-4:
        print(f"   ✅ Models are equivalent")
    else:
        print(f"   ⚠️  WARNING: Difference = {diff:.6f} (may need investigation)")

    # 導出 ONNX
    print("\n[Step 4] Exporting to ONNX...")
    dummy_input = torch.randn(1, 72, 36, 36)

    torch.onnx.export(
        model_stm32,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,  # STM32N6 recommended
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        # ✅ 無 dynamic_axes（固定 batch=1）
        verbose=False
    )

    print(f"   [OK] Exported: {onnx_path}")

    # 驗證 ONNX
    print("\n[Step 5] Validating ONNX...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"   [OK] ONNX validation passed")

    # 檢查形狀
    print("\n[Step 6] Checking shapes...")
    for input_tensor in onnx_model.graph.input:
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"   Input '{input_tensor.name}': {shape}")

    for output_tensor in onnx_model.graph.output:
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"   Output '{output_tensor.name}': {shape}")

    # 完成
    print("\n" + "="*70)
    print("[SUCCESS] Clean ONNX Export Complete!")
    print("="*70)
    print(f"\nOutput: {onnx_path}")
    print(f"Size: {onnx_path.stat().st_size / 1024:.2f} KB")

    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print(f"{'='*70}")
    print(f"1. Diagnose the exported model:")
    print(f"   python diagnose_onnx_stm32.py --onnx {onnx_path}")
    print(f"")
    print(f"2. If there are still violations, apply graph surgery:")
    print(f"   python fix_onnx_for_stm32.py --input {onnx_path} --output models/rppg_stm32_clean_fixed.onnx")
    print(f"")
    print(f"3. Quantize to INT8:")
    print(f"   python quantize_4d_model_v2.py")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
