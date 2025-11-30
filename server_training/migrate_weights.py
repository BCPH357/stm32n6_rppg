"""
權重遷移腳本：從 UltraLightRPPG 遷移到拆分模型

從訓練好的 best_model.pth 中遷移權重到：
- SpatialCNN: checkpoints/spatial_cnn.pth
- TemporalFusion: checkpoints/temporal_fusion.pth

無需重新訓練！直接複製權重。

執行方法:
    python migrate_weights.py
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# 導入模型
from model import UltraLightRPPG
from model_split import SpatialCNN, TemporalFusion, CombinedModel


def migrate_weights():
    """從 UltraLightRPPG 遷移權重到拆分模型"""

    print("=" * 70)
    print("權重遷移：UltraLightRPPG → 拆分模型")
    print("=" * 70)

    # Step 1: 載入訓練好的模型
    checkpoint_path = Path("checkpoints/best_model.pth")

    if not checkpoint_path.exists():
        print(f"\n[ERROR] 找不到檢查點: {checkpoint_path}")
        print("請確認訓練好的模型存在")
        sys.exit(1)

    print(f"\n[Step 1] 載入訓練好的模型")
    print(f"  檢查點: {checkpoint_path}")

    # 載入檢查點
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 創建原始模型
    original_model = UltraLightRPPG(window_size=8, num_rois=3)

    # 載入權重
    if 'model_state_dict' in checkpoint:
        original_model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'N/A')
        mae = checkpoint.get('mae', 'N/A')
        print(f"  Epoch: {epoch}")
        print(f"  MAE: {mae}")
    else:
        original_model.load_state_dict(checkpoint)

    original_model.eval()
    print(f"  [OK] 原始模型載入完成")

    # Step 2: 創建拆分模型
    print(f"\n[Step 2] 創建拆分模型")
    spatial_cnn = SpatialCNN()
    temporal_fusion = TemporalFusion(window_size=8, num_rois=3)

    print(f"  Spatial CNN: {spatial_cnn.get_num_params():,} params")
    print(f"  Temporal Fusion: {temporal_fusion.get_num_params():,} params")

    # Step 3: 遷移 Spatial CNN 權重
    print(f"\n[Step 3] 遷移 Spatial CNN 權重")

    # 原始模型的 spatial 層與 SpatialCNN 的 features 層結構完全相同
    # 直接複製 state_dict
    original_spatial_state = original_model.spatial.state_dict()
    spatial_cnn.features.load_state_dict(original_spatial_state)

    print(f"  [OK] 已複製 {len(original_spatial_state)} 個權重")

    # 驗證參數數量
    orig_params = sum(p.numel() for p in original_model.spatial.parameters())
    spat_params = sum(p.numel() for p in spatial_cnn.parameters())
    assert orig_params == spat_params, "參數數量不匹配！"
    print(f"  [OK] 參數數量驗證通過: {spat_params:,}")

    # Step 4: 遷移 Temporal Fusion 權重
    print(f"\n[Step 4] 遷移 Temporal Fusion 權重")

    # Temporal Conv1D
    original_temporal_state = original_model.temporal.state_dict()
    temporal_fusion.temporal.load_state_dict(original_temporal_state)
    print(f"  [OK] Temporal Conv1D: {len(original_temporal_state)} 個權重")

    # FC 層
    original_fc_state = original_model.fc.state_dict()
    temporal_fusion.fc.load_state_dict(original_fc_state)
    print(f"  [OK] FC 層: {len(original_fc_state)} 個權重")

    # 驗證參數數量
    orig_temp_params = sum(p.numel() for p in original_model.temporal.parameters())
    orig_fc_params = sum(p.numel() for p in original_model.fc.parameters())
    temp_params = sum(p.numel() for p in temporal_fusion.parameters())

    expected_params = orig_temp_params + orig_fc_params
    assert temp_params == expected_params, "參數數量不匹配！"
    print(f"  [OK] 參數數量驗證通過: {temp_params:,}")

    # Step 5: 驗證等價性
    print(f"\n[Step 5] 驗證模型等價性")

    # 創建組合模型
    combined_model = CombinedModel(spatial_cnn, temporal_fusion)
    combined_model.eval()

    # 生成測試數據（使用多個樣本以提高可靠性）
    num_test_samples = 10
    test_inputs = torch.randn(num_test_samples, 8, 3, 36, 36, 3)

    # 推理對比
    with torch.no_grad():
        outputs_original = original_model(test_inputs)
        outputs_combined = combined_model(test_inputs)

    # 計算差異
    diff_abs = (outputs_original - outputs_combined).abs()
    max_diff = diff_abs.max().item()
    mean_diff = diff_abs.mean().item()

    print(f"  測試樣本數: {num_test_samples}")
    print(f"  最大差異: {max_diff:.8f} BPM")
    print(f"  平均差異: {mean_diff:.8f} BPM")

    # 驗證閾值
    if max_diff < 1e-4:
        print(f"  [OK] 模型完全等價 (diff < 1e-4)")
    elif max_diff < 1e-2:
        print(f"  [OK] 模型基本等價 (diff < 1e-2)")
    else:
        print(f"  [WARNING] 差異較大 (diff = {max_diff:.6f})")

    # 顯示一些樣本輸出
    print(f"\n  樣本輸出對比:")
    for i in range(min(5, num_test_samples)):
        orig = outputs_original[i, 0].item()
        comb = outputs_combined[i, 0].item()
        diff = abs(orig - comb)
        print(f"    樣本 {i+1}: Original={orig:7.4f}, Combined={comb:7.4f}, Diff={diff:.6f}")

    # Step 6: 保存拆分模型
    print(f"\n[Step 6] 保存拆分模型")

    # 確保輸出目錄存在
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # 保存 Spatial CNN
    spatial_path = checkpoint_dir / "spatial_cnn.pth"
    torch.save({
        'model_state_dict': spatial_cnn.state_dict(),
        'num_params': spatial_cnn.get_num_params(),
        'architecture': 'SpatialCNN',
        'note': 'Migrated from UltraLightRPPG'
    }, spatial_path)
    print(f"  [OK] Spatial CNN 已保存: {spatial_path}")

    # 保存 Temporal Fusion
    temporal_path = checkpoint_dir / "temporal_fusion.pth"
    torch.save({
        'model_state_dict': temporal_fusion.state_dict(),
        'num_params': temporal_fusion.get_num_params(),
        'architecture': 'TemporalFusion',
        'window_size': 8,
        'num_rois': 3,
        'note': 'Migrated from UltraLightRPPG'
    }, temporal_path)
    print(f"  [OK] Temporal Fusion 已保存: {temporal_path}")

    # 也保存組合模型的檢查點（用於驗證）
    combined_path = checkpoint_dir / "combined_model.pth"
    torch.save({
        'spatial_state_dict': spatial_cnn.state_dict(),
        'temporal_state_dict': temporal_fusion.state_dict(),
        'num_params': combined_model.get_num_params(),
        'original_checkpoint': str(checkpoint_path),
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'note': 'Combined model for validation'
    }, combined_path)
    print(f"  [OK] Combined Model 已保存: {combined_path}")

    # 完成
    print("\n" + "=" * 70)
    print("[SUCCESS] 權重遷移完成！")
    print("=" * 70)
    print(f"\n輸出文件:")
    print(f"  - {spatial_path}")
    print(f"  - {temporal_path}")
    print(f"  - {combined_path}")

    print(f"\n驗證結果:")
    print(f"  - 最大差異: {max_diff:.8f} BPM")
    print(f"  - 平均差異: {mean_diff:.8f} BPM")

    if max_diff < 1e-4:
        print(f"  - 質量: EXCELLENT (完全等價)")
    elif max_diff < 1e-2:
        print(f"  - 質量: GOOD (基本等價)")
    else:
        print(f"  - 質量: WARNING (建議檢查)")

    print(f"\n下一步:")
    print(f"  1. 運行 validate_split.py 進一步驗證")
    print(f"  2. 運行 export_tflite_split.py 導出為 TFLite")
    print("=" * 70)


if __name__ == "__main__":
    migrate_weights()
