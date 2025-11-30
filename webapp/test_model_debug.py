"""
深入調試模型內部
"""
import torch
import numpy as np
from model import UltraLightRPPG

print("=" * 60)
print("Debugging Model Internals")
print("=" * 60)

# 載入模型
model = UltraLightRPPG(window_size=8, num_rois=3)
checkpoint = torch.load('models/best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded: Epoch {checkpoint.get('epoch', 'unknown')}, MAE {checkpoint.get('mae', 'unknown'):.4f}")

# 測試隨機輸入
x = torch.rand((1, 8, 3, 36, 36, 3), dtype=torch.float32)
print(f"\nInput shape: {x.shape}")
print(f"Input range: [{x.min():.3f}, {x.max():.3f}], mean={x.mean():.3f}")

# 手動執行前向傳播並檢查每一步
B, T, ROI, H, W, C = x.shape
print(f"\nBatch={B}, Time={T}, ROI={ROI}, Height={H}, Width={W}, Channels={C}")

# Reshape for CNN
x_reshaped = x.permute(0, 1, 2, 5, 3, 4)  # (B, T, ROI, C, H, W)
x_reshaped = x_reshaped.contiguous().view(B * T * ROI, C, H, W)
print(f"\nAfter reshape: {x_reshaped.shape}")
print(f"  Range: [{x_reshaped.min():.3f}, {x_reshaped.max():.3f}]")

# Spatial features
with torch.no_grad():
    spatial_feats = model.spatial(x_reshaped)
    print(f"\nAfter spatial CNN: {spatial_feats.shape}")
    print(f"  Range: [{spatial_feats.min():.3f}, {spatial_feats.max():.3f}]")
    print(f"  Mean: {spatial_feats.mean():.3f}, Std: {spatial_feats.std():.3f}")

    spatial_feats = spatial_feats.squeeze(-1).squeeze(-1)
    spatial_feats = spatial_feats.view(B, T, ROI, 16)
    print(f"\nAfter view: {spatial_feats.shape}")

    # ROI fusion
    fused_feats = spatial_feats.reshape(B, T, ROI * 16)
    fused_feats = fused_feats.transpose(1, 2)
    print(f"\nAfter ROI fusion: {fused_feats.shape}")
    print(f"  Range: [{fused_feats.min():.3f}, {fused_feats.max():.3f}]")

    # Temporal modeling
    temporal_feats = model.temporal(fused_feats)
    print(f"\nAfter temporal Conv1D: {temporal_feats.shape}")
    print(f"  Range: [{temporal_feats.min():.3f}, {temporal_feats.max():.3f}]")
    print(f"  Mean: {temporal_feats.mean():.3f}, Std: {temporal_feats.std():.3f}")

    # Flatten
    temporal_feats = temporal_feats.flatten(1)
    print(f"\nAfter flatten: {temporal_feats.shape}")

    # FC layers
    out = model.fc(temporal_feats)
    print(f"\nAfter FC layers (before Sigmoid): {out.shape}")
    print(f"  Value: {out.item():.6f}")
    print(f"  Range: [{out.min():.6f}, {out.max():.6f}]")

    # Sigmoid
    sigmoid_out = model.output_act(out)
    print(f"\nAfter Sigmoid: {sigmoid_out.item():.6f}")

    # Final HR
    hr = sigmoid_out * 150 + 30
    print(f"\nFinal HR: {hr.item():.2f} BPM")

    # 分析
    print("\n" + "=" * 60)
    print("Analysis")
    print("=" * 60)

    if out.item() > 10:
        print(f"[ERROR] FC output is TOO HIGH: {out.item():.3f}")
        print("  → Sigmoid({out:.3f}) ≈ 1.0 → HR = 180 BPM")
        print("\nPossible causes:")
        print("  1. FC layer weights are too large")
        print("  2. Model not properly trained (collapsed to maximum)")
        print("  3. Normalization issue during training")
    elif out.item() < -10:
        print(f"[ERROR] FC output is TOO LOW: {out.item():.3f}")
        print("  → Sigmoid({out:.3f}) ≈ 0.0 → HR = 30 BPM")
    else:
        print(f"[OK] FC output is in reasonable range: {out.item():.3f}")
        print(f"  → Sigmoid({out:.3f}) = {sigmoid_out.item():.3f}")
        print(f"  → HR = {hr.item():.2f} BPM")
