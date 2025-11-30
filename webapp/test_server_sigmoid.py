import torch
from model import UltraLightRPPG

model = UltraLightRPPG(window_size=8, num_rois=3)
checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f'Loaded Epoch {checkpoint.get("epoch")}, MAE {checkpoint.get("mae"):.4f}')

# Test with random input
x = torch.rand((1, 8, 3, 36, 36, 3))

# Manual forward pass
B, T, ROI, H, W, C = x.shape
x_reshaped = x.permute(0, 1, 2, 5, 3, 4)
x_reshaped = x_reshaped.contiguous().view(B * T * ROI, C, H, W)

with torch.no_grad():
    spatial_feats = model.spatial(x_reshaped)
    spatial_feats = spatial_feats.squeeze(-1).squeeze(-1)
    spatial_feats = spatial_feats.view(B, T, ROI, 16)

    fused_feats = spatial_feats.reshape(B, T, ROI * 16)
    fused_feats = fused_feats.transpose(1, 2)

    temporal_feats = model.temporal(fused_feats)
    temporal_feats = temporal_feats.flatten(1)

    fc_out = model.fc(temporal_feats)
    print(f'FC output (before Sigmoid): {fc_out.item():.6f}')

    sigmoid_out = model.output_act(fc_out)
    print(f'Sigmoid output: {sigmoid_out.item():.6f}')

    hr = sigmoid_out * 150 + 30
    print(f'Final HR (Sigmoid * 150 + 30): {hr.item():.2f} BPM')

    # Also test direct model call
    hr_direct = model(x)
    print(f'Direct model output: {hr_direct.item():.2f} BPM')
