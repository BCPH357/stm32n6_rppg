import torch
import numpy as np
from model import UltraLightRPPG

model = UltraLightRPPG(window_size=8, num_rois=3)
checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

epoch = checkpoint.get('epoch', 'unknown')
mae = checkpoint.get('mae', 0)
print(f'Loaded Epoch {epoch}, MAE {mae:.4f}')

# Test 1: All zeros
x_zero = torch.zeros((1, 8, 3, 36, 36, 3))
with torch.no_grad():
    out_zero = model(x_zero).item()
print(f'Zero input -> {out_zero:.2f} BPM')

# Test 2: All ones
x_one = torch.ones((1, 8, 3, 36, 36, 3))
with torch.no_grad():
    out_one = model(x_one).item()
print(f'One input  -> {out_one:.2f} BPM')

# Test 3: Random
x_rand = torch.rand((1, 8, 3, 36, 36, 3))
with torch.no_grad():
    out_rand = model(x_rand).item()
print(f'Random     -> {out_rand:.2f} BPM')
