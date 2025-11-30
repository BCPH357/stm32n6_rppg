"""
測試模型是否正常工作
"""
import torch
import numpy as np
from model import UltraLightRPPG

print("=" * 60)
print("Testing Model")
print("=" * 60)

# 載入模型
model = UltraLightRPPG(window_size=8, num_rois=3)
checkpoint = torch.load('models/best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded successfully")
print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
print(f"MAE: {checkpoint.get('mae', 'unknown'):.4f} BPM")

# 測試 1: 全零輸入
print("\n" + "=" * 60)
print("Test 1: All-zero input")
print("=" * 60)
x_zero = torch.zeros((1, 8, 3, 36, 36, 3), dtype=torch.float32)
with torch.no_grad():
    output_zero = model(x_zero).item()
print(f"Input: all zeros")
print(f"Output: {output_zero:.2f} BPM")
print(f"Expected: Should be around 30 BPM (minimum)")

# 測試 2: 全一輸入
print("\n" + "=" * 60)
print("Test 2: All-one input")
print("=" * 60)
x_one = torch.ones((1, 8, 3, 36, 36, 3), dtype=torch.float32)
with torch.no_grad():
    output_one = model(x_one).item()
print(f"Input: all ones")
print(f"Output: {output_one:.2f} BPM")
print(f"Expected: Should be around 180 BPM (maximum)")

# 測試 3: 隨機輸入
print("\n" + "=" * 60)
print("Test 3: Random input (0-1 range)")
print("=" * 60)
x_random = torch.rand((1, 8, 3, 36, 36, 3), dtype=torch.float32)
with torch.no_grad():
    output_random = model(x_random).item()
print(f"Input: random values in [0, 1]")
print(f"Input mean: {x_random.mean():.3f}")
print(f"Output: {output_random:.2f} BPM")
print(f"Expected: Should be between 30-180 BPM")

# 測試 4: 模擬真實輸入（0.3-0.7 範圍）
print("\n" + "=" * 60)
print("Test 4: Simulated real input (0.3-0.7 range)")
print("=" * 60)
x_real = torch.rand((1, 8, 3, 36, 36, 3), dtype=torch.float32) * 0.4 + 0.3
with torch.no_grad():
    output_real = model(x_real).item()
print(f"Input: random values in [0.3, 0.7]")
print(f"Input mean: {x_real.mean():.3f}")
print(f"Output: {output_real:.2f} BPM")
print(f"Expected: Should be realistic HR (50-120 BPM)")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Zero input  → {output_zero:.2f} BPM")
print(f"One input   → {output_one:.2f} BPM")
print(f"Random      → {output_random:.2f} BPM")
print(f"Real-like   → {output_real:.2f} BPM")

if output_zero > 170 and output_one > 170:
    print("\n[ERROR] Model always outputs ~180 BPM!")
    print("Possible issues:")
    print("  1. Model not properly trained")
    print("  2. Model weights not loaded correctly")
    print("  3. Input data format issue")
elif output_zero < 40 and output_one < 40:
    print("\n[ERROR] Model always outputs ~30 BPM!")
    print("Possible issues:")
    print("  1. Model not properly trained")
    print("  2. Model weights not loaded correctly")
else:
    print("\n[OK] Model responds to different inputs correctly!")
