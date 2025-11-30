"""
將 6D 輸入模型權重轉換為 4D 輸入模型，並導出為 ONNX

原始模型:
- 輸入: (B, 8, 3, 36, 36, 3) = 6D ❌ X-CUBE-AI 不支持

新模型:
- 輸入: (B, 72, 36, 36) = 4D ✅ X-CUBE-AI 支持
- 內部邏輯完全相同，只是輸入形狀不同

用法:
    python convert_6d_to_4d_onnx.py
"""

import sys
sys.path.append('D:\\MIAT\\rppg\\server_training')

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

# 導入原始 6D 模型和新 4D 模型
from server_training.model import UltraLightRPPG as Model_6D
from model_4d_stm32 import UltraLightRPPG_4D as Model_4D


def load_6d_model(checkpoint_path):
    """載入原始 6D 模型權重"""
    print("\n" + "="*70)
    print("[Step 1] Loading 6D Model Checkpoint")
    print("="*70)

    model_6d = Model_6D(window_size=8, num_rois=3)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 處理不同的 checkpoint 格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_6d.load_state_dict(state_dict)
    model_6d.eval()

    print(f"[OK] Loaded checkpoint: {checkpoint_path}")
    print(f"   Parameters: {model_6d.get_num_params():,}")

    return model_6d


def transfer_weights(model_6d, model_4d):
    """將 6D 模型權重轉移到 4D 模型"""
    print("\n" + "="*70)
    print("[Step 2] Transferring Weights (6D -> 4D)")
    print("="*70)

    # 獲取權重字典
    state_dict_6d = model_6d.state_dict()
    state_dict_4d = model_4d.state_dict()

    # 驗證所有權重名稱相同（應該完全一樣）
    keys_6d = set(state_dict_6d.keys())
    keys_4d = set(state_dict_4d.keys())

    assert keys_6d == keys_4d, "Model architectures mismatch!"

    # 直接複製所有權重
    model_4d.load_state_dict(state_dict_6d)

    print(f"[OK] Transferred {len(state_dict_6d)} weight tensors")
    print(f"   Weight names match: {len(keys_6d)} / {len(keys_6d)}")

    return model_4d


def verify_equivalence(model_6d, model_4d):
    """驗證兩個模型輸出等價"""
    print("\n" + "="*70)
    print("[Step 3] Verifying Model Equivalence")
    print("="*70)

    model_6d.eval()
    model_4d.eval()

    # 生成隨機 6D 輸入
    x_6d = torch.randn(1, 8, 3, 36, 36, 3)

    # 轉換為 4D 輸入 (B, T, ROI, H, W, C) -> (B, T*ROI*C, H, W)
    B, T, ROI, H, W, C = x_6d.shape
    x_4d = x_6d.permute(0, 1, 2, 5, 3, 4)  # (B, T, ROI, C, H, W)
    x_4d = x_4d.reshape(B, T*ROI*C, H, W)  # (B, 72, 36, 36)

    # 推論
    with torch.no_grad():
        y_6d = model_6d(x_6d)
        y_4d = model_4d(x_4d)

    # 計算誤差
    diff = torch.abs(y_6d - y_4d).item()
    rel_diff = (diff / y_6d.item()) * 100

    print(f"[INPUT]")
    print(f"   6D Input: {x_6d.shape}")
    print(f"   4D Input: {x_4d.shape}")
    print(f"\n[OUTPUT]")
    print(f"   6D Model Output: {y_6d.item():.4f} BPM")
    print(f"   4D Model Output: {y_4d.item():.4f} BPM")
    print(f"\n[DIFFERENCE]")
    print(f"   Absolute: {diff:.6f} BPM")
    print(f"   Relative: {rel_diff:.6f} %")

    # 驗證誤差在容忍範圍內（應該完全相同）
    assert diff < 1e-5, f"Models are not equivalent! Diff: {diff}"
    print(f"\n[OK] Models are equivalent (diff < 1e-5)")

    return True


def export_onnx(model_4d, output_path, opset_version=13):
    """導出 4D 模型為 ONNX"""
    print("\n" + "="*70)
    print("[Step 4] Exporting to ONNX (FP32)")
    print("="*70)

    model_4d.eval()

    # 準備 dummy 輸入
    dummy_input = torch.randn(1, 72, 36, 36)

    # 導出 ONNX
    torch.onnx.export(
        model_4d,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch'},
            'output': {0: 'batch'}
        }
    )

    print(f"[OK] ONNX exported to: {output_path}")
    print(f"   Opset version: {opset_version}")
    print(f"   Input: (batch, 72, 36, 36)")
    print(f"   Output: (batch, 1)")

    # 驗證 ONNX 模型
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"[OK] ONNX model validation passed")

    return output_path


def verify_onnx(onnx_path, model_4d):
    """驗證 ONNX 模型與 PyTorch 模型輸出一致"""
    print("\n" + "="*70)
    print("[Step 5] Verifying ONNX Inference")
    print("="*70)

    # 載入 ONNX 模型
    ort_session = ort.InferenceSession(onnx_path)

    # 準備輸入
    x = np.random.randn(1, 72, 36, 36).astype(np.float32)

    # PyTorch 推論
    model_4d.eval()
    with torch.no_grad():
        y_torch = model_4d(torch.from_numpy(x)).numpy()

    # ONNX 推論
    y_onnx = ort_session.run(None, {'input': x})[0]

    # 比較結果
    diff = np.abs(y_torch - y_onnx).max()
    rel_diff = (diff / np.abs(y_torch).max()) * 100

    print(f"[OUTPUT]")
    print(f"   PyTorch: {y_torch[0, 0]:.4f} BPM")
    print(f"   ONNX:    {y_onnx[0, 0]:.4f} BPM")
    print(f"\n[DIFFERENCE]")
    print(f"   Max Absolute: {diff:.6f}")
    print(f"   Max Relative: {rel_diff:.6f} %")

    assert diff < 1e-4, f"ONNX output mismatch! Diff: {diff}"
    print(f"\n[OK] ONNX inference verified (diff < 1e-4)")

    return True


def main():
    """主流程"""
    print("="*70)
    print("Converting 6D Model to 4D ONNX for STM32 Deployment")
    print("="*70)

    # 路徑配置
    checkpoint_path = Path("D:/MIAT/rppg/webapp/models/best_model.pth")
    onnx_output_path = Path("D:/MIAT/rppg/webapp/models/rppg_4d_fp32.onnx")

    # 檢查輸入文件
    if not checkpoint_path.exists():
        print(f"\n[ERROR] Checkpoint not found: {checkpoint_path}")
        print("   Please train the model first or check the path.")
        return

    # Step 1: 載入 6D 模型
    model_6d = load_6d_model(checkpoint_path)

    # Step 2: 創建 4D 模型並轉移權重
    model_4d = Model_4D(window_size=8, num_rois=3, in_channels=3)
    model_4d = transfer_weights(model_6d, model_4d)

    # Step 3: 驗證等價性
    verify_equivalence(model_6d, model_4d)

    # Step 4: 導出 ONNX
    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(model_4d, str(onnx_output_path), opset_version=13)

    # Step 5: 驗證 ONNX
    verify_onnx(str(onnx_output_path), model_4d)

    # 完成
    print("\n" + "="*70)
    print("[SUCCESS] Conversion Complete!")
    print("="*70)
    print(f"\n[OUTPUT FILE]")
    print(f"   {onnx_output_path}")
    print(f"   Size: {onnx_output_path.stat().st_size / 1024:.2f} KB")
    print(f"\n[NEXT STEPS]")
    print(f"   1. Import {onnx_output_path.name} into STM32CubeMX")
    print(f"   2. Configure X-CUBE-AI:")
    print(f"      - Optimization: Time (O2) or Default (O1)")
    print(f"      - Avoid: Balanced (O3)")
    print(f"   3. Generate code and verify inference")
    print("="*70)


if __name__ == "__main__":
    main()
