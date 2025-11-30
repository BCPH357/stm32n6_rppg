"""
FP32 ONNX 模型導出工具
將訓練好的 PyTorch Multi-ROI rPPG 模型導出為 ONNX 格式

使用方式:
    conda activate zerodce_tf
    python export_onnx.py
"""

import torch
import torch.onnx
import os
import sys

# 添加父目錄到路徑以導入 model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 從父目錄導入 model（server_training/model.py）
from model import UltraLightRPPG


def export_to_onnx(
    checkpoint_path='../checkpoints/best_model.pth',
    output_path='models/rppg_fp32.onnx',
    opset_version=13
):
    """
    導出 FP32 ONNX 模型

    重要配置：
    - opset_version=13：支持 BatchNorm folding（X-CUBE-AI 推薦）
    - do_constant_folding=True：優化常量運算
    - input_names/output_names：便於 X-CUBE-AI 識別

    Args:
        checkpoint_path: PyTorch 模型檢查點路徑
        output_path: ONNX 模型保存路徑
        opset_version: ONNX opset 版本（推薦 13）

    Returns:
        bool: 導出是否成功
    """
    print("="*70)
    print("Exporting PyTorch Model to FP32 ONNX")
    print("="*70)

    # [1/5] 載入訓練好的模型
    print(f"\n[1/5] Loading PyTorch model from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        print(f"   Please ensure model training is completed.")
        return False

    model = UltraLightRPPG(window_size=8, num_rois=3)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 處理不同的檢查點格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint:
            print(f"   Loaded checkpoint from epoch {checkpoint['epoch']}")
        if 'val_mae' in checkpoint:
            print(f"   Validation MAE: {checkpoint['val_mae']:.4f} BPM")
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 計算參數量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")

    # [2/5] 準備 dummy input
    print(f"\n[2/5] Preparing dummy input...")
    dummy_input = torch.randn(1, 8, 3, 36, 36, 3)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Input dtype: {dummy_input.dtype}")

    # [3/5] 測試前向傳播
    print(f"\n[3/5] Testing forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   Output shape: {output.shape}")
    print(f"   Output value: {output.item():.2f} BPM")

    # [4/5] 導出 ONNX
    print(f"\n[4/5] Exporting to ONNX (opset {opset_version})...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,  # 關鍵！支持 BN folding
        do_constant_folding=True,     # 優化常量運算
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch'},
            'output': {0: 'batch'}
        },
        verbose=False
    )

    print(f"   ✅ Exported to: {output_path}")

    # [5/5] 驗證導出的 ONNX 模型
    print(f"\n[5/5] Validating ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"   ✅ ONNX model validation passed")

        # 顯示模型信息
        print(f"\n   Model info:")
        print(f"     IR version: {onnx_model.ir_version}")
        print(f"     Opset version: {onnx_model.opset_import[0].version}")
        print(f"     Producer: {onnx_model.producer_name} {onnx_model.producer_version}")

        # 輸入/輸出信息
        input_tensor = onnx_model.graph.input[0]
        output_tensor = onnx_model.graph.output[0]
        print(f"\n   Input tensor:")
        print(f"     Name: {input_tensor.name}")
        print(f"     Shape: {[dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]}")

        print(f"\n   Output tensor:")
        print(f"     Name: {output_tensor.name}")
        print(f"     Shape: {[dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]}")

        # 文件大小
        file_size_mb = os.path.getsize(output_path) / (1024**2)
        print(f"\n   File size: {file_size_mb:.2f} MB")

    except ImportError:
        print(f"   ⚠️  Warning: onnx package not found, skipping validation")
    except Exception as e:
        print(f"   ❌ Error during validation: {e}")
        return False

    print("\n" + "="*70)
    print("✅ FP32 ONNX export completed successfully!")
    print("="*70)
    print(f"\nNext step: Run quantization")
    print(f"  python quantize_onnx.py")
    print("="*70)

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Export PyTorch model to FP32 ONNX')
    parser.add_argument('--checkpoint', type=str,
                        default='../checkpoints/best_model.pth',
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, default='models/rppg_fp32.onnx',
                        help='Output path for ONNX model')
    parser.add_argument('--opset', type=int, default=13,
                        help='ONNX opset version (default: 13)')

    args = parser.parse_args()

    success = export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset
    )

    sys.exit(0 if success else 1)
