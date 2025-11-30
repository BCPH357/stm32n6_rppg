"""簡化的 ONNX 導出腳本（無 emoji）"""
import torch
import torch.onnx
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server_training.model import UltraLightRPPG

def export_to_onnx(checkpoint_path, output_path, opset_version=13):
    print("="*70)
    print("Exporting PyTorch Model to FP32 ONNX")
    print("="*70)

    print(f"\n[1/5] Loading PyTorch model from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return False

    model = UltraLightRPPG(window_size=8, num_rois=3)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint:
            print(f"   Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")

    print(f"\n[2/5] Preparing dummy input...")
    dummy_input = torch.randn(1, 8, 3, 36, 36, 3)
    print(f"   Input shape: {dummy_input.shape}")

    print(f"\n[3/5] Testing forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
    print(f"   Output shape: {output.shape}")
    print(f"   Output value: {output.item():.2f} BPM")

    print(f"\n[4/5] Exporting to ONNX (opset {opset_version})...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.onnx.export(
        model,
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
        },
        verbose=False
    )

    print(f"   Exported to: {output_path}")

    print(f"\n[5/5] Validating ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"   ONNX model validation passed")

        file_size_mb = os.path.getsize(output_path) / (1024**2)
        print(f"\n   File size: {file_size_mb:.2f} MB")

    except ImportError:
        print(f"   Warning: onnx package not found, skipping validation")
    except Exception as e:
        print(f"   Error during validation: {e}")
        return False

    print("\n" + "="*70)
    print("FP32 ONNX export completed successfully!")
    print("="*70)
    print(f"\nOutput: {output_path}")
    print("\nNext step: Run quantization")
    print(f"  python quantize_onnx_simple.py")
    print("="*70)

    return True

if __name__ == "__main__":
    success = export_to_onnx(
        checkpoint_path='../webapp/models/best_model.pth',
        output_path='models/rppg_fp32.onnx'
    )
    sys.exit(0 if success else 1)
