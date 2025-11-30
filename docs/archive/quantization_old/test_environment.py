"""
測試量化環境和依賴

檢查所有必要的套件是否已安裝
"""

import sys

def test_imports():
    """測試必要的套件導入"""
    print("="*70)
    print("Testing Quantization Environment")
    print("="*70)

    tests = [
        ("torch", "PyTorch"),
        ("onnx", "ONNX"),
        ("onnxruntime", "ONNX Runtime"),
        ("numpy", "NumPy"),
    ]

    all_passed = True

    for module, name in tests:
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {name:20s} version {version}")
        except ImportError as e:
            print(f"❌ {name:20s} NOT FOUND - {e}")
            all_passed = False

    print("\n" + "="*70)

    if all_passed:
        print("✅ All dependencies are installed!")
        print("\nReady to run quantization workflow:")
        print("  1. Prepare calibration data")
        print("  2. Export FP32 ONNX")
        print("  3. Quantize to INT8")
        print("  4. Verify accuracy")
    else:
        print("❌ Some dependencies are missing!")
        print("\nPlease install missing packages:")
        print("  conda activate zerodce_tf")
        print("  pip install onnx onnxruntime torch")

    print("="*70)

    return all_passed

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
