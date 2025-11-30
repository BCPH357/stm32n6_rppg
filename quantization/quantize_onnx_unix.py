"""
ONNX 模型 INT8 量化工具
使用 ONNX Runtime 進行 Post-Training Quantization (PTQ)

使用方式:
    conda activate zerodce_tf
    python quantize_onnx.py
"""

from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader
import onnxruntime as ort
import numpy as np
import torch
import os
import sys


class RPPGCalibrationDataReader(CalibrationDataReader):
    """
    rPPG 校準數據讀取器

    用於 ONNX Runtime 量化過程中提供校準數據
    """
    def __init__(self, calibration_data_path='calibration_data.pt'):
        """
        初始化校準數據讀取器

        Args:
            calibration_data_path: 校準數據文件路徑
        """
        print(f"   Loading calibration data from: {calibration_data_path}")

        if not os.path.exists(calibration_data_path):
            raise FileNotFoundError(
                f"Calibration data not found: {calibration_data_path}\n"
                f"Please run: python quantize_utils.py"
            )

        self.data = torch.load(calibration_data_path)
        self.samples = self.data['samples'].numpy()  # (N, 8, 3, 36, 36, 3)
        self.num_samples = self.data['num_samples']
        self.current_idx = 0

        print(f"   Loaded {self.num_samples} calibration samples")
        print(f"   HR range: {self.data['hr_range'][0]:.2f} - {self.data['hr_range'][1]:.2f} BPM")

    def get_next(self):
        """
        獲取下一個校準樣本

        Returns:
            dict: {input_name: numpy_array} 或 None（無更多數據）
        """
        if self.current_idx >= self.num_samples:
            return None

        # 返回字典 {input_name: numpy_array}
        sample = self.samples[self.current_idx:self.current_idx+1]
        self.current_idx += 1

        return {'input': sample.astype(np.float32)}

    def rewind(self):
        """重置讀取索引"""
        self.current_idx = 0


def quantize_onnx_model(
    model_input='models/rppg_fp32.onnx',
    model_output='models/rppg_int8_qdq.onnx',
    calibration_data_path='calibration_data.pt',
    per_channel=True
):
    """
    使用 ONNX Runtime 進行 INT8 量化（QDQ 格式）

    關鍵配置：
    - per_channel=True：每通道量化（準確度更高）
    - weight_type=INT8：權重 INT8
    - activation_type=INT8：激活 INT8
    - optimize_model=False：避免 X-CUBE-AI 解析問題

    Args:
        model_input: 輸入 FP32 ONNX 模型路徑
        model_output: 輸出 INT8 ONNX 模型路徑
        calibration_data_path: 校準數據路徑
        per_channel: 是否使用 per-channel 量化

    Returns:
        bool: 量化是否成功
    """
    print("="*70)
    print("Quantizing ONNX Model to INT8 (QDQ Format)")
    print("="*70)

    # [1/4] 檢查輸入文件
    print(f"\n[1/4] Checking input files...")
    if not os.path.exists(model_input):
        print(f"❌ Error: FP32 ONNX model not found: {model_input}")
        print(f"   Please run: python export_onnx.py")
        return False

    if not os.path.exists(calibration_data_path):
        print(f"❌ Error: Calibration data not found: {calibration_data_path}")
        print(f"   Please run: python quantize_utils.py")
        return False

    print(f"   ✅ FP32 model: {model_input}")
    print(f"   ✅ Calibration data: {calibration_data_path}")

    # [2/4] 創建校準數據讀取器
    print(f"\n[2/4] Preparing calibration data reader...")
    try:
        calibration_reader = RPPGCalibrationDataReader(calibration_data_path)
    except Exception as e:
        print(f"❌ Error loading calibration data: {e}")
        return False

    # [3/4] 執行量化
    print(f"\n[3/4] Running INT8 quantization...")
    print(f"   Quantization format: QDQ (Quantize-DeQuantize)")
    print(f"   Weight type: INT8")
    print(f"   Activation type: INT8")
    print(f"   Per-channel: {per_channel}")

    try:
        quantize_static(
            model_input,
            model_output,
            calibration_data_reader=calibration_reader,
            quant_format=QuantFormat.QDQ,      # QDQ 格式（X-CUBE-AI 支持）
            per_channel=per_channel,           # Per-channel 量化
            weight_type=QuantType.QInt8,       # INT8 權重
            activation_type=QuantType.QInt8,   # INT8 激活
            use_external_data_format=False
        )
        print(f"   ✅ Quantization completed")
    except Exception as e:
        print(f"❌ Error during quantization: {e}")
        import traceback
        traceback.print_exc()
        return False

    # [4/4] 驗證量化模型
    print(f"\n[4/4] Validating quantized model...")
    try:
        import onnx
        quant_model = onnx.load(model_output)
        onnx.checker.check_model(quant_model)
        print(f"   ✅ Quantized model validation passed")

        # 文件大小比較
        fp32_size = os.path.getsize(model_input) / (1024**2)
        int8_size = os.path.getsize(model_output) / (1024**2)
        compression_ratio = fp32_size / int8_size

        print(f"\n   Model size comparison:")
        print(f"     FP32 model:  {fp32_size:.2f} MB")
        print(f"     INT8 model:  {int8_size:.2f} MB")
        print(f"     Compression: {compression_ratio:.2f}x")

    except ImportError:
        print(f"   ⚠️  Warning: onnx package not found, skipping validation")
    except Exception as e:
        print(f"   ⚠️  Warning: Validation failed: {e}")

    print("\n" + "="*70)
    print("✅ INT8 quantization completed successfully!")
    print("="*70)
    print(f"\nOutput: {model_output}")
    print(f"\nNext step: Verify quantization accuracy")
    print(f"  python verify_quantization.py")
    print("="*70)

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Quantize ONNX model to INT8')
    parser.add_argument('--input', type=str, default='models/rppg_fp32.onnx',
                        help='Input FP32 ONNX model')
    parser.add_argument('--output', type=str, default='models/rppg_int8_qdq.onnx',
                        help='Output INT8 ONNX model')
    parser.add_argument('--calibration', type=str, default='calibration_data.pt',
                        help='Calibration data path')
    parser.add_argument('--per_channel', action='store_true', default=True,
                        help='Use per-channel quantization (default: True)')

    args = parser.parse_args()

    success = quantize_onnx_model(
        model_input=args.input,
        model_output=args.output,
        calibration_data_path=args.calibration,
        per_channel=args.per_channel
    )

    sys.exit(0 if success else 1)
