"""
TFLite 導出和量化腳本 - Pattern A 拆分模型

從 PyTorch 拆分模型導出為 TFLite INT8 量化模型：
- Part 1: Spatial CNN (INT8 全量化)
- Part 2: Temporal Fusion (混合量化: INT8 權重 + FP32 激活)

Pipeline:
PyTorch (.pth) → ONNX → TensorFlow SavedModel → TFLite (INT8)

執行方法:
    python export_tflite_split.py

輸出:
    models/spatial_cnn_int8.tflite
    models/temporal_fusion_int8.tflite
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import tf2onnx
import tensorflow as tf
from pathlib import Path
import sys

# 導入模型定義
from model_split import SpatialCNN, TemporalFusion


def load_calibration_data(data_path, num_samples=100, stratified=True):
    """
    載入校準數據（用於 INT8 量化）

    Args:
        data_path: ubfc_processed.pt 路徑
        num_samples: 校準樣本數
        stratified: 是否使用分層採樣

    Returns:
        spatial_calib_data: list of (1, 3, 36, 36) arrays for Spatial CNN
        temporal_calib_data: list of (1, 24, 16) arrays for Temporal Fusion
    """
    print(f"\n[載入校準數據]")
    print(f"  數據路徑: {data_path}")

    # 載入完整數據集
    data = torch.load(data_path)
    samples = data['samples']  # (N, 8, 3, 36, 36, 3)
    labels = data['labels']    # (N,)

    print(f"  總樣本數: {len(samples)}")
    print(f"  HR 範圍: [{labels.min():.1f}, {labels.max():.1f}] BPM")

    # 分層採樣（確保各 HR 範圍都有代表）
    if stratified:
        # 定義 HR 區間
        bins = [40, 60, 80, 100, 120, 160]
        samples_per_bin = num_samples // (len(bins) - 1)

        indices = []
        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i+1]
            mask = (labels >= low) & (labels < high)
            bin_indices = torch.where(mask)[0]

            if len(bin_indices) > 0:
                # 隨機選擇
                selected = bin_indices[torch.randperm(len(bin_indices))[:samples_per_bin]]
                indices.append(selected)
                print(f"  HR [{low:3.0f}, {high:3.0f}): {len(selected)} 樣本")

        indices = torch.cat(indices)
    else:
        # 隨機採樣
        indices = torch.randperm(len(samples))[:num_samples]

    calib_samples = samples[indices]  # (num_samples, 8, 3, 36, 36, 3)

    print(f"  校準樣本數: {len(calib_samples)}")

    # 為 Spatial CNN 準備數據
    # 輸入: (1, 3, 36, 36) NCHW
    spatial_calib_data = []

    for sample in calib_samples:
        # sample: (8, 3, 36, 36, 3)
        for t in range(8):
            for roi in range(3):
                patch = sample[t, roi, :, :, :]  # (36, 36, 3) HWC

                # 轉換為 NCHW
                patch = torch.from_numpy(patch.numpy()) if isinstance(patch, torch.Tensor) else patch
                patch = patch.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 36, 36)

                spatial_calib_data.append(patch.numpy().astype(np.float32))

    print(f"  Spatial CNN 校準數據: {len(spatial_calib_data)} 個 ROI patches")

    # 為 Temporal Fusion 準備數據
    # 需要先通過 Spatial CNN 提取特徵
    print(f"\n[提取時序特徵用於 Temporal Fusion 校準]")

    # 載入 Spatial CNN
    spatial_cnn = SpatialCNN()
    spatial_checkpoint = torch.load("checkpoints/spatial_cnn.pth", map_location='cpu')
    spatial_cnn.load_state_dict(spatial_checkpoint['model_state_dict'])
    spatial_cnn.eval()

    temporal_calib_data = []

    with torch.no_grad():
        for sample in calib_samples:
            # sample: (8, 3, 36, 36, 3)
            features_list = []

            for t in range(8):
                for roi in range(3):
                    patch = sample[t, roi, :, :, :]  # (36, 36, 3)
                    patch = torch.from_numpy(patch.numpy()) if isinstance(patch, torch.Tensor) else patch
                    patch = patch.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 36, 36)

                    # 提取特徵
                    feat = spatial_cnn(patch)  # (1, 16)
                    features_list.append(feat)

            # 堆疊為 (1, 24, 16)
            features = torch.cat(features_list, dim=0)  # (24, 16)
            features = features.unsqueeze(0)  # (1, 24, 16)

            temporal_calib_data.append(features.numpy().astype(np.float32))

    print(f"  Temporal Fusion 校準數據: {len(temporal_calib_data)} 個時序窗口")

    return spatial_calib_data, temporal_calib_data


def export_spatial_cnn_to_tflite(checkpoint_path, output_path, calib_data):
    """
    導出 Spatial CNN 為 TFLite INT8 量化模型

    Args:
        checkpoint_path: spatial_cnn.pth 路徑
        output_path: 輸出 .tflite 路徑
        calib_data: 校準數據 list of (1, 3, 36, 36)
    """
    print(f"\n{'='*70}")
    print(f"導出 Spatial CNN → TFLite INT8")
    print(f"{'='*70}")

    # Step 1: 載入 PyTorch 模型
    print(f"\n[Step 1] 載入 PyTorch 模型")
    spatial_cnn = SpatialCNN()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    spatial_cnn.load_state_dict(checkpoint['model_state_dict'])
    spatial_cnn.eval()
    print(f"  參數量: {spatial_cnn.get_num_params():,}")

    # Step 2: 導出為 ONNX
    print(f"\n[Step 2] 導出為 ONNX")
    dummy_input = torch.randn(1, 3, 36, 36)
    onnx_path = "models/spatial_cnn.onnx"

    torch.onnx.export(
        spatial_cnn,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch'},
            'output': {0: 'batch'}
        }
    )
    print(f"  ONNX 模型已保存: {onnx_path}")

    # 驗證 ONNX
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX 驗證通過")

    # Step 3: ONNX → TensorFlow SavedModel
    print(f"\n[Step 3] ONNX → TensorFlow SavedModel")
    saved_model_path = "models/spatial_cnn_saved_model"

    # 使用 onnx-tf 轉換
    import onnx_tf
    tf_rep = onnx_tf.backend.prepare(onnx_model)
    tf_rep.export_graph(saved_model_path)
    print(f"  SavedModel 已保存: {saved_model_path}")

    # Step 4: TensorFlow SavedModel → TFLite INT8
    print(f"\n[Step 4] TensorFlow SavedModel → TFLite INT8")

    # 載入 SavedModel
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

    # INT8 量化配置
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # 代表性數據集生成器
    def representative_dataset():
        for data in calib_data[:100]:  # 使用前 100 個樣本
            yield [data.astype(np.float32)]

    converter.representative_dataset = representative_dataset

    # 轉換
    tflite_model = converter.convert()

    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"  TFLite INT8 模型已保存: {output_path}")
    print(f"  模型大小: {len(tflite_model) / 1024:.2f} KB")

    # Step 5: 驗證 TFLite 模型
    print(f"\n[Step 5] 驗證 TFLite 模型")
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  輸入: {input_details[0]['name']}")
    print(f"    Shape: {input_details[0]['shape']}")
    print(f"    Type: {input_details[0]['dtype']}")

    print(f"  輸出: {output_details[0]['name']}")
    print(f"    Shape: {output_details[0]['shape']}")
    print(f"    Type: {output_details[0]['dtype']}")

    # 測試推論
    test_input = calib_data[0]

    # 量化輸入
    input_scale, input_zero_point = input_details[0]['quantization']
    test_input_quant = (test_input / input_scale + input_zero_point).astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], test_input_quant)
    interpreter.invoke()

    output_quant = interpreter.get_tensor(output_details[0]['index'])

    # 反量化輸出
    output_scale, output_zero_point = output_details[0]['quantization']
    output = (output_quant.astype(np.float32) - output_zero_point) * output_scale

    print(f"  測試推論成功")
    print(f"  輸出形狀: {output.shape}")
    print(f"  輸出範圍: [{output.min():.4f}, {output.max():.4f}]")

    print(f"\n[SUCCESS] Spatial CNN 導出完成")


def export_temporal_fusion_to_tflite(checkpoint_path, output_path, calib_data):
    """
    導出 Temporal Fusion 為 TFLite 混合量化模型

    Args:
        checkpoint_path: temporal_fusion.pth 路徑
        output_path: 輸出 .tflite 路徑
        calib_data: 校準數據 list of (1, 24, 16)
    """
    print(f"\n{'='*70}")
    print(f"導出 Temporal Fusion → TFLite 混合量化")
    print(f"{'='*70}")

    # Step 1: 載入 PyTorch 模型
    print(f"\n[Step 1] 載入 PyTorch 模型")
    temporal_fusion = TemporalFusion(window_size=8, num_rois=3)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    temporal_fusion.load_state_dict(checkpoint['model_state_dict'])
    temporal_fusion.eval()
    print(f"  參數量: {temporal_fusion.get_num_params():,}")

    # Step 2: 導出為 ONNX
    print(f"\n[Step 2] 導出為 ONNX")
    dummy_input = torch.randn(1, 24, 16)
    onnx_path = "models/temporal_fusion.onnx"

    torch.onnx.export(
        temporal_fusion,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch'},
            'output': {0: 'batch'}
        }
    )
    print(f"  ONNX 模型已保存: {onnx_path}")

    # 驗證 ONNX
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX 驗證通過")

    # Step 3: ONNX → TensorFlow SavedModel
    print(f"\n[Step 3] ONNX → TensorFlow SavedModel")
    saved_model_path = "models/temporal_fusion_saved_model"

    import onnx_tf
    tf_rep = onnx_tf.backend.prepare(onnx_model)
    tf_rep.export_graph(saved_model_path)
    print(f"  SavedModel 已保存: {saved_model_path}")

    # Step 4: TensorFlow SavedModel → TFLite 混合量化
    print(f"\n[Step 4] TensorFlow SavedModel → TFLite 混合量化")

    # 載入 SavedModel
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

    # 混合量化配置（INT8 權重 + FP32 激活）
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 代表性數據集生成器
    def representative_dataset():
        for data in calib_data[:100]:
            yield [data.astype(np.float32)]

    converter.representative_dataset = representative_dataset

    # 轉換
    tflite_model = converter.convert()

    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"  TFLite 混合量化模型已保存: {output_path}")
    print(f"  模型大小: {len(tflite_model) / 1024:.2f} KB")

    # Step 5: 驗證 TFLite 模型
    print(f"\n[Step 5] 驗證 TFLite 模型")
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  輸入: {input_details[0]['name']}")
    print(f"    Shape: {input_details[0]['shape']}")
    print(f"    Type: {input_details[0]['dtype']}")

    print(f"  輸出: {output_details[0]['name']}")
    print(f"    Shape: {output_details[0]['shape']}")
    print(f"    Type: {output_details[0]['dtype']}")

    # 測試推論
    test_input = calib_data[0].astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    print(f"  測試推論成功")
    print(f"  輸出形狀: {output.shape}")
    print(f"  HR 預測: {output[0, 0]:.2f} BPM")

    if 30 <= output[0, 0] <= 180:
        print(f"  [OK] HR 在合理範圍內 [30, 180] BPM")
    else:
        print(f"  [WARNING] HR 超出範圍")

    print(f"\n[SUCCESS] Temporal Fusion 導出完成")


def main():
    print(f"{'='*70}")
    print(f"TFLite 導出和量化 - Pattern A 拆分模型")
    print(f"{'='*70}")

    # 檢查依賴
    print(f"\n[檢查依賴]")
    try:
        import onnx_tf
        print(f"  [OK] onnx-tf: {onnx_tf.__version__}")
    except ImportError:
        print(f"  [ERROR] 缺少 onnx-tf，請執行: pip install onnx-tf")
        sys.exit(1)

    print(f"  [OK] tensorflow: {tf.__version__}")
    print(f"  [OK] onnx: {onnx.__version__}")

    # 確保模型目錄存在
    Path("models").mkdir(exist_ok=True)

    # 載入校準數據
    spatial_calib, temporal_calib = load_calibration_data(
        "data/ubfc_processed.pt",
        num_samples=100,
        stratified=True
    )

    # 導出 Spatial CNN
    export_spatial_cnn_to_tflite(
        checkpoint_path="checkpoints/spatial_cnn.pth",
        output_path="models/spatial_cnn_int8.tflite",
        calib_data=spatial_calib
    )

    # 導出 Temporal Fusion
    export_temporal_fusion_to_tflite(
        checkpoint_path="checkpoints/temporal_fusion.pth",
        output_path="models/temporal_fusion_int8.tflite",
        calib_data=temporal_calib
    )

    # 總結
    print(f"\n{'='*70}")
    print(f"[SUCCESS] 所有模型導出完成")
    print(f"{'='*70}")
    print(f"\n輸出文件:")
    print(f"  - models/spatial_cnn_int8.tflite")
    print(f"  - models/temporal_fusion_int8.tflite")
    print(f"\n下一步:")
    print(f"  1. 運行 validate_tflite.py 驗證量化精度")
    print(f"  2. 下載 .tflite 文件到本地")
    print(f"  3. 導入 STM32CubeMX")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
