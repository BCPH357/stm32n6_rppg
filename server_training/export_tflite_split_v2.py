"""
TFLite 導出和量化腳本 v2 - Pattern A 拆分模型

修改版：直接使用 TensorFlow/Keras 重建模型，避免 ONNX 轉換問題

Pipeline:
PyTorch (.pth) → TensorFlow/Keras → TFLite (INT8)

執行方法:
    python export_tflite_split_v2.py

輸出:
    models/spatial_cnn_int8.tflite
    models/temporal_fusion_int8.tflite
"""

import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
from tensorflow import keras
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
        spatial_calib_data: list of (1, 36, 36, 3) arrays for Spatial CNN (NHWC)
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
    # 輸入: (1, 36, 36, 3) NHWC（TFLite 格式）
    spatial_calib_data = []

    for sample in calib_samples:
        # sample: (8, 3, 36, 36, 3)
        for t in range(8):
            for roi in range(3):
                patch = sample[t, roi, :, :, :]  # (36, 36, 3) HWC

                # 轉換為 numpy
                if isinstance(patch, torch.Tensor):
                    patch = patch.numpy()

                # 添加 batch 維度 (1, 36, 36, 3)
                patch = np.expand_dims(patch, axis=0).astype(np.float32)

                spatial_calib_data.append(patch)

    print(f"  Spatial CNN 校準數據: {len(spatial_calib_data)} 個 ROI patches")

    # 為 Temporal Fusion 準備數據
    # 需要先通過 Spatial CNN 提取特徵
    print(f"\n[提取時序特徵用於 Temporal Fusion 校準]")

    # 載入 Spatial CNN
    spatial_cnn_pt = SpatialCNN()
    spatial_checkpoint = torch.load("checkpoints/spatial_cnn.pth", map_location='cpu')
    spatial_cnn_pt.load_state_dict(spatial_checkpoint['model_state_dict'])
    spatial_cnn_pt.eval()

    temporal_calib_data = []

    with torch.no_grad():
        for sample in calib_samples:
            # sample: (8, 3, 36, 36, 3)
            features_list = []

            for t in range(8):
                for roi in range(3):
                    patch = sample[t, roi, :, :, :]  # (36, 36, 3)

                    # 轉換為 PyTorch NCHW
                    patch_pt = torch.from_numpy(patch.numpy() if isinstance(patch, torch.Tensor) else patch)
                    patch_pt = patch_pt.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 36, 36)

                    # 提取特徵
                    feat = spatial_cnn_pt(patch_pt)  # (1, 16)
                    features_list.append(feat)

            # 堆疊為 (1, 24, 16)
            features = torch.cat(features_list, dim=0)  # (24, 16)
            features = features.unsqueeze(0).numpy().astype(np.float32)  # (1, 24, 16)

            temporal_calib_data.append(features)

    print(f"  Temporal Fusion 校準數據: {len(temporal_calib_data)} 個時序窗口")

    return spatial_calib_data, temporal_calib_data


def build_spatial_cnn_keras(pytorch_model):
    """
    用 Keras 重建 Spatial CNN 並複製 PyTorch 權重

    Args:
        pytorch_model: PyTorch SpatialCNN 模型

    Returns:
        keras_model: Keras 模型
    """
    print(f"\n[使用 Keras 重建 Spatial CNN]")

    # 定義 Keras 模型（NHWC 格式）
    inputs = keras.Input(shape=(36, 36, 3), name='input')

    # Layer 1: Conv2D(3→16) + BN + ReLU + MaxPool2D
    x = keras.layers.Conv2D(16, (3, 3), padding='same', use_bias=False, name='conv1')(inputs)
    x = keras.layers.BatchNormalization(name='bn1')(x)
    x = keras.layers.ReLU(name='relu1')(x)
    x = keras.layers.MaxPooling2D((2, 2), name='pool1')(x)

    # Layer 2: Conv2D(16→32) + BN + ReLU + MaxPool2D
    x = keras.layers.Conv2D(32, (3, 3), padding='same', use_bias=False, name='conv2')(x)
    x = keras.layers.BatchNormalization(name='bn2')(x)
    x = keras.layers.ReLU(name='relu2')(x)
    x = keras.layers.MaxPooling2D((2, 2), name='pool2')(x)

    # Layer 3: Conv2D(32→16) + BN + ReLU
    x = keras.layers.Conv2D(16, (3, 3), padding='same', use_bias=False, name='conv3')(x)
    x = keras.layers.BatchNormalization(name='bn3')(x)
    x = keras.layers.ReLU(name='relu3')(x)

    # Global Average Pooling
    x = keras.layers.GlobalAveragePooling2D(name='gap')(x)

    # 輸出 (B, 16)
    outputs = x

    model = keras.Model(inputs=inputs, outputs=outputs, name='spatial_cnn')

    print(f"  Keras 模型已創建")
    print(f"  參數量: {model.count_params():,}")

    # 複製 PyTorch 權重到 Keras
    print(f"\n[複製 PyTorch 權重到 Keras]")

    pt_state = pytorch_model.state_dict()

    # Conv1
    conv1_weight = pt_state['features.0.weight'].numpy()  # (16, 3, 3, 3) NCHW
    conv1_weight = np.transpose(conv1_weight, (2, 3, 1, 0))  # → (3, 3, 3, 16) HWIO
    model.get_layer('conv1').set_weights([conv1_weight])

    # BN1
    bn1_gamma = pt_state['features.1.weight'].numpy()
    bn1_beta = pt_state['features.1.bias'].numpy()
    bn1_mean = pt_state['features.1.running_mean'].numpy()
    bn1_var = pt_state['features.1.running_var'].numpy()
    model.get_layer('bn1').set_weights([bn1_gamma, bn1_beta, bn1_mean, bn1_var])

    # Conv2
    conv2_weight = pt_state['features.4.weight'].numpy()  # (32, 16, 3, 3)
    conv2_weight = np.transpose(conv2_weight, (2, 3, 1, 0))  # → (3, 3, 16, 32)
    model.get_layer('conv2').set_weights([conv2_weight])

    # BN2
    bn2_gamma = pt_state['features.5.weight'].numpy()
    bn2_beta = pt_state['features.5.bias'].numpy()
    bn2_mean = pt_state['features.5.running_mean'].numpy()
    bn2_var = pt_state['features.5.running_var'].numpy()
    model.get_layer('bn2').set_weights([bn2_gamma, bn2_beta, bn2_mean, bn2_var])

    # Conv3
    conv3_weight = pt_state['features.8.weight'].numpy()  # (16, 32, 3, 3)
    conv3_weight = np.transpose(conv3_weight, (2, 3, 1, 0))  # → (3, 3, 32, 16)
    model.get_layer('conv3').set_weights([conv3_weight])

    # BN3
    bn3_gamma = pt_state['features.9.weight'].numpy()
    bn3_beta = pt_state['features.9.bias'].numpy()
    bn3_mean = pt_state['features.9.running_mean'].numpy()
    bn3_var = pt_state['features.9.running_var'].numpy()
    model.get_layer('bn3').set_weights([bn3_gamma, bn3_beta, bn3_mean, bn3_var])

    print(f"  [OK] 權重複製完成")

    return model


def build_temporal_fusion_keras(pytorch_model):
    """
    用 Keras 重建 Temporal Fusion 並複製 PyTorch 權重

    Args:
        pytorch_model: PyTorch TemporalFusion 模型

    Returns:
        keras_model: Keras 模型
    """
    print(f"\n[使用 Keras 重建 Temporal Fusion]")

    # 定義 Keras 模型
    inputs = keras.Input(shape=(24, 16), name='input')

    # 使用 Lambda 層模擬完整的 PyTorch 處理流程
    # (B, 24, 16) → (B, 8, 3, 16) → (B, 8, 48) → (B, 48, 8)
    def reshape_and_transpose(x):
        # x shape: (B, 24, 16)
        B = tf.shape(x)[0]
        x = tf.reshape(x, [B, 8, 3, 16])  # (B, 8, 3, 16)
        x = tf.reshape(x, [B, 8, 48])     # (B, 8, 48)
        x = tf.transpose(x, [0, 2, 1])    # (B, 48, 8) - 注意 TF 的 transpose
        return x

    x = keras.layers.Lambda(reshape_and_transpose)(inputs)

    # Temporal Conv1D: input (B, 48, 8)
    # 注意：Keras Conv1D 期望 (batch, steps, channels)，但我們有 (B, 48, 8)
    # 所以這裡 48 是 steps，8 是 channels
    # 但 PyTorch 期望 (B, 48, 8) 其中 48 是 in_channels，8 是 sequence_length
    # 需要再次轉置
    x = keras.layers.Lambda(lambda x: tf.transpose(x, [0, 2, 1]))(x)  # (B, 8, 48)

    # 現在 Conv1D 輸入是 (B, 8, 48) - 8 個時間步，每步 48 個特徵
    x = keras.layers.Conv1D(32, 3, padding='same', activation='relu', name='conv1d_1')(x)  # (B, 8, 32)
    x = keras.layers.Conv1D(16, 3, padding='same', activation='relu', name='conv1d_2')(x)  # (B, 8, 16)

    # Flatten
    x = keras.layers.Flatten()(x)  # (128)

    # FC layers
    x = keras.layers.Dense(32, activation='relu', name='fc1')(x)
    x = keras.layers.Dense(1, activation='sigmoid', name='fc2')(x)

    # Scale to [30, 180] BPM
    outputs = keras.layers.Lambda(lambda x: x * 150 + 30, name='hr_output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='temporal_fusion')

    print(f"  Keras 模型已創建")
    print(f"  參數量: {model.count_params():,}")

    # 複製 PyTorch 權重到 Keras
    print(f"\n[複製 PyTorch 權重到 Keras]")

    pt_state = pytorch_model.state_dict()

    # Conv1D layers (PyTorch: (out_ch, in_ch, kernel), Keras: (kernel, in_ch, out_ch))
    conv1d_1_weight = pt_state['temporal.0.weight'].numpy()  # (32, 48, 3)
    conv1d_1_weight = np.transpose(conv1d_1_weight, (2, 1, 0))  # → (3, 48, 32)
    conv1d_1_bias = pt_state['temporal.0.bias'].numpy()
    model.get_layer('conv1d_1').set_weights([conv1d_1_weight, conv1d_1_bias])

    conv1d_2_weight = pt_state['temporal.2.weight'].numpy()  # (16, 32, 3)
    conv1d_2_weight = np.transpose(conv1d_2_weight, (2, 1, 0))  # → (3, 32, 16)
    conv1d_2_bias = pt_state['temporal.2.bias'].numpy()
    model.get_layer('conv1d_2').set_weights([conv1d_2_weight, conv1d_2_bias])

    # FC layers
    fc1_weight = pt_state['fc.0.weight'].numpy().T  # (128, 32) → (32, 128)^T
    fc1_bias = pt_state['fc.0.bias'].numpy()
    model.get_layer('fc1').set_weights([fc1_weight, fc1_bias])

    fc2_weight = pt_state['fc.2.weight'].numpy().T  # (32, 1) → (1, 32)^T
    fc2_bias = pt_state['fc.2.bias'].numpy()
    model.get_layer('fc2').set_weights([fc2_weight, fc2_bias])

    print(f"  [OK] 權重複製完成")

    return model


def export_spatial_cnn_to_tflite(keras_model, output_path, calib_data):
    """
    導出 Spatial CNN 為 TFLite INT8 量化模型

    Args:
        keras_model: Keras 模型
        output_path: 輸出 .tflite 路徑
        calib_data: 校準數據 list of (1, 36, 36, 3)
    """
    print(f"\n{'='*70}")
    print(f"導出 Spatial CNN → TFLite INT8")
    print(f"{'='*70}")

    # TFLite 轉換器
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

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
    print(f"\n[轉換為 TFLite INT8]")
    tflite_model = converter.convert()

    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"  [OK] TFLite INT8 模型已保存: {output_path}")
    print(f"  模型大小: {len(tflite_model) / 1024:.2f} KB")

    # 驗證
    print(f"\n[驗證 TFLite 模型]")
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


def export_temporal_fusion_to_tflite(keras_model, output_path, calib_data):
    """
    導出 Temporal Fusion 為 TFLite 混合量化模型

    Args:
        keras_model: Keras 模型
        output_path: 輸出 .tflite 路徑
        calib_data: 校準數據 list of (1, 24, 16)
    """
    print(f"\n{'='*70}")
    print(f"導出 Temporal Fusion → TFLite 混合量化")
    print(f"{'='*70}")

    # TFLite 轉換器
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    # 混合量化配置（INT8 權重 + FP32 激活）
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 代表性數據集生成器
    def representative_dataset():
        for data in calib_data[:100]:
            yield [data.astype(np.float32)]

    converter.representative_dataset = representative_dataset

    # 轉換
    print(f"\n[轉換為 TFLite 混合量化]")
    tflite_model = converter.convert()

    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"  [OK] TFLite 混合量化模型已保存: {output_path}")
    print(f"  模型大小: {len(tflite_model) / 1024:.2f} KB")

    # 驗證
    print(f"\n[驗證 TFLite 模型]")
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
    print(f"TFLite 導出和量化 v2 - Pattern A 拆分模型")
    print(f"{'='*70}")

    # 檢查 TensorFlow 版本
    print(f"\n[檢查依賴]")
    print(f"  [OK] TensorFlow: {tf.__version__}")

    # 確保模型目錄存在
    Path("models").mkdir(exist_ok=True)

    # 載入 PyTorch 模型
    print(f"\n[載入 PyTorch 模型]")
    spatial_cnn_pt = SpatialCNN()
    temporal_fusion_pt = TemporalFusion(window_size=8, num_rois=3)

    spatial_checkpoint = torch.load("checkpoints/spatial_cnn.pth", map_location='cpu')
    temporal_checkpoint = torch.load("checkpoints/temporal_fusion.pth", map_location='cpu')

    spatial_cnn_pt.load_state_dict(spatial_checkpoint['model_state_dict'])
    temporal_fusion_pt.load_state_dict(temporal_checkpoint['model_state_dict'])

    spatial_cnn_pt.eval()
    temporal_fusion_pt.eval()

    print(f"  [OK] PyTorch 模型已載入")

    # 載入校準數據
    spatial_calib, temporal_calib = load_calibration_data(
        "data/ubfc_processed.pt",
        num_samples=100,
        stratified=True
    )

    # 重建 Keras 模型
    spatial_keras = build_spatial_cnn_keras(spatial_cnn_pt)
    temporal_keras = build_temporal_fusion_keras(temporal_fusion_pt)

    # 導出 Spatial CNN
    export_spatial_cnn_to_tflite(
        keras_model=spatial_keras,
        output_path="models/spatial_cnn_int8.tflite",
        calib_data=spatial_calib
    )

    # 導出 Temporal Fusion
    export_temporal_fusion_to_tflite(
        keras_model=temporal_keras,
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
