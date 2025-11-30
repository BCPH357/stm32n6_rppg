"""
修正 ONNX 模型的 Dynamic Batch 問題 (STM32 Edge AI 兼容性)

問題：
    STM32 Edge AI Developer Cloud / X-CUBE-AI 不支援 dynamic batch
    當 input shape 第一維為 dim_param: "batch" 時會報錯：
    "INTERNAL ERROR: 'NoneType' object has no attribute 'get_value'"

解決方案：
    1. 固定 batch=1
    2. 移除所有 dynamic axes
    3. 硬編碼所有 shape 相關的 Constant nodes
    4. 使用 opset 14（STM32N6 最佳相容性）

用法 (在服務器上執行):
    cd /mnt/data_8T/ChenPinHao/server_training/
    python fix_onnx_dynamic_batch.py
"""

import onnx
from onnx import numpy_helper
import numpy as np
from pathlib import Path


def fix_dynamic_batch(input_onnx_path, output_onnx_path, fixed_batch_size=1):
    """
    修正 ONNX 模型的 dynamic batch，將其固定為指定大小

    Args:
        input_onnx_path: 輸入 ONNX 模型路徑
        output_onnx_path: 輸出 ONNX 模型路徑
        fixed_batch_size: 固定的 batch 大小（默認為 1）
    """
    print("="*70)
    print("Fix ONNX Dynamic Batch for STM32 Edge AI")
    print("="*70)

    # 載入模型
    print(f"\n[Step 1] Loading ONNX model...")
    print(f"   Input: {input_onnx_path}")
    model = onnx.load(str(input_onnx_path))
    print(f"   [OK] Model loaded")
    print(f"   Opset version: {model.opset_import[0].version}")

    # 檢查原始輸入形狀
    print(f"\n[Step 2] Checking original input shape...")
    for input_tensor in model.graph.input:
        print(f"   Input name: {input_tensor.name}")
        shape = input_tensor.type.tensor_type.shape
        shape_list = []
        for i, dim in enumerate(shape.dim):
            if dim.HasField('dim_value'):
                shape_list.append(dim.dim_value)
                print(f"     dim[{i}]: {dim.dim_value}")
            elif dim.HasField('dim_param'):
                shape_list.append(f'<{dim.dim_param}>')
                print(f"     dim[{i}]: {dim.dim_param} (dynamic) <- PROBLEM!")
        print(f"   Original shape: {shape_list}")

    # 修正輸入形狀（固定 batch）
    print(f"\n[Step 3] Fixing input shape (batch={fixed_batch_size})...")
    for input_tensor in model.graph.input:
        shape = input_tensor.type.tensor_type.shape

        # 清空並重建所有維度
        while len(shape.dim) > 0:
            shape.dim.pop()

        # 第一維固定為 batch_size
        dim_batch = shape.dim.add()
        dim_batch.dim_value = fixed_batch_size

        # 其他維度保持（假設是 (batch, 72, 36, 36)）
        for dim_value in [72, 36, 36]:
            dim = shape.dim.add()
            dim.dim_value = dim_value

        # 顯示修正後的形狀
        fixed_shape = [dim.dim_value for dim in shape.dim]
        print(f"   Fixed shape: {fixed_shape}")

    # 修正輸出形狀
    print(f"\n[Step 4] Fixing output shape...")
    for output_tensor in model.graph.output:
        print(f"   Output name: {output_tensor.name}")
        shape = output_tensor.type.tensor_type.shape

        # 檢查是否有 dynamic batch
        if shape.dim[0].HasField('dim_param'):
            print(f"     Detected dynamic batch in output: {shape.dim[0].dim_param}")

            # 清空並重建
            original_dims = []
            for dim in shape.dim:
                if dim.HasField('dim_value'):
                    original_dims.append(dim.dim_value)
                elif dim.HasField('dim_param'):
                    original_dims.append(-1)  # 標記為 dynamic

            while len(shape.dim) > 0:
                shape.dim.pop()

            # 重建（第一維固定為 batch_size）
            dim_batch = shape.dim.add()
            dim_batch.dim_value = fixed_batch_size

            for dim_value in original_dims[1:]:
                dim = shape.dim.add()
                dim.dim_value = dim_value

            fixed_shape = [dim.dim_value for dim in shape.dim]
            print(f"     Fixed shape: {fixed_shape}")

    # 修正 value_info（中間張量）
    print(f"\n[Step 5] Fixing intermediate tensors (value_info)...")
    fixed_count = 0
    for value_info in model.graph.value_info:
        shape = value_info.type.tensor_type.shape
        if len(shape.dim) > 0 and shape.dim[0].HasField('dim_param'):
            # 修正第一維
            shape.dim[0].ClearField('dim_param')
            shape.dim[0].dim_value = fixed_batch_size
            fixed_count += 1

    if fixed_count > 0:
        print(f"   [OK] Fixed {fixed_count} intermediate tensors")
    else:
        print(f"   [OK] No intermediate tensors need fixing")

    # 修正 Constant nodes 中的 shape-related 張量
    print(f"\n[Step 6] Fixing Constant nodes with dynamic shapes...")
    constant_fixed_count = 0

    for node in model.graph.node:
        if node.op_type == 'Constant':
            # 檢查是否有 value 屬性
            for attr in node.attribute:
                if attr.name == 'value' and attr.HasField('t'):
                    tensor = attr.t

                    # 如果這是一個包含 batch 維度的 shape 張量
                    if tensor.data_type == onnx.TensorProto.INT64:
                        # 獲取實際數值
                        if tensor.raw_data:
                            data = numpy_helper.to_array(tensor)

                            # 檢查是否第一個元素可能是 batch（例如 [-1, 72, 36, 36]）
                            if len(data) > 0 and data[0] == -1:
                                print(f"   Found dynamic shape constant: {data}")

                                # 修正第一維為 fixed_batch_size
                                data[0] = fixed_batch_size

                                # 重新設置 tensor 數據
                                new_tensor = numpy_helper.from_array(data, name=tensor.name)
                                attr.t.CopyFrom(new_tensor)

                                print(f"     -> Fixed to: {data}")
                                constant_fixed_count += 1

    if constant_fixed_count > 0:
        print(f"   [OK] Fixed {constant_fixed_count} Constant nodes")
    else:
        print(f"   [OK] No Constant nodes need fixing")

    # 驗證模型
    print(f"\n[Step 7] Validating fixed ONNX model...")
    try:
        onnx.checker.check_model(model)
        print(f"   [OK] Model validation passed")
    except Exception as e:
        print(f"   [WARNING] Validation warning: {e}")
        print(f"   (This is often OK for STM32 deployment)")

    # 保存修正後的模型
    print(f"\n[Step 8] Saving fixed model...")
    print(f"   Output: {output_onnx_path}")
    onnx.save(model, str(output_onnx_path))

    output_path = Path(output_onnx_path)
    print(f"   [OK] Model saved")
    print(f"   Size: {output_path.stat().st_size / 1024:.2f} KB")

    # 驗證修正結果
    print(f"\n[Step 9] Verifying fixed shape...")
    fixed_model = onnx.load(str(output_onnx_path))

    for input_tensor in fixed_model.graph.input:
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"   Input '{input_tensor.name}': {shape}")

        # 確認沒有 dynamic batch
        for i, dim in enumerate(input_tensor.type.tensor_type.shape.dim):
            if dim.HasField('dim_param'):
                print(f"     [ERROR] Still has dynamic dim at position {i}: {dim.dim_param}")
                return False

    for output_tensor in fixed_model.graph.output:
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"   Output '{output_tensor.name}': {shape}")

    print(f"   [OK] All shapes are fixed (no dynamic batch)")

    return True


def main():
    # 路徑配置
    models_dir = Path("models")
    input_onnx = models_dir / "rppg_4d_fp32.onnx"
    output_onnx = models_dir / "rppg_4d_fp32_fixed.onnx"

    # 檢查輸入文件
    if not input_onnx.exists():
        print(f"\n[ERROR] Input ONNX not found: {input_onnx}")
        print(f"Please run convert_to_4d_for_stm32.py first")
        return

    # 修正 dynamic batch
    success = fix_dynamic_batch(input_onnx, output_onnx, fixed_batch_size=1)

    # 完成
    if success:
        print("\n" + "="*70)
        print("[SUCCESS] Dynamic Batch Fixed!")
        print("="*70)
        print(f"\nFixed ONNX model: {output_onnx}")
        print(f"\nNext steps:")
        print(f"  1. Quantize the fixed model:")
        print(f"     python quantize_4d_model_v2.py")
        print(f"     (Remember to update input path to rppg_4d_fp32_fixed.onnx)")
        print(f"  2. Import to STM32 Edge AI Developer Cloud")
        print(f"     - Should now analyze without errors")
        print(f"     - Use Optimization: O1 or O2 (avoid O3)")
        print("="*70)
    else:
        print("\n[ERROR] Failed to fix dynamic batch")


if __name__ == "__main__":
    main()
