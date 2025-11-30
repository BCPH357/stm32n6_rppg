"""
STM32N6 ONNX Graph Surgery - 修復違規項

使用 ONNX Graph Surgeon 修復模型以符合 STM32N6 限制

修復項目:
    1. 移除所有 6D 張量和 6D→4D Reshape
    2. 固定所有 dynamic batch 為 1
    3. 將 Squeeze 的 tensor axes 改為 attribute axes
    4. 移除動態 shape constants
    5. 優化圖結構

用法:
    python fix_onnx_for_stm32.py --input models/rppg_4d_fp32.onnx --output models/rppg_4d_fp32_fixed.onnx

依賴:
    pip install onnx onnx-graphsurgeon
"""

import onnx
import onnx_graphsurgeon as gs
import numpy as np
import argparse
from pathlib import Path


class ONNXSTM32Fixer:
    """STM32N6 ONNX 修復器"""

    def __init__(self, input_onnx):
        self.input_path = Path(input_onnx)
        print(f"Loading ONNX model: {self.input_path}")

        # 載入為 GraphSurgeon 圖
        self.onnx_model = onnx.load(str(self.input_path))
        self.graph = gs.import_onnx(self.onnx_model)

        self.fixes_applied = []

    def fix_all(self):
        """執行所有修復"""
        print("\n" + "="*70)
        print("STM32N6 ONNX Graph Surgery")
        print("="*70)

        self.fix_input_batch()
        self.fix_output_batch()
        self.remove_6d_reshape()
        self.fix_squeeze_nodes()
        self.fix_dynamic_constants()
        self.optimize_graph()

        print("\n" + "="*70)
        print(f"Total fixes applied: {len(self.fixes_applied)}")
        for fix in self.fixes_applied:
            print(f"  ✅ {fix}")
        print("="*70)

    def fix_input_batch(self):
        """固定輸入 batch = 1"""
        print("\n[Fix 1] Fixing input batch dimension...")

        for input_tensor in self.graph.inputs:
            if input_tensor.shape and input_tensor.shape[0] != 1:
                old_shape = input_tensor.shape
                input_tensor.shape[0] = 1
                print(f"  ✅ {input_tensor.name}: {old_shape} → {input_tensor.shape}")
                self.fixes_applied.append(f"Input batch fixed: {input_tensor.name}")
            else:
                print(f"  ✓ {input_tensor.name}: already batch=1")

    def fix_output_batch(self):
        """固定輸出 batch = 1"""
        print("\n[Fix 2] Fixing output batch dimension...")

        for output_tensor in self.graph.outputs:
            if output_tensor.shape and output_tensor.shape[0] != 1:
                old_shape = output_tensor.shape
                output_tensor.shape[0] = 1
                print(f"  ✅ {output_tensor.name}: {old_shape} → {output_tensor.shape}")
                self.fixes_applied.append(f"Output batch fixed: {output_tensor.name}")
            else:
                print(f"  ✓ {output_tensor.name}: already batch=1")

    def remove_6d_reshape(self):
        """移除 6D 張量和相關的 Reshape 節點"""
        print("\n[Fix 3] Removing 6D reshapes...")

        nodes_to_remove = []

        for node in self.graph.nodes:
            if node.op == "Reshape":
                # 獲取目標形狀
                if len(node.inputs) > 1 and isinstance(node.inputs[1], gs.Constant):
                    target_shape = node.inputs[1].values

                    # 檢查是否 reshape 到 6D
                    if len(target_shape) > 5:
                        print(f"  ❌ Found 6D Reshape: {node.name or node.outputs[0].name}")
                        print(f"       Target shape: {target_shape.tolist()}")

                        # 策略: 跳過這個 reshape，直接連接輸入到輸出
                        input_tensor = node.inputs[0]
                        output_tensor = node.outputs[0]

                        # 重新連接所有使用 output_tensor 的節點
                        for consumer in output_tensor.outputs:
                            # 找到使用這個輸出的輸入索引
                            for i, inp in enumerate(consumer.inputs):
                                if inp == output_tensor:
                                    consumer.inputs[i] = input_tensor

                        nodes_to_remove.append(node)
                        self.fixes_applied.append(f"Removed 6D reshape: {node.name or 'unnamed'}")

        # 移除標記的節點
        for node in nodes_to_remove:
            self.graph.nodes.remove(node)

        if nodes_to_remove:
            print(f"  ✅ Removed {len(nodes_to_remove)} 6D reshape nodes")
        else:
            print(f"  ✓ No 6D reshapes found")

    def fix_squeeze_nodes(self):
        """修復 Squeeze/Unsqueeze 節點 - 將 tensor axes 改為 attribute"""
        print("\n[Fix 4] Fixing Squeeze/Unsqueeze nodes...")

        for op_type in ['Squeeze', 'Unsqueeze']:
            nodes = [n for n in self.graph.nodes if n.op == op_type]

            if not nodes:
                print(f"  ✓ No {op_type} nodes found")
                continue

            print(f"  Processing {len(nodes)} {op_type} nodes...")

            for node in nodes:
                # Opset 13+: axes 可能是第二個輸入 (tensor)
                if len(node.inputs) > 1:
                    axes_input = node.inputs[1]

                    # 如果 axes 是 Constant，提取值並轉換為屬性
                    if isinstance(axes_input, gs.Constant):
                        axes_values = axes_input.values.tolist()

                        # 移除 axes 輸入
                        node.inputs = node.inputs[:1]

                        # 添加 axes 屬性
                        node.attrs["axes"] = axes_values

                        print(f"    ✅ {node.name or node.outputs[0].name}: axes={axes_values} (tensor→attribute)")
                        self.fixes_applied.append(f"{op_type} axes fixed: {node.name or 'unnamed'}")
                    else:
                        print(f"    ⚠️  {node.name or node.outputs[0].name}: axes is not constant (cannot fix)")

                # Opset 11-12: axes 已經是屬性
                elif "axes" in node.attrs:
                    print(f"    ✓ {node.name or node.outputs[0].name}: axes already attribute")

    def fix_dynamic_constants(self):
        """修復動態 shape 的 Constant 節點"""
        print("\n[Fix 5] Fixing dynamic shape constants...")

        fixed_count = 0

        for node in self.graph.nodes:
            if node.op == "Constant":
                # 檢查 value 屬性
                if "value" in node.attrs:
                    value = node.attrs["value"].values

                    # 如果是 INT64 陣列（shape 常用格式）
                    if value.dtype == np.int64:
                        # 檢查是否包含 -1 或 0 (動態維度)
                        if np.any(value == -1) or np.any(value == 0):
                            print(f"  ⚠️  Found dynamic constant: {value.tolist()}")

                            # 將 -1 替換為 1 (假設 batch=1)
                            # 注意: 這可能需要根據實際情況調整
                            value[value == -1] = 1
                            value[value == 0] = 1

                            node.attrs["value"] = gs.Constant(node.attrs["value"].name, value)
                            print(f"      → Fixed to: {value.tolist()}")

                            fixed_count += 1
                            self.fixes_applied.append(f"Dynamic constant fixed: {node.name or 'unnamed'}")

        if fixed_count > 0:
            print(f"  ✅ Fixed {fixed_count} dynamic constants")
        else:
            print(f"  ✓ No dynamic constants found")

    def optimize_graph(self):
        """優化圖結構"""
        print("\n[Fix 6] Optimizing graph...")

        # Cleanup: 移除孤立節點
        self.graph.cleanup()
        print(f"  ✅ Graph cleaned up")

        # Topological sort
        self.graph.toposort()
        print(f"  ✅ Topological sort applied")

        self.fixes_applied.append("Graph optimized and cleaned")

    def save(self, output_path):
        """保存修復後的模型"""
        output_path = Path(output_path)

        print(f"\n{'='*70}")
        print(f"Saving fixed model to: {output_path}")

        # 導出為 ONNX
        fixed_onnx = gs.export_onnx(self.graph)

        # 保存
        onnx.save(fixed_onnx, str(output_path))

        print(f"✅ Model saved successfully")
        print(f"Size: {output_path.stat().st_size / 1024:.2f} KB")
        print(f"{'='*70}")

        return output_path


def main():
    parser = argparse.ArgumentParser(description='STM32N6 ONNX Fixer')
    parser.add_argument('--input', type=str, required=True, help='Input ONNX model')
    parser.add_argument('--output', type=str, required=True, help='Output fixed ONNX model')
    args = parser.parse_args()

    # 創建修復器
    fixer = ONNXSTM32Fixer(args.input)

    # 執行所有修復
    fixer.fix_all()

    # 保存
    output_path = fixer.save(args.output)

    print(f"\n{'='*70}")
    print(f"NEXT STEPS:")
    print(f"{'='*70}")
    print(f"1. Validate fixed model:")
    print(f"   python diagnose_onnx_stm32.py --onnx {output_path}")
    print(f"")
    print(f"2. Test with STM32 Edge AI Developer Cloud:")
    print(f"   stedgeai analyze --model {output_path} --target stm32n6")
    print(f"")
    print(f"3. If validation passes, quantize to INT8:")
    print(f"   python quantize_4d_model_v2.py")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
