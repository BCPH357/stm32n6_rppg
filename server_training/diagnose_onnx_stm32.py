"""
STM32N6 ONNX 診斷工具
檢測 ONNX 模型是否違反 STM32 Edge AI Core v2.2.0 的限制

用法:
    python diagnose_onnx_stm32.py --onnx models/rppg_4d_fp32.onnx

檢查項目:
    1. 張量維度 (max rank = 5, no 6D)
    2. Batch 固定 (batch = 1, no dynamic)
    3. Dynamic shapes
    4. Squeeze 操作 (static axes only)
    5. Reshape 節點 (no 6D→4D)
    6. Opset version (recommend 14)
"""

import onnx
from onnx import numpy_helper
import argparse
from pathlib import Path
from collections import defaultdict


class STM32N6Diagnostics:
    """STM32N6 ONNX 兼容性診斷"""

    def __init__(self, onnx_path):
        self.onnx_path = Path(onnx_path)
        self.model = onnx.load(str(self.onnx_path))
        self.graph = self.model.graph
        self.violations = []
        self.warnings = []

    def diagnose_all(self):
        """執行所有診斷檢查"""
        print("="*70)
        print(f"STM32N6 ONNX 診斷報告")
        print("="*70)
        print(f"模型: {self.onnx_path.name}")
        print(f"Opset: {self.model.opset_import[0].version}")
        print("="*70)

        # 執行所有檢查
        self.check_opset()
        self.check_input_output_shapes()
        self.check_tensor_ranks()
        self.check_dynamic_shapes()
        self.check_reshape_nodes()
        self.check_squeeze_nodes()
        self.check_batch_dimensions()
        self.check_unsupported_ops()

        # 生成報告
        self.print_report()

        return len(self.violations) == 0

    def check_opset(self):
        """檢查 Opset 版本"""
        opset = self.model.opset_import[0].version

        if opset == 14:
            print(f"\n✅ Opset Version: {opset} (RECOMMENDED)")
        elif opset <= 14:
            print(f"\n⚠️  Opset Version: {opset} (OK, but 14 is better)")
            self.warnings.append(f"Opset {opset} is old, recommend upgrading to 14")
        else:
            print(f"\n❌ Opset Version: {opset} (TOO NEW, use 14)")
            self.violations.append(f"Opset {opset} > 14 may not be supported")

    def check_input_output_shapes(self):
        """檢查輸入輸出形狀"""
        print(f"\n{'─'*70}")
        print("📥 Input/Output Shape Analysis")
        print(f"{'─'*70}")

        # 檢查輸入
        for input_tensor in self.graph.input:
            shape = self._get_shape(input_tensor.type.tensor_type.shape)
            rank = len([s for s in shape if s != '?'])

            print(f"\nInput '{input_tensor.name}':")
            print(f"  Shape: {shape}")
            print(f"  Rank: {rank}")

            # Check 1: Rank <= 5
            if rank > 5:
                self.violations.append(
                    f"Input '{input_tensor.name}' has rank {rank} > 5 (6D not supported)"
                )
                print(f"  ❌ VIOLATION: Rank > 5")
            elif rank > 4:
                self.warnings.append(
                    f"Input '{input_tensor.name}' has rank {rank} (5D is max, 4D is safer)"
                )
                print(f"  ⚠️  WARNING: Rank = 5 (4D is safer)")
            else:
                print(f"  ✅ Rank OK")

            # Check 2: Dynamic batch
            has_dynamic = self._check_dynamic_dims(input_tensor.type.tensor_type.shape)
            if has_dynamic:
                self.violations.append(
                    f"Input '{input_tensor.name}' has dynamic dimensions"
                )
                print(f"  ❌ VIOLATION: Dynamic dimensions detected")
            else:
                print(f"  ✅ All dimensions are fixed")

            # Check 3: Batch = 1
            if shape and shape[0] != 1 and shape[0] != '?':
                self.violations.append(
                    f"Input '{input_tensor.name}' has batch={shape[0]} (must be 1)"
                )
                print(f"  ❌ VIOLATION: Batch = {shape[0]} (must be 1)")
            elif shape and shape[0] == 1:
                print(f"  ✅ Batch = 1")

        # 檢查輸出
        for output_tensor in self.graph.output:
            shape = self._get_shape(output_tensor.type.tensor_type.shape)
            print(f"\nOutput '{output_tensor.name}': {shape}")

            has_dynamic = self._check_dynamic_dims(output_tensor.type.tensor_type.shape)
            if has_dynamic:
                self.violations.append(
                    f"Output '{output_tensor.name}' has dynamic dimensions"
                )
                print(f"  ❌ VIOLATION: Dynamic dimensions")

    def check_tensor_ranks(self):
        """檢查所有中間張量的 rank"""
        print(f"\n{'─'*70}")
        print("🔍 Intermediate Tensor Rank Analysis")
        print(f"{'─'*70}")

        rank_stats = defaultdict(list)

        # 檢查所有 value_info（中間張量）
        for value_info in self.graph.value_info:
            if value_info.type.tensor_type.HasField('shape'):
                shape = self._get_shape(value_info.type.tensor_type.shape)
                rank = len([s for s in shape if s != '?'])
                rank_stats[rank].append((value_info.name, shape))

                if rank > 5:
                    self.violations.append(
                        f"Tensor '{value_info.name}' has rank {rank} > 5: {shape}"
                    )

        # 顯示統計
        print(f"\nTensor Rank Distribution:")
        for rank in sorted(rank_stats.keys()):
            tensors = rank_stats[rank]
            status = "❌" if rank > 5 else "⚠️" if rank == 5 else "✅"
            print(f"  {status} Rank {rank}: {len(tensors)} tensors")

            # 顯示 6D 張量詳情
            if rank > 5:
                print(f"     6D Tensors (VIOLATIONS):")
                for name, shape in tensors[:5]:  # 最多顯示 5 個
                    print(f"       - {name}: {shape}")
                if len(tensors) > 5:
                    print(f"       - ... and {len(tensors) - 5} more")

    def check_dynamic_shapes(self):
        """檢查所有動態形狀"""
        print(f"\n{'─'*70}")
        print("🔀 Dynamic Shape Analysis")
        print(f"{'─'*70}")

        dynamic_count = 0

        # 檢查所有節點的輸出
        for node in self.graph.node:
            # 檢查 Constant 節點（可能包含 shape 參數）
            if node.op_type == 'Constant':
                for attr in node.attribute:
                    if attr.name == 'value' and attr.HasField('t'):
                        tensor = attr.t
                        if tensor.data_type == onnx.TensorProto.INT64:
                            if tensor.raw_data:
                                data = numpy_helper.to_array(tensor)
                                if any(data == -1) or any(data == 0):
                                    dynamic_count += 1
                                    self.violations.append(
                                        f"Constant node '{node.output[0]}' contains dynamic shape: {data.tolist()}"
                                    )
                                    print(f"  ❌ {node.output[0]}: {data.tolist()}")

        if dynamic_count == 0:
            print(f"  ✅ No dynamic shape constants detected")
        else:
            print(f"  ❌ Found {dynamic_count} dynamic shape constants")

    def check_reshape_nodes(self):
        """檢查 Reshape 節點（尋找 6D→4D）"""
        print(f"\n{'─'*70}")
        print("🔄 Reshape Node Analysis")
        print(f"{'─'*70}")

        reshape_count = 0

        for node in self.graph.node:
            if node.op_type == 'Reshape':
                reshape_count += 1

                # 嘗試找到 shape 參數
                shape_input = node.input[1] if len(node.input) > 1 else None

                if shape_input:
                    # 查找對應的 Constant 節點
                    shape_value = self._find_constant_value(shape_input)
                    if shape_value is not None:
                        print(f"\n  Reshape '{node.output[0]}':")
                        print(f"    Shape input: {shape_input}")
                        print(f"    Target shape: {shape_value.tolist()}")

                        # 檢查是否包含 6D
                        if len(shape_value) > 5:
                            self.violations.append(
                                f"Reshape to 6D shape detected: {shape_value.tolist()}"
                            )
                            print(f"    ❌ VIOLATION: Reshape to 6D")
                        elif any(shape_value == -1):
                            self.warnings.append(
                                f"Reshape with dynamic dimension (-1): {shape_value.tolist()}"
                            )
                            print(f"    ⚠️  WARNING: Dynamic dimension (-1)")
                        else:
                            print(f"    ✅ OK")

        print(f"\n  Total Reshape nodes: {reshape_count}")

    def check_squeeze_nodes(self):
        """檢查 Squeeze 節點（axes 必須是靜態屬性）"""
        print(f"\n{'─'*70}")
        print("🗜️  Squeeze/Unsqueeze Node Analysis")
        print(f"{'─'*70}")

        for op_type in ['Squeeze', 'Unsqueeze']:
            nodes = [n for n in self.graph.node if n.op_type == op_type]

            if not nodes:
                print(f"\n  {op_type}: None")
                continue

            print(f"\n  {op_type} nodes: {len(nodes)}")

            for node in nodes:
                # Opset 13+: axes 可能是第二個輸入
                if len(node.input) > 1:
                    axes_input = node.input[1]
                    axes_value = self._find_constant_value(axes_input)

                    if axes_value is None:
                        self.violations.append(
                            f"{op_type} node '{node.output[0]}' uses tensor input for axes (not supported)"
                        )
                        print(f"    ❌ '{node.output[0]}': axes from tensor (VIOLATION)")
                    else:
                        print(f"    ✅ '{node.output[0]}': axes = {axes_value.tolist()}")

                # Opset 11-12: axes 可能是屬性
                else:
                    has_axes_attr = any(attr.name == 'axes' for attr in node.attribute)
                    if has_axes_attr:
                        axes = [attr.ints for attr in node.attribute if attr.name == 'axes'][0]
                        print(f"    ✅ '{node.output[0]}': axes = {list(axes)} (attribute)")
                    else:
                        print(f"    ⚠️  '{node.output[0]}': no explicit axes")

    def check_batch_dimensions(self):
        """檢查 Conv/Pooling 層的 batch 維度"""
        print(f"\n{'─'*70}")
        print("🔢 Batch Dimension Analysis (Conv/Pool layers)")
        print(f"{'─'*70}")

        conv_ops = ['Conv', 'ConvTranspose', 'MaxPool', 'AveragePool', 'GlobalAveragePool']

        batch_violations = []

        for node in self.graph.node:
            if any(node.op_type.startswith(op) for op in conv_ops):
                # 嘗試找到輸入張量的形狀
                input_name = node.input[0]
                input_shape = self._find_tensor_shape(input_name)

                if input_shape:
                    batch = input_shape[0] if input_shape else None

                    if batch and batch != 1 and batch != '?':
                        batch_violations.append((node.name, node.op_type, input_name, batch))
                        print(f"  ❌ {node.op_type} '{node.name or node.output[0]}':")
                        print(f"       Input: {input_name}, Batch: {batch} (MUST BE 1)")

        if batch_violations:
            self.violations.append(
                f"Found {len(batch_violations)} Conv/Pool layers with batch > 1"
            )
        else:
            print(f"  ✅ All Conv/Pool layers appear to have batch=1 or unknown")

    def check_unsupported_ops(self):
        """檢查不支援的運算"""
        print(f"\n{'─'*70}")
        print("⚙️  Operation Support Check")
        print(f"{'─'*70}")

        # ST 已知不支援的運算（示例列表，需根據實際文檔更新）
        unsupported_ops = {
            'Loop', 'Scan', 'If', 'SequenceConstruct', 'SequenceAt',
            'NonMaxSuppression', 'RoiAlign', 'TopK'
        }

        op_counts = defaultdict(int)
        for node in self.graph.node:
            op_counts[node.op_type] += 1

        print(f"\n  Operations used:")
        for op_type in sorted(op_counts.keys()):
            count = op_counts[op_type]
            if op_type in unsupported_ops:
                print(f"    ❌ {op_type}: {count} (UNSUPPORTED)")
                self.violations.append(f"Unsupported operation: {op_type}")
            else:
                print(f"    ✅ {op_type}: {count}")

    def print_report(self):
        """打印最終報告"""
        print(f"\n{'='*70}")
        print("📊 FINAL DIAGNOSTIC REPORT")
        print(f"{'='*70}")

        print(f"\n總計檢查項目:")
        print(f"  - ❌ Violations: {len(self.violations)}")
        print(f"  - ⚠️  Warnings: {len(self.warnings)}")

        if self.violations:
            print(f"\n❌ VIOLATIONS (必須修復):")
            for i, v in enumerate(self.violations, 1):
                print(f"  {i}. {v}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS (建議修復):")
            for i, w in enumerate(self.warnings, 1):
                print(f"  {i}. {w}")

        print(f"\n{'='*70}")
        if not self.violations:
            print("✅ MODEL IS STM32N6-COMPATIBLE!")
            print("Ready for stedgeai analyze --model ... --target stm32n6")
        else:
            print("❌ MODEL HAS VIOLATIONS - MUST FIX BEFORE STM32 DEPLOYMENT")
            print("Use fix_onnx_for_stm32.py to repair the model")
        print(f"{'='*70}")

    # Helper methods

    def _get_shape(self, tensor_shape):
        """獲取張量形狀（包括動態維度）"""
        shape = []
        for dim in tensor_shape.dim:
            if dim.HasField('dim_value'):
                shape.append(dim.dim_value)
            elif dim.HasField('dim_param'):
                shape.append(f'?({dim.dim_param})')
            else:
                shape.append('?')
        return shape

    def _check_dynamic_dims(self, tensor_shape):
        """檢查是否有動態維度"""
        for dim in tensor_shape.dim:
            if dim.HasField('dim_param'):
                return True
        return False

    def _find_constant_value(self, name):
        """查找 Constant 節點的值"""
        for node in self.graph.node:
            if node.op_type == 'Constant' and node.output[0] == name:
                for attr in node.attribute:
                    if attr.name == 'value' and attr.HasField('t'):
                        return numpy_helper.to_array(attr.t)
        return None

    def _find_tensor_shape(self, name):
        """查找張量的形狀"""
        # 檢查 value_info
        for value_info in self.graph.value_info:
            if value_info.name == name and value_info.type.tensor_type.HasField('shape'):
                return self._get_shape(value_info.type.tensor_type.shape)

        # 檢查 input
        for input_tensor in self.graph.input:
            if input_tensor.name == name:
                return self._get_shape(input_tensor.type.tensor_type.shape)

        return None


def main():
    parser = argparse.ArgumentParser(description='STM32N6 ONNX Diagnostics')
    parser.add_argument('--onnx', type=str, required=True, help='Path to ONNX model')
    args = parser.parse_args()

    # 執行診斷
    diagnostics = STM32N6Diagnostics(args.onnx)
    is_compatible = diagnostics.diagnose_all()

    # 返回狀態碼
    exit(0 if is_compatible else 1)


if __name__ == "__main__":
    main()
