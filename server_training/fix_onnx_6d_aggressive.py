"""
激進的 6D 張量修復 - 直接移除第一個 Reshape 節點
"""
import onnx
import onnx_graphsurgeon as gs
import numpy as np
import argparse
from pathlib import Path


def fix_6d_reshape_aggressive(input_path, output_path):
    print(f"Loading ONNX: {input_path}")
    model = onnx.load(str(input_path))
    graph = gs.import_onnx(model)
    
    print(f"\n{'='*70}")
    print("Aggressive 6D Reshape Removal")
    print(f"{'='*70}\n")
    
    # 策略：找到所有 Reshape 節點，檢查輸出是否為 6D
    nodes_to_remove = []
    fixes = []
    
    for node in graph.nodes:
        if node.op == "Reshape":
            # 檢查輸出張量的 shape
            if len(node.outputs) > 0:
                output_tensor = node.outputs[0]
                
                # 嘗試獲取輸出 shape
                if output_tensor.shape is not None and len(output_tensor.shape) > 5:
                    print(f"❌ Found 6D Reshape: {node.name}")
                    print(f"   Output shape: {output_tensor.shape}")
                    print(f"   Will bypass this node\n")
                    
                    # 策略：將此節點的輸入直接連接到所有消費者
                    input_tensor = node.inputs[0]
                    
                    # 重新連接所有使用這個輸出的節點
                    for consumer in list(output_tensor.outputs):
                        for i, inp in enumerate(consumer.inputs):
                            if inp == output_tensor:
                                consumer.inputs[i] = input_tensor
                                print(f"   → Reconnected {consumer.name} input[{i}] to {input_tensor.name}")
                    
                    nodes_to_remove.append(node)
                    fixes.append(f"Removed 6D Reshape: {node.name}")
                
                # 備用方案：檢查 shape input 的值
                elif len(node.inputs) > 1:
                    shape_input = node.inputs[1]
                    
                    # 如果是 Constant
                    if isinstance(shape_input, gs.Constant):
                        target_shape = shape_input.values
                        if len(target_shape) > 5:
                            print(f"❌ Found 6D Reshape (via Constant): {node.name}")
                            print(f"   Target shape: {target_shape.tolist()}")
                            print(f"   Will bypass this node\n")
                            
                            input_tensor = node.inputs[0]
                            output_tensor = node.outputs[0]
                            
                            for consumer in list(output_tensor.outputs):
                                for i, inp in enumerate(consumer.inputs):
                                    if inp == output_tensor:
                                        consumer.inputs[i] = input_tensor
                                        print(f"   → Reconnected {consumer.name} input[{i}]")
                            
                            nodes_to_remove.append(node)
                            fixes.append(f"Removed 6D Reshape: {node.name}")
    
    # 移除節點
    for node in nodes_to_remove:
        if node in graph.nodes:
            graph.nodes.remove(node)
    
    if nodes_to_remove:
        print(f"\n✅ Removed {len(nodes_to_remove)} 6D Reshape nodes")
    else:
        print("\n⚠️  No 6D Reshape nodes found")
    
    # 清理和優化
    print("\nOptimizing graph...")
    graph.cleanup()
    graph.toposort()
    
    # 導出
    print(f"\nSaving to: {output_path}")
    fixed_model = gs.export_onnx(graph)
    onnx.save(fixed_model, str(output_path))
    
    print(f"✅ Fixed model saved")
    print(f"Size: {Path(output_path).stat().st_size / 1024:.2f} KB")
    
    print(f"\n{'='*70}")
    print(f"Fixes applied: {len(fixes)}")
    for fix in fixes:
        print(f"  - {fix}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    fix_6d_reshape_aggressive(args.input, args.output)
