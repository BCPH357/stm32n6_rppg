"""
找出產生 6D 張量的節點
"""
import onnx
import argparse

def find_6d_producer(onnx_path):
    print(f"Loading ONNX: {onnx_path}")
    model = onnx.load(onnx_path)
    graph = model.graph
    
    # 建立 value_info 映射
    all_tensors = {}
    for tensor in list(graph.input) + list(graph.output) + list(graph.value_info):
        if tensor.type.tensor_type.shape.dim:
            rank = len(tensor.type.tensor_type.shape.dim)
            shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in tensor.type.tensor_type.shape.dim]
            all_tensors[tensor.name] = (rank, shape)
    
    print(f"\n{'='*70}")
    print("6D Tensor Analysis")
    print(f"{'='*70}\n")
    
    # 找出所有 6D 張量
    tensor_6d = {name: (rank, shape) for name, (rank, shape) in all_tensors.items() if rank == 6}
    
    if not tensor_6d:
        print("✅ No 6D tensors found!")
        return
    
    print(f"Found {len(tensor_6d)} 6D tensors:\n")
    
    for tensor_name, (rank, shape) in tensor_6d.items():
        print(f"  ❌ {tensor_name}: {shape}")
        
        # 找出產生這個張量的節點
        producer = None
        for node in graph.node:
            if tensor_name in node.output:
                producer = node
                break
        
        if producer:
            print(f"     Produced by: {producer.op_type} (name: {producer.name or 'unnamed'})")
            print(f"     Inputs: {list(producer.input)}")
            
            # 如果是 Reshape，嘗試找出 shape 來源
            if producer.op_type == "Reshape" and len(producer.input) > 1:
                shape_input = producer.input[1]
                print(f"     Shape input: {shape_input}")
                
                # 找出 shape 的來源
                shape_producer = None
                for node in graph.node:
                    if shape_input in node.output:
                        shape_producer = node
                        break
                
                if shape_producer:
                    print(f"     Shape produced by: {shape_producer.op_type} (name: {shape_producer.name or 'unnamed'})")
                    
                    # 如果是 Constant
                    if shape_producer.op_type == "Constant":
                        for attr in shape_producer.attribute:
                            if attr.name == "value":
                                import numpy as np
                                shape_value = onnx.numpy_helper.to_array(attr.t)
                                print(f"     Shape value: {shape_value.tolist()}")
        
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', required=True, help='ONNX model path')
    args = parser.parse_args()
    
    find_6d_producer(args.onnx)
