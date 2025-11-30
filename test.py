import onnx

model = onnx.load("rppg_4d_fp32.onnx")

for node in model.graph.node:
    print("NODE:", node.name, "OP:", node.op_type)
    for attr in node.attribute:
        print("   ATTR:", attr.name, "=>", attr)
    print("----")
