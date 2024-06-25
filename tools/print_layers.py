import onnx


# 加载模型

model = onnx.load('resnet50-v2-7__mod.onnx')


# 打印模型的图形结构

print(onnx.helper.printable_graph(model.graph))