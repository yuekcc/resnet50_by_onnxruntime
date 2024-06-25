import onnx
from onnx import helper


model = onnx.load('resnet50-v2-7.onnx')
intermediate_layer_value_info = helper.ValueInfoProto()
intermediate_layer_value_info.name = 'resnetv24_pool1_fwd'
model.graph.output.append(intermediate_layer_value_info)
onnx.save(model, 'resnet50-v2-7__mod.onnx')