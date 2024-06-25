# resnet50 by onnxruntime

使用 ONNXRuntime 推理 ResNet 50 模型。

## 说明

1. model 可以在 https://github.com/onnx/models/tree/main/validated/vision/classification/resnet 下载。当前的代码只在 resnet50-v2-7.onnx 模型中验证过。
2. 默认不会输出图像的特征向量，先使用 python add_output_layer.py 增加一项目输出。
3. 使用 python main.py 进行验证。

## License

MIT
