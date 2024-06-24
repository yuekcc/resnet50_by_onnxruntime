import numpy as np
import onnxruntime as ort

import preprocess_image

# 初始化ONNX Runtime运行时
# model 文件在 https://github.com/onnx/models/tree/main/validated/vision/classification/resnet 下载
ort_session = ort.InferenceSession("resnet50-v1-12-int8.onnx")

# 准备输入数据，例如一张图片，需要调整至模型期望的形状和类型
image_path = "cat.png"
image = preprocess_image.parse(image_path)  # 假设preprocess_image函数处理图像至(1, 3, 224, 224)的形状和合适的数值范围
# print(image)

# 运行模型推理
outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: image})

# 输出预测结果
predictions = outputs[0]  # 假设输出是一个数组，包含了每个类别的概率
# print(outputs[0])
predicted_class = np.argmax(predictions)
print(f"Predicted class: {predicted_class}")
