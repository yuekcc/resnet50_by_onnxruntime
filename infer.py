import numpy as np
import onnxruntime as ort

import preprocess_image

with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]


def parse(image_path):
    ort_session = ort.InferenceSession("resnet50-v2-7__mod.onnx")
    image = preprocess_image.parse(image_path)

    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    feature_layer_name = 'resnetv24_pool1_fwd' # 特征层的名称，model文件不同，名称不同
    outputs = ort_session.run([output_name, feature_layer_name], {input_name: image})

    # 输出预测结果
    predictions = outputs[0]  # 假设输出是一个数组，包含了每个类别的概率
    predictions = np.squeeze(predictions)
    predicted_class = np.argsort(predictions)[::-1]
    # print('class=%s ; probability=%f' % (labels[predicted_class[0]], predictions[predicted_class[0]]))

    feature_vector = outputs[1].flatten()
    image_class = labels[predicted_class[0]]
    image_class_id = predictions[predicted_class[0]]
    return image_class, image_class_id, feature_vector
