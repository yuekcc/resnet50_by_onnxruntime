import cv2
import numpy as np
from PIL import Image


def parse(image_path):
    """
    预处理图像，使其适合ResNet模型输入
    :param image_path: 图像文件路径
    :return: 预处理后的图像数组，形状为(1, 3, 224, 224)，值范围为0-1
    """
    # 使用PIL读取图像
    img = Image.open(image_path).convert('RGB')

    # 调整图像大小到ResNet模型所需的输入尺寸
    pillow_img = img.resize((224, 224))

    # 将PIL图像转换为OpenCV格式（注意OpenCV使用BGR格式，但此处我们直接转为NumPy数组后调整通道顺序）
    input_data = np.float32(pillow_img) - np.array(
        [123.68, 116.78, 103.94], dtype=np.float32
    )
    nhwc_data = np.expand_dims(input_data, axis=0)
    nchw_data = nhwc_data.transpose(0, 3, 1, 2)

    return nchw_data
# 
# # 示例使用
# image_path = 'path_to_your_image.jpg'
# processed_image = preprocess_image(image_path)
