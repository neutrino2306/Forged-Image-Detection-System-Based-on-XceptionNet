import cv2
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D


# 提取RGB通道
def extract_rgb_channels(image):
    r_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    b_channel = image[:, :, 2]
    return r_channel, g_channel, b_channel


# 定义SRM滤波器
def srm_filter(image):
    filters = [
        np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]]),  # 水平边缘检测
        np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]]),  # 垂直边缘检测
        np.array([[1, -1, 0], [-1, 1, 0], [0, 0, 0]])   # 对角线边缘检测
    ]
    feature_maps = []
    for f in filters:
        feature_map = cv2.filter2D(image, -1, f)
        feature_maps.append(feature_map)
    return np.stack(feature_maps, axis=-1)


# 对每个通道应用SRM滤波器
def apply_srm_filters(r_channel, g_channel, b_channel):
    r_features = srm_filter(r_channel)
    g_features = srm_filter(g_channel)
    b_features = srm_filter(b_channel)
    return r_features, g_features, b_features


# 生成9通道特征图
def generate_9_channel_feature_map(r_features, g_features, b_features):
    return np.concatenate([r_features, g_features, b_features], axis=-1)


# 1x1点卷积操作
def pointwise_conv(inputs, filters):
    return Conv2D(filters, (1, 1), padding='same')(inputs)


# 融合RGB和SRM特征
def fuse_features(rgb_features, srm_features):
    srm_3_channel = pointwise_conv(srm_features, 3)
    fused_features = tf.add(rgb_features, srm_3_channel)
    return fused_features
