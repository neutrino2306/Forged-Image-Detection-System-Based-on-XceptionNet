import os

import tensorflow as tf
from keras import Sequential
from keras import backend as K, Input
from keras.applications import Xception
from keras.models import Model
from keras.layers import Layer, ReLU, add, Dense, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Flatten, \
    Concatenate, Dropout, \
    BatchNormalization, Activation, Reshape, Add, Multiply, Lambda, Conv2D, MultiHeadAttention
from keras.regularizers import l1_l2
import numpy as np
from keras.src.initializers.initializers import HeNormal

os.environ[
    'TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all messages are logged (default behavior), 1 = filter out INFO messages, 2 = filter out WARNING messages, 3 = filter out all messages


def ensure_real(x):
    if tf.as_dtype(x.dtype) == tf.complex64:
        x = tf.math.real(x)
    return tf.cast(x, tf.float32)


def basic_block(x, filters, stride=1, downsample=None):
    identity = x

    out = Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False,
                 kernel_initializer=tf.keras.initializers.HeNormal())(x)
    out = BatchNormalization()(out)
    out = ReLU()(out)

    out = Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False,
                 kernel_initializer=tf.keras.initializers.HeNormal())(out)
    out = BatchNormalization()(out)

    if downsample is not None:
        identity = downsample(x)

    out = add([out, identity])
    out = ReLU()(out)

    return out


def bottleneck(x, filters, stride=1, downsample=None):
    identity = x

    out = Conv2D(filters, kernel_size=1, strides=1, padding='same', use_bias=False,
                 kernel_initializer=tf.keras.initializers.HeNormal())(x)
    out = BatchNormalization()(out)
    out = ReLU()(out)

    out = Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False,
                 kernel_initializer=tf.keras.initializers.HeNormal())(out)
    out = BatchNormalization()(out)
    out = ReLU()(out)

    out = Conv2D(filters * 4, kernel_size=1, strides=1, padding='same', use_bias=False,
                 kernel_initializer=tf.keras.initializers.HeNormal())(out)
    out = BatchNormalization()(out)

    if downsample is not None:
        identity = downsample(x)

    out = add([out, identity])
    out = ReLU()(out)

    return out


class HFreqWHLayer(Layer):
    def __init__(self, scale):
        super(HFreqWHLayer, self).__init__()
        self.scale = scale

    def call(self, x):
        assert self.scale > 2
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        h, w = x.shape[-2], x.shape[-1]
        norm_factor = tf.cast(tf.sqrt(tf.cast(h * w, tf.float32)), tf.complex64)
        x = tf.signal.fft2d(tf.cast(x, tf.complex64)) / norm_factor
        x = tf.signal.fftshift(x, axes=[-2, -1])
        mask = np.ones((h, w), dtype=np.complex64)
        mask[h // 2 - h // self.scale:h // 2 + h // self.scale, w // 2 - w // self.scale:w // 2 + w // self.scale] = 0
        mask = tf.convert_to_tensor(mask)
        mask = tf.expand_dims(mask, axis=0)
        mask = tf.expand_dims(mask, axis=0)
        x = x * mask
        x = tf.signal.ifftshift(x, axes=[-2, -1])
        x = tf.signal.ifft2d(x) * norm_factor
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        x = ensure_real(x)
        x = ReLU()(x)
        return x


class HFreqCLayer(Layer):
    def __init__(self, scale):
        super(HFreqCLayer, self).__init__()
        self.scale = scale

    def call(self, x):
        assert self.scale > 2
        c = x.shape[-1]
        norm_factor = tf.cast(tf.sqrt(tf.cast(c, tf.float32)), tf.complex64)
        x = tf.signal.fft(tf.cast(x, tf.complex64)) / norm_factor
        x = tf.signal.fftshift(x, axes=[-1])
        mask = np.ones((c,), dtype=np.complex64)
        mask[c // 2 - c // self.scale:c // 2 + c // self.scale] = 0
        mask = tf.convert_to_tensor(mask)
        mask = tf.reshape(mask, (1, 1, 1, c))
        x = x * mask
        x = tf.signal.ifftshift(x, axes=[-1])
        x = tf.signal.ifft(x) * norm_factor
        x = ensure_real(x)
        x = ReLU()(x)
        return x


class FrequencyConvLayer(Layer):
    def __init__(self, filters):
        super(FrequencyConvLayer, self).__init__()
        self.filters = filters
        self.conv_real = Conv2D(filters, kernel_size=1, strides=1, padding='same', use_bias=False, kernel_initializer=HeNormal())
        self.conv_imag = Conv2D(filters, kernel_size=1, strides=1, padding='same', use_bias=False, kernel_initializer=HeNormal())

    def call(self, x):
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        h, w = x.shape[-2], x.shape[-1]
        norm_factor = tf.cast(tf.sqrt(tf.cast(h * w, tf.float32)), tf.complex64)
        x = tf.signal.fft2d(tf.cast(x, tf.complex64)) / norm_factor
        x = tf.signal.fftshift(x, axes=[-2, -1])
        real_part = tf.math.real(x)
        imag_part = tf.math.imag(x)
        real_part = tf.transpose(real_part, perm=[0, 2, 3, 1])
        imag_part = tf.transpose(imag_part, perm=[0, 2, 3, 1])
        real_conv = self.conv_real(real_part)
        imag_conv = self.conv_imag(imag_part)
        real_conv = tf.cast(real_conv, tf.float32)
        imag_conv = tf.cast(imag_conv, tf.float32)
        real_conv = tf.transpose(real_conv, perm=[0, 3, 1, 2])
        imag_conv = tf.transpose(imag_conv, perm=[0, 3, 1, 2])
        x = tf.complex(real_conv, imag_conv)
        x = tf.signal.ifftshift(x, axes=[-2, -1])
        x = tf.signal.ifft2d(x) * norm_factor
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        x = ensure_real(x)
        x = ReLU()(x)
        return x


def make_layer(x, block, filters, blocks, stride=1):
    downsample = None
    inplanes = x.shape[-1]
    expansion = 4
    if stride != 1 or inplanes != filters * expansion:
        downsample = Sequential([
            Conv2D(filters * expansion, kernel_size=1, strides=stride, padding='same', use_bias=False,
                   kernel_initializer=tf.keras.initializers.HeNormal()),
            BatchNormalization()
        ])

    x = block(x, filters, stride, downsample)
    for _ in range(1, blocks):
        x = block(x, filters)

    return x


def freqnet(input_tensor):
    x = HFreqWHLayer(4)(input_tensor)
    x = Conv2D(64, kernel_size=1, strides=1, padding='VALID', use_bias=True, kernel_initializer=HeNormal())(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = HFreqCLayer(4)(x)
    x = FrequencyConvLayer(64)(x)

    x = HFreqWHLayer(4)(x)
    x = Conv2D(64, kernel_size=1, strides=2, padding='VALID', use_bias=True, kernel_initializer=HeNormal())(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = HFreqCLayer(4)(x)
    x = FrequencyConvLayer(64)(x)

    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    x = make_layer(x, bottleneck, 64, 3)

    x = HFreqWHLayer(4)(x)
    x = Conv2D(256, kernel_size=1, strides=1, padding='VALID', use_bias=True, kernel_initializer=HeNormal())(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = FrequencyConvLayer(256)(x)

    x = HFreqWHLayer(4)(x)
    x = Conv2D(256, kernel_size=1, strides=2, padding='VALID', use_bias=True, kernel_initializer=HeNormal())(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = FrequencyConvLayer(256)(x)

    x = make_layer(x, bottleneck, 128, 4, stride=2)
    x = HFreqWHLayer(4)(x)
    x = Conv2D(512, kernel_size=1, strides=1, padding='VALID', use_bias=True, kernel_initializer=HeNormal())(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = FrequencyConvLayer(512)(x)

    x = make_layer(x, bottleneck, 512, 4, stride=2)
    return x


def mean(x):
    return K.mean(x, axis=3, keepdims=True)


def max(x):
    return K.max(x, axis=3, keepdims=True)


def channel_attention(input_feature, ratio=8):
    channel_axis = -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    # return Multiply()([input_feature, cbam_feature])

    output = Multiply()([input_feature, cbam_feature])
    print(f"channel_attention output shape: {output.shape}, dtype: {output.dtype}")

    return output


def spatial_attention(input_feature):
    kernel_size = 7

    # lamda层其实是一种自定义的层，操作方法就是前面定义的函数
    avg_pool = Lambda(mean)(input_feature)
    max_pool = Lambda(max)(input_feature)
    # 假设avg_pool和max_pool每个都有C个通道，合并后的结果将有2C个通道，其中前C个通道来自avg_pool，接下来的C个通道来自max_pool
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',  # 保证输出和输入具有相同的空间维度
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)

    # 原图中每个像素的所有通道都与空间注意力图的对应位置的通道值相乘
    # return Multiply()([input_feature, cbam_feature])
    output = Multiply()([input_feature, cbam_feature])
    print(f"spatial_attention output shape: {output.shape}, dtype: {output.dtype}")

    return output


def attention_fusion(x1, x2):
    query = x1
    key_value = x2
    attention_output = MultiHeadAttention(num_heads=8, key_dim=x1.shape[-1])(query, key_value)
    print(f"attention_fusion output shape: {attention_output.shape}, dtype: {attention_output.dtype}")

    return attention_output


def complex_fusion(x1, x2):
    # 特征交互
    interaction = Multiply()([x1, x2])

    # 通道注意力
    x1_channel = channel_attention(x1)

    # 注意力融合
    attention_output = attention_fusion(x1_channel, x2)

    # 自适应权重
    alpha = Dense(1, activation='sigmoid')(x1)
    beta = Dense(1, activation='sigmoid')(x2)
    weighted_x1 = Multiply()([x1, alpha])
    weighted_x2 = Multiply()([x2, beta])

    # 特征组合
    combined = Concatenate()([weighted_x1, weighted_x2, interaction, attention_output])
    print(f"complex_fusion output shape: {combined.shape}, dtype: {combined.dtype}")

    return combined


def build_model():
    input_tensor = Input(shape=(299, 299, 3))

    # 加载预训练的Xception模型
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_tensor=input_tensor
    )

    # 冻结预训练模型的权重
    for layer in base_model.layers:
        layer.trainable = False

    # 第一分支：傅里叶变换和残差网络
    x1 = freqnet(input_tensor)

    # 特征提取层
    x2 = base_model.output
    original_x = x2

    # CBAM模块
    x2 = channel_attention(x2, ratio=8)
    x2 = spatial_attention(x2)

    # 在CBAM后添加跳层连接
    x2 = Add()([x2, original_x])

    # 全局平均池化层
    # x2 = GlobalAveragePooling2D()(x2)
    # x2 = Flatten()(x2)

    combined = complex_fusion(x1, x2)

    combined_pool = GlobalAveragePooling2D()(combined)

    # Dropout层
    x = Dropout(0.5)(combined_pool)

    # 预测层
    predictions = Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(x)

    # 构建完整模型
    model = Model(inputs=input_tensor, outputs=predictions)

    return model


model = build_model()
model.summary()