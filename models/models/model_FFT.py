import os

import tensorflow as tf
from keras import Sequential
from keras import backend as K, Input
from keras.applications import Xception
from keras.models import Model
from keras.layers import ReLU, add, Dense, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Flatten, \
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


def hfreqWH(x, scale):
    assert scale > 2
    # 将输入转置，使得 h 和 w 维度成为最后两个维度
    x = tf.transpose(x, perm=[0, 3, 1, 2])

    # 进行2D傅里叶变换
    # x = tf.signal.fft2d(tf.cast(x, tf.complex64))
    h, w = x.shape[-2], x.shape[-1]
    norm_factor = tf.cast(tf.sqrt(tf.cast(h * w, tf.float32)), tf.complex64)
    x = tf.signal.fft2d(tf.cast(x, tf.complex64)) / norm_factor

    # 进行频域平移，将低频部分移动到中心
    x = tf.signal.fftshift(x, axes=[-2, -1])
    b, c, h, w = x.shape
    # 创建掩码，将低频部分设置为0
    mask = np.ones((h, w), dtype=np.complex64)
    mask[h // 2 - h // scale:h // 2 + h // scale, w // 2 - w // scale:w // 2 + w // scale] = 0
    mask = tf.convert_to_tensor(mask)
    # 扩张维度以匹配图像的形状
    mask = tf.expand_dims(mask, axis=0)
    mask = tf.expand_dims(mask, axis=0)
    # 应用掩码进行低频滤波
    x = x * mask
    # 逆频域平移，将频谱还原
    x = tf.signal.ifftshift(x, axes=[-2, -1])
    # 逆2D傅里叶变换
    # x = tf.signal.ifft2d(x)
    x = tf.signal.ifft2d(x) * norm_factor
    # 将输出转置回原来的维度顺序
    x = tf.transpose(x, perm=[0, 2, 3, 1])
    # 取实部
    # x = tf.math.real(x)
    x = ensure_real(x)
    # ReLU激活
    x = ReLU()(x)

    print(f"hfreqWH output shape: {x.shape}, dtype: {x.dtype}")

    return x


def hfreqC(x, scale):
    assert scale > 2
    # 对第4个维度进行1D傅里叶变换
    # x = tf.signal.fft(tf.cast(x, tf.complex64), axis=3)
    # x = tf.signal.fft(tf.cast(x, tf.complex64))
    c = x.shape[-1]
    norm_factor = tf.cast(tf.sqrt(tf.cast(c, tf.float32)), tf.complex64)
    x = tf.signal.fft(tf.cast(x, tf.complex64)) / norm_factor

    # 进行频域平移，将低频部分移动到中心
    # x = tf.signal.fftshift(x, axes=3)
    x = tf.signal.fftshift(x, axes=[-1])
    b, h, w, c = x.shape
    # 创建掩码，将低频部分设置为0
    mask = np.ones((c,), dtype=np.complex64)
    mask[c // 2 - c // scale:c // 2 + c // scale] = 0
    mask = tf.convert_to_tensor(mask)
    mask = tf.reshape(mask, (1, 1, 1, c))
    # 应用掩码进行低频滤波
    x = x * mask
    # 逆频域平移，将频谱还原
    # x = tf.signal.ifftshift(x, axes=3)
    x = tf.signal.ifftshift(x, axes=[-1])
    # 逆1D傅里叶变换
    # x = tf.signal.ifft(x)
    x = tf.signal.ifft(x) * norm_factor
    # x = tf.signal.ifft(x, axis=3)
    # 取实部
    # x = tf.math.real(x)
    x = ensure_real(x)
    # ReLU激活
    x = ReLU()(x)

    print(f"hfreqC output shape: {x.shape}, dtype: {x.dtype}")

    return x


def frequency_conv_layer(x, filters):
    # 将输入张量的维度从 (batch_size, height, width, channels) 转换为 (batch_size, channels, height, width)
    x = tf.transpose(x, perm=[0, 3, 1, 2])

    # 获取 height 和 width
    h, w = x.shape[-2], x.shape[-1]
    norm_factor = tf.cast(tf.sqrt(tf.cast(h * w, tf.float32)), tf.complex64)

    # 进行2D傅里叶变换，对height和width维度进行变换
    x = tf.signal.fft2d(tf.cast(x, tf.complex64)) / norm_factor

    # 进行2D傅里叶变换，对height和width维度进行变换
    # x = tf.signal.fft2d(tf.cast(x, tf.complex64))

    # 进行频域平移，将低频部分移动到中心
    x = tf.signal.fftshift(x, axes=[-2, -1])

    real_part = tf.math.real(x)
    imag_part = tf.math.imag(x)

    # 交换回 channels 作为最后一维
    real_part = tf.transpose(real_part, perm=[0, 2, 3, 1])
    imag_part = tf.transpose(imag_part, perm=[0, 2, 3, 1])

    # 对频域信号的实部和虚部分别进行卷积操作
    real_conv = Conv2D(filters, kernel_size=1, strides=1, padding='same', use_bias=False,
                       kernel_initializer=HeNormal())(real_part)
    imag_conv = Conv2D(filters, kernel_size=1, strides=1, padding='same', use_bias=False,
                       kernel_initializer=HeNormal())(imag_part)

    real_conv = tf.transpose(real_conv, perm=[0, 3, 1, 2])
    imag_conv = tf.transpose(imag_conv, perm=[0, 3, 1, 2])

    x = tf.complex(real_conv, imag_conv)

    # 逆频域平移，将频谱的零频率成分移回左上角
    x = tf.signal.ifftshift(x, axes=[-2, -1])

    # 逆2D傅里叶变换，将信号从频域转换回时域
    # x = tf.signal.ifft2d(x)
    x = tf.signal.ifft2d(x) * norm_factor

    # 将输出转置回原来的维度顺序 (batch_size, height, width, channels)
    x = tf.transpose(x, perm=[0, 2, 3, 1])

    # 确保数据为实数
    x = ensure_real(x)

    # ReLU激活
    x = ReLU()(x)

    print(f"frequency_conv_layer output shape: {x.shape}, dtype: {x.dtype}")

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
    # 第一层卷积
    x = hfreqWH(input_tensor, 4)
    print(f"After hfreqWH: shape = {x.shape}, dtype = {x.dtype}")

    x = Conv2D(64, kernel_size=1, strides=1, padding='VALID', use_bias=True, kernel_initializer=HeNormal())(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    print(f"After Conv2D and ReLU: shape = {x.shape}, dtype = {x.dtype}")

    # 第二层卷积
    x = hfreqC(x, 4)
    print(f"After hfreqC: shape = {x.shape}, dtype = {x.dtype}")

    # 频域卷积层
    x = frequency_conv_layer(x, 64)
    print(f"After frequency_conv_layer: shape = {x.shape}, dtype = {x.dtype}")

    # 高频滤波和卷积（带下采样）
    x = hfreqWH(x, 4)
    x = Conv2D(64, kernel_size=1, strides=2, padding='VALID', use_bias=True, kernel_initializer=HeNormal())(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    print(f"After second Conv2D and ReLU: shape = {x.shape}, dtype = {x.dtype}")

    # 高频滤波
    x = hfreqC(x, 4)
    print(f"After second hfreqC: shape = {x.shape}, dtype = {x.dtype}")

    # 频域卷积层
    x = frequency_conv_layer(x, 64)
    print(f"After second frequency_conv_layer: shape = {x.shape}, dtype = {x.dtype}")

    # 最大池化层
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    print(f"After MaxPooling2D: shape = {x.shape}, dtype = {x.dtype}")

    # 残差层
    x = make_layer(x, bottleneck, 64, 3)
    print(f"After make_layer: shape = {x.shape}, dtype = {x.dtype}")

    # 高频滤波和卷积
    x = hfreqWH(x, 4)
    x = Conv2D(256, kernel_size=1, strides=1, padding='VALID', use_bias=True, kernel_initializer=HeNormal())(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    print(f"After third Conv2D and ReLU: shape = {x.shape}, dtype = {x.dtype}")

    # 频域卷积层
    x = frequency_conv_layer(x, 256)
    print(f"After third frequency_conv_layer: shape = {x.shape}, dtype = {x.dtype}")

    # 高频滤波和卷积（带下采样）
    x = hfreqWH(x, 4)
    x = Conv2D(256, kernel_size=1, strides=2, padding='VALID', use_bias=True, kernel_initializer=HeNormal())(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    print(f"After fourth Conv2D and ReLU: shape = {x.shape}, dtype = {x.dtype}")

    # 频域卷积层
    x = frequency_conv_layer(x, 256)
    print(f"After fourth frequency_conv_layer: shape = {x.shape}, dtype = {x.dtype}")

    x = make_layer(x, bottleneck, 128, 4, stride=2)  # Layer2
    print(f"After second make_layer: shape = {x.shape}, dtype = {x.dtype}")

    # 高频滤波和卷积
    x = hfreqWH(x, 4)
    x = Conv2D(512, kernel_size=1, strides=1, padding='VALID', use_bias=True, kernel_initializer=HeNormal())(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    print(f"After fifth Conv2D and ReLU: shape = {x.shape}, dtype = {x.dtype}")

    # 频域卷积层
    x = frequency_conv_layer(x, 512)
    print(f"After fifth frequency_conv_layer: shape = {x.shape}, dtype = {x.dtype}")

    # 残差层3
    x = make_layer(x, bottleneck, 512, 4, stride=2)  # 直接输出通道数 2048
    print(f"After third make_layer: shape = {x.shape}, dtype = {x.dtype}")

    # 全局平均池化和全连接层
    # x = GlobalAveragePooling2D()(x)
    # x = Flatten()(x)

    # return Model(inputs, outputs)
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
# # 确保模型中的所有层输出的数据类型为 float32，忽略特定的傅里叶变换层
# for layer in model.layers:
#     output = layer.output
#     print(f"Layer {layer.name} output shape: {output.shape}, dtype: {output.dtype}")
#     # 忽略特定的傅里叶变换层
#     if 'fft' in layer.name or 'ifft' in layer.name:
#         continue
#     if tf.as_dtype(output.dtype) == tf.complex64:
#         raise ValueError(f"Layer {layer.name} has complex output")


# model.save('my_model.h5')
