import tensorflow as tf
from keras import Sequential, Input
from keras import backend as K
from keras.applications import Xception
# from keras.initializers.initializers_v2 import HeNormal
from keras.initializers import HeNormal

from keras.models import Model
from keras.layers import ReLU, add, Dense, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Flatten, \
    Concatenate, Dropout, BatchNormalization, Activation, Reshape, Add, Multiply, Lambda, Conv2D, MultiHeadAttention
from keras.regularizers import l1_l2
import numpy as np


def ensure_real(x):
    if tf.as_dtype(x.dtype) == tf.complex64:
        x = tf.math.real(x)
    return tf.cast(x, tf.float32)


def bottleneck(x, filters, stride=1, downsample=None):
    identity = x

    out = Conv2D(filters, kernel_size=1, strides=1, padding='same', use_bias=False,
                 kernel_initializer=HeNormal())(x)
    out = BatchNormalization()(out)
    out = ReLU()(out)

    out = Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False,
                 kernel_initializer=HeNormal())(out)
    out = BatchNormalization()(out)
    out = ReLU()(out)

    out = Conv2D(filters * 4, kernel_size=1, strides=1, padding='same', use_bias=False,
                 kernel_initializer=HeNormal())(out)
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
    x = tf.signal.ifft2d(x) * norm_factor
    # 将输出转置回原来的维度顺序
    x = tf.transpose(x, perm=[0, 2, 3, 1])
    # 取实部
    x = ensure_real(x)
    # ReLU激活
    x = ReLU()(x)

    print(f"hfreqWH output shape: {x.shape}, dtype: {x.dtype}")

    return x


def hfreqC(x, scale):
    assert scale > 2
    # 对第4个维度进行1D傅里叶变换
    c = x.shape[-1]
    norm_factor = tf.cast(tf.sqrt(tf.cast(c, tf.float32)), tf.complex64)
    x = tf.signal.fft(tf.cast(x, tf.complex64)) / norm_factor
    # 进行频域平移，将低频部分移动到中心
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
    x = tf.signal.ifftshift(x, axes=[-1])
    # 逆1D傅里叶变换
    x = tf.signal.ifft(x) * norm_factor
    # 取实部
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
    real_conv = tf.cast(real_conv, tf.float32)  # 转换为 float32
    imag_conv = tf.cast(imag_conv, tf.float32)  # 转换为 float32
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

    # print(f"frequency_conv_layer output shape: {x.shape}, dtype: {x.dtype}")

    return x


def make_layer(x, block, filters, blocks, stride=1):
    downsample = None
    inplanes = x.shape[-1]
    expansion = 4
    if stride != 1 or inplanes != filters * expansion:
        downsample = Sequential([
            Conv2D(filters * expansion, kernel_size=1, strides=stride, padding='same', use_bias=False,
                   kernel_initializer=HeNormal()),
            BatchNormalization()
        ])

    x = block(x, filters, stride, downsample)
    for _ in range(1, blocks):
        x = block(x, filters)

    return x


def freqnet(input_tensor):
    # 第一层卷积
    x = hfreqWH(input_tensor, 4)
    x = Conv2D(64, kernel_size=1, strides=1, padding='VALID', use_bias=True, kernel_initializer=HeNormal())(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # 第二层卷积
    x = hfreqC(x, 4)

    # 频域卷积层
    x = frequency_conv_layer(x, 64)

    # 高频滤波和卷积（带下采样）
    x = hfreqWH(x, 4)
    x = Conv2D(64, kernel_size=1, strides=2, padding='VALID', use_bias=True, kernel_initializer=HeNormal())(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # 高频滤波
    x = hfreqC(x, 4)

    # 频域卷积层
    x = frequency_conv_layer(x, 64)

    # 最大池化层
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # 残差层
    x = make_layer(x, bottleneck, 64, 3)

    # 高频滤波和卷积
    x = hfreqWH(x, 4)
    x = Conv2D(256, kernel_size=1, strides=1, padding='VALID', use_bias=True, kernel_initializer=HeNormal())(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # 频域卷积层
    x = frequency_conv_layer(x, 256)

    # 高频滤波和卷积（带下采样）
    x = hfreqWH(x, 4)
    x = Conv2D(256, kernel_size=1, strides=2, padding='VALID', use_bias=True, kernel_initializer=HeNormal())(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # 频域卷积层
    x = frequency_conv_layer(x, 256)

    x = make_layer(x, bottleneck, 128, 4, stride=2)  # Layer2

    # 高频滤波和卷积
    x = hfreqWH(x, 4)
    x = Conv2D(512, kernel_size=1, strides=1, padding='VALID', use_bias=True, kernel_initializer=HeNormal())(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # 频域卷积层
    x = frequency_conv_layer(x, 512)

    # 残差层3
    x = make_layer(x, bottleneck, 512, 4, stride=2)  # 直接输出通道数 2048

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
    return Multiply()([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 7
    avg_pool = Lambda(mean)(input_feature)
    max_pool = Lambda(max)(input_feature)
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid',
                          kernel_initializer='he_normal', use_bias=False)(concat)
    return Multiply()([input_feature, cbam_feature])


def self_attn(x, in_dim):
    query = Conv2D(in_dim // 8, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    key = Conv2D(in_dim // 8, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    value = Conv2D(in_dim, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    query = Flatten()(query)
    key = Flatten()(key)
    value = Flatten()(value)
    attention = tf.nn.softmax(tf.matmul(query, key, transpose_b=True))
    out = tf.matmul(attention, value)
    out = Reshape((x.shape[1], x.shape[2], in_dim))(out)
    gamma = tf.Variable(initial_value=0.0, trainable=True)
    out = gamma * out + x
    return out


def dual_cross_modal_attention(x, y, in_dim):
    def _get_att(a, b):
        proj_key1 = Conv2D(in_dim // 8, kernel_size=1, strides=1, padding='same', use_bias=False)(a)
        proj_key2 = Conv2D(in_dim // 8, kernel_size=1, strides=1, padding='same', use_bias=False)(b)

        # Reshape to [batch_size, height*width, channels]
        proj_key1 = tf.reshape(proj_key1, [tf.shape(proj_key1)[0], -1, tf.shape(proj_key1)[-1]])
        proj_key2 = tf.reshape(proj_key2, [tf.shape(proj_key2)[0], -1, tf.shape(proj_key2)[-1]])

        energy = tf.matmul(proj_key1, proj_key2, transpose_b=True)
        attention1 = tf.nn.softmax(energy)
        attention2 = tf.nn.softmax(tf.transpose(energy, perm=[0, 2, 1]))
        return attention1, attention2

    att_y_on_x, att_x_on_y = _get_att(x, y)

    proj_value_y_on_x = Conv2D(in_dim, kernel_size=1, strides=1, padding='same', use_bias=False)(y)
    proj_value_y_on_x = tf.reshape(proj_value_y_on_x,
                                   [tf.shape(proj_value_y_on_x)[0], -1, tf.shape(proj_value_y_on_x)[-1]])

    out_y_on_x = tf.matmul(att_y_on_x, proj_value_y_on_x)
    out_y_on_x = tf.reshape(out_y_on_x, [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], in_dim])
    gamma1 = tf.Variable(initial_value=0.0, trainable=True)
    out_x = gamma1 * out_y_on_x + x

    proj_value_x_on_y = Conv2D(in_dim, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    proj_value_x_on_y = tf.reshape(proj_value_x_on_y,
                                   [tf.shape(proj_value_x_on_y)[0], -1, tf.shape(proj_value_x_on_y)[-1]])

    out_x_on_y = tf.matmul(att_x_on_y, proj_value_x_on_y)
    out_x_on_y = tf.reshape(out_x_on_y, [tf.shape(y)[0], tf.shape(y)[1], tf.shape(y)[2], in_dim])
    gamma2 = tf.Variable(initial_value=0.0, trainable=True)
    out_y = gamma2 * out_x_on_y + y

    return out_x, out_y


def feature_fusion_module(x, y, out_chan=2048, ratio=16):
    in_chan = x.shape[-1] + y.shape[-1]

    fused_features = Conv2D(out_chan, kernel_size=1, strides=1, padding='same', use_bias=False,
                            kernel_initializer=HeNormal())(Concatenate(axis=-1)([x, y]))
    fused_features = BatchNormalization()(fused_features)
    fused_features = ReLU()(fused_features)

    ca = channel_attention(fused_features, ratio=ratio)

    fused_features = Add()([fused_features, Multiply()([fused_features, ca])])

    return fused_features


def srm_conv2d_separate(input_data, inc, outc, learnable=False):

    def build_kernel(inc):
        # 定义滤波器
        filter1 = np.array([[0, 0, 0, 0, 0],
                            [0, -1, 2, -1, 0],
                            [0, 2, -4, 2, 0],
                            [0, -1, 2, -1, 0],
                            [0, 0, 0, 0, 0]]) / 4.0

        filter2 = np.array([[-1, 2, -2, 2, -1],
                            [2, -6, 8, -6, 2],
                            [-2, 8, -12, 8, -2],
                            [2, -6, 8, -6, 2],
                            [-1, 2, -2, 2, -1]]) / 12.0

        filter3 = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, -2, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]]) / 2.0

        filters = np.array([filter1, filter2, filter3])  # (3, 5, 5)
        filters = np.transpose(filters, (1, 2, 0))  # (5, 5, 3)

        # 扩展维度以匹配 depthwise_conv2d 的输入
        filters = np.repeat(filters[:, :, np.newaxis, :], inc, axis=2)  # (5, 5, inc, 3)
        filters = tf.convert_to_tensor(filters, dtype=tf.float32)
        return filters

    # 构建卷积核
    kernel = build_kernel(inc)

    if learnable:
        # 如果卷积核是可学习的，则将其转化为可训练的变量
        kernel = tf.Variable(kernel, trainable=True, name='srm_kernel')
    else:
        kernel = tf.constant(kernel, name='srm_kernel')

    # 执行深度卷积
    out = tf.nn.depthwise_conv2d(input_data, kernel, strides=[1, 1, 1, 1], padding='SAME')

    # 使用 Hardtanh 激活函数（限制输出值在 -3 和 3 之间）
    out = tf.clip_by_value(out, -3.0, 3.0)

    # 1x1 卷积将通道数从 3*inc 减少到 outc
    out = tf.keras.layers.Conv2D(outc, (1, 1), use_bias=False)(out)

    out = tf.keras.layers.BatchNormalization()(out)

    out = tf.nn.relu(out)

    return out


def feature_extraction(input_tensor):
    base_model_rgb = Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)
    for layer in base_model_rgb.layers:
        layer.trainable = False
    xception_output = base_model_rgb.output
    # srm_output = srm_conv2d_simple(input_tensor, 3, 3)
    # base_model_srm_input = Input(shape=(299, 299, 3))
    # base_model_srm = Xception(weights='imagenet', include_top=False, input_tensor=srm_output)
    # for layer in base_model_srm.layers:
    #     layer.trainable = False
    # xception_output_srm = base_model_srm.output
    # 对 Xception 提取的特征进行 SRM 处理
    srm_output = srm_conv2d_separate(xception_output, inc=xception_output.shape[-1], outc=2048)
    # freq_output = freqnet(srm_output)
    return xception_output, srm_output


def apply_attention(xception_output, freq_output):
    cbam_output = channel_attention(xception_output, ratio=8)
    cbam_output = spatial_attention(cbam_output)
    cbam_output = Add()([xception_output, cbam_output])
    self_att_output = self_attn(freq_output, in_dim=freq_output.shape[-1])
    return cbam_output, self_att_output


def fuse_features(cbam_output, self_att_output):
    self_att_output_resized = tf.image.resize(self_att_output, (cbam_output.shape[1], cbam_output.shape[2]))
    dual_cma_output_x, dual_cma_output_y = dual_cross_modal_attention(cbam_output, self_att_output_resized, in_dim=cbam_output.shape[-1])
    fused_features = feature_fusion_module(dual_cma_output_x, dual_cma_output_y)
    return fused_features


def build_model():
    input_tensor = Input(shape=(299, 299, 3))

    # 特征提取
    xception_output, freq_output = feature_extraction(input_tensor)

    # 应用注意力机制
    cbam_output, self_att_output = apply_attention(xception_output, freq_output)

    # 融合特征
    fused_features = fuse_features(cbam_output, self_att_output)

    # 全局平均池化
    combined_pool = GlobalAveragePooling2D()(fused_features)

    # Dropout层
    x = Dropout(0.5)(combined_pool)

    # 预测层
    predictions = Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)

    # 构建完整模型
    model = Model(inputs=input_tensor, outputs=predictions)

    return model


# 构建和编译模型
# model = build_model()
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# # 打印模型结构
# model.summary()
