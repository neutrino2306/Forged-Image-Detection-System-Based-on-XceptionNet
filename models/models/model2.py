import math
import tensorflow as tf
import numpy as np
from keras import Input, Model
from keras.layers import Conv2D, Conv2DTranspose, GlobalAveragePooling2D, GlobalMaxPooling2D, ReLU, Activation, Add, \
    Concatenate, BatchNormalization
from keras.src.layers import Reshape, Dense, Softmax, Permute, Multiply, Lambda
import pywt
import tfwavelets as tfw
from tfwavelets.dwtcoeffs import Filter, Wavelet
from tfwavelets.nodes import dwt2d

# def reorder_coefficients(src_weight, N=8):
#     array_size = N * N
#     # order_index = np.zeros((N, N))
#     i = 0
#     j = 0
#     rearrange_weigth = src_weight.copy()  # (N*N) * N * N
#     for k in range(array_size - 1):
#         if (i == 0 or i == N - 1) and j % 2 == 0:
#             j += 1
#         elif (j == 0 or j == N - 1) and i % 2 == 1:
#             i += 1
#         elif (i + j) % 2 == 1:
#             i += 1
#             j -= 1
#         elif (i + j) % 2 == 0:
#             i -= 1
#             j += 1
#         index = i * N + j
#         rearrange_weigth[k + 1, ...] = src_weight[index, ...]
#     return rearrange_weigth
#
#
# def create_dct_kernel(N=8, rearrange=True):
#     dct_weight = np.zeros((N * N, N, N))
#     for k in range(N * N):
#         u = k // N
#         v = k % N
#         for i in range(N):
#             for j in range(N):
#                 tmp1 = cos_1d(i, u, N)
#                 tmp2 = cos_1d(j, v, N)
#                 coeff = tmp1 * tmp2 * scale_factor(u, N) * scale_factor(v, N)
#                 dct_weight[k, i, j] += coeff
#     if rearrange:
#         dct_weight = reorder_coefficients(dct_weight, N)
#     return tf.convert_to_tensor(dct_weight, dtype=tf.float32)
#
#
# def cos_1d(ij, uv, N=8):
#     return math.cos(math.pi * uv * (ij + 0.5) / N)
#
#
# def scale_factor(u, N=8):
#     if u == 0:
#         return math.sqrt(1 / N)
#     else:
#         return math.sqrt(2 / N)
#
#
# def rgb_to_ycbcr(image):
#     # 定义RGB到YCbCr的转换矩阵
#     trans_matrix = np.array([[0.299, 0.587, 0.114],
#                              [-0.169, -0.331, 0.5],
#                              [0.5, -0.419, -0.081]], dtype=np.float32).reshape((1, 1, 3, 3))
#     # 转换为TensorFlow tensor
#     trans_matrix = tf.constant(trans_matrix)
#
#     # 创建一个Conv2D层，权重为转换矩阵，不带偏置
#     ycbcr_image = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1),
#                          padding='same', use_bias=False,
#                          kernel_initializer=tf.constant_initializer(trans_matrix),
#                          input_shape=image.shape[1:])(image)
#
#     return ycbcr_image
#
#
# def ycbcr_to_rgb(image):
#     # 定义YCbCr到RGB的逆转换矩阵
#     re_matrix = np.linalg.pinv(np.array([[0.299, 0.587, 0.114],
#                                          [-0.169, -0.331, 0.5],
#                                          [0.5, -0.419, -0.081]], dtype=np.float32)).reshape((1, 1, 3, 3))
#     # 转换为TensorFlow tensor
#     re_matrix = tf.constant(re_matrix)
#
#     # 创建一个Conv2D层，权重为逆转换矩阵，不带偏置
#     rgb_output = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1),
#                         padding='same', use_bias=False,
#                         kernel_initializer=tf.constant_initializer(re_matrix),
#                         input_shape=image.shape[1:])(image)
#     return rgb_output
#
#
# def forward_transform(x, N=8, in_channel=3, rearrange=False):
#     # 首先，执行RGB到YCbCr的颜色空间转换
#     ycbcr_image = rgb_to_ycbcr(x)
#
#     # 创建DCT核心权重
#     dct_weights = create_dct_kernel(N=N, rearrange=rearrange)
#
#     # 将DCT核心权重格式调整为卷积层所需的格式
#     # dct_weights = dct_weights.view(N * N * in_channel, 1, N, N).repeat(1, in_channel, 1, 1)
#
#     # 将DCT核心权重格式调整为卷积层所需的格式（调整维度以符合TF的卷积格式）
#     dct_weights = tf.reshape(dct_weights, (N, N, 1, N * N))
#     dct_weights = tf.tile(dct_weights, [1, 1, in_channel, 1])
#
#     # 应用DCT变换
#     # dct_image = F.conv2d(ycbcr_image, dct_weights, stride=N, padding=0, groups=in_channel)
#
#     dct_layer = Conv2D(filters=N * N * in_channel, kernel_size=(N, N), strides=(N, N),
#                        padding='valid', use_bias=False, trainable=False)
#     dct_layer.build(x.shape)
#     dct_layer.set_weights([dct_weights])
#
#     dct_image = dct_layer(ycbcr_image)
#     return ycbcr_image, dct_image
#
#
# def reverse_transform(dct_image, N=8, in_channel=3, rearrange=False):
#     # 创建逆DCT核心权重，注意逆变换的权重与正变换相同
#     dct_weights = create_dct_kernel(N=N, rearrange=rearrange)
#
#     # 将DCT核心权重格式调整为适合逆卷积的格式
#     dct_weights = tf.reshape(dct_weights, (N, N, 1, N * N))
#     dct_weights = tf.tile(dct_weights, [1, 1, in_channel, 1])
#
#     # 应用逆DCT变换
#     output_shape = [dct_image.shape[0], dct_image.shape[1] * N, dct_image.shape[2] * N, in_channel]
#     reverse_dct_layer = Conv2DTranspose(filters=in_channel, kernel_size=(N, N), strides=(N, N),
#                                         padding='valid', output_padding=0, use_bias=False, trainable=False)
#     reverse_dct_layer.build(dct_image.shape)
#     reverse_dct_layer.set_weights([dct_weights])
#
#     ycbcr_image = reverse_dct_layer(dct_image)
#     # 执行YCbCr到RGB的颜色空间逆转换
#     rgb_image = ycbcr_to_rgb(ycbcr_image)
#
#     return rgb_image, ycbcr_image
#
#
# def channel_attention(inputs, in_planes, ratio=8):
#     # 使用 Keras 的全局平均池化和全局最大池化
#     avg_pool = GlobalAveragePooling2D(keepdims=True)(inputs)
#     max_pool = GlobalMaxPooling2D(keepdims=True)(inputs)
#
#     # 创建共享的 MLP (使用两个 Conv2D 层，中间接 ReLU 激活)
#     # 第一个卷积层：降维
#     mlp_reduce = Conv2D(filters=in_planes // ratio, kernel_size=1, use_bias=False,
#                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))
#     # ReLU 激活层
#     relu = ReLU()
#     # 第二个卷积层：升维
#     mlp_expand = Conv2D(filters=in_planes, kernel_size=1, use_bias=False,
#                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))
#
#     # 对平均池化结果进行 MLP 处理
#     avg_out = mlp_expand(relu(mlp_reduce(avg_pool)))
#     # 对最大池化结果进行 MLP 处理
#     max_out = mlp_expand(relu(mlp_reduce(max_pool)))
#
#     # 将两个输出相加并通过 Sigmoid 激活函数
#     sigmoid = Activation('sigmoid')
#     attention_output = sigmoid(avg_out + max_out)
#
#     return attention_output
#
#
# def dual_cross_modal_attention(x, y, in_dim, size, ratio=8, ret_att=False):
#     key_conv1 = Conv2D(in_dim // ratio, kernel_size=1, use_bias=False)
#     key_conv2 = Conv2D(in_dim // ratio, kernel_size=1, use_bias=False)
#     key_conv_share = Conv2D(in_dim // ratio, kernel_size=1, use_bias=False)
#
#     value_conv1 = Conv2D(in_dim, kernel_size=1, use_bias=False)
#     value_conv2 = Conv2D(in_dim, kernel_size=1, use_bias=False)
#
#     proj_key1 = key_conv_share(key_conv1(x))
#     proj_key2 = key_conv_share(key_conv2(y))
#     proj_key1 = Reshape((size * size, in_dim // ratio))(proj_key1)
#     proj_key2 = Reshape((size * size, in_dim // ratio))(proj_key2)
#     proj_key1 = Permute((2, 1))(proj_key1)
#
#     proj_value_y_on_x = value_conv2(y)
#     proj_value_x_on_y = value_conv1(x)
#
#     linear1 = Dense(size * size, use_bias=False, activation='linear')
#     linear2 = Dense(size * size, use_bias=False, activation='linear')
#
#     energy = tf.matmul(proj_key1, proj_key2)
#
#     # Keras 中，Dense 层期望输入是 [batch_size, features]，这里的 features 应对应于一维特征
#     flatten = Reshape((size * size,))
#     softmax = Softmax(axis=-1)
#
#     attention1 = softmax(linear1(flatten(energy)))
#     attention2 = softmax(linear2(flatten(Permute((2, 1))(energy))))
#
#     attention1 = Reshape((size, size, 1))(attention1)
#     attention2 = Reshape((size, size, 1))(attention2)
#
#     out_y_on_x = Multiply()([proj_value_y_on_x, attention1])
#     out_x_on_y = Multiply()([proj_value_x_on_y, attention2])
#
#     gamma1 = tf.Variable(initial_value=tf.zeros(1), trainable=True)
#     gamma2 = tf.Variable(initial_value=tf.zeros(1), trainable=True)
#
#     out_x = Add()([x, gamma1 * out_y_on_x])
#     out_y = Add()([y, gamma2 * out_x_on_y])
#
#     if ret_att:
#         return out_x, out_y, attention1, attention2
#
#     return out_x, out_y
#
#
# def feature_fusion_function(x, y, out_channels):
#     """
#     特征融合函数，结合通道注意力机制。
#     参数:
#     x, y: 输入特征张量。
#     in_channels: 输入通道数。
#     out_channels: 输出通道数。
#     channel_attention_func: 通道注意力函数，已经定义。
#     """
#     # 使用 Concatenate 层在通道维度上合并 x 和 y
#     fused_input = Concatenate(axis=-1)([x, y])
#
#     # 定义卷积块：1x1卷积，批量归一化，ReLU激活
#     conv = Conv2D(out_channels, (1, 1), strides=(1, 1), padding='valid', use_bias=False)(fused_input)
#     bn = BatchNormalization()(conv)
#     relu = ReLU()(bn)
#
#     # 应用通道注意力
#     ca_features = channel_attention(relu, out_channels, ratio=16)
#
#     # 融合加权特征：原始融合特征 + 通道注意力加权的特征
#     enhanced_features = Add()([relu, relu * ca_features])
#
#     return enhanced_features
#
#
# # 局部高频增强分支
# ycbcr1, dct_x = self.dct(x)  # YCBCR:(8,3,224,224)   dct_x:(8,192,28,28)
# ca1 = self.ca1(dct_x)
# dct_choose = dct_x * ca1
# dct_choose = self.conv_dct(dct_choose)  # (8,192,28,28)
# dctx, ycbcr2 = self.idct(dct_choose)
#
# dctx1 = self.conv_dctx_1(dctx)  # （8,32，112,112）
# dctx2 = self.conv_dctx_2(dctx1)  # （8,64，56,56）
# dctx3 = self.conv_dctx_3(dctx2)  # （8,256，28,28）
# dctx4 = self.conv_dctx_4(dctx3)  # （8,728，14,14）

# self.dct(x)在应用forward_transform
# self.ca1(dct_x)在应用channel_attention
# channels = 192
# self.ca1 = ChannelAttention(channels)
# self.conv_dct = nn.Sequential(
#     nn.Conv2d(192, 32, kernel_size=3, stride=1, padding=1),
#     nn.BatchNorm2d(32),
#     nn.ReLU(True),
#
#     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#     nn.BatchNorm2d(64),
#     nn.ReLU(True),
#
#     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#     nn.BatchNorm2d(128),
#     nn.ReLU(True),
#
#     nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
#     nn.BatchNorm2d(192),
#     nn.ReLU(True),
# )
# self.idct(dct_choose)在应用reverse_transform
#         self.conv_dctx_1 = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 112
#             nn.BatchNorm2d(32),
#             nn.ReLU(True),
#
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 112
#             nn.BatchNorm2d(32),
#             nn.ReLU(True),
#         )
#         self.conv_dctx_2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 56
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 56
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#         )
#         self.conv_dctx_3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 28
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#
#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 28
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 28
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#         )
#         self.conv_dctx_4 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 14
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#
#             nn.Conv2d(512, 728, kernel_size=3, stride=1, padding=1),  # 14
#             nn.BatchNorm2d(728),
#             nn.ReLU(True),
#         )

# XceptionNet的处理
# x = self.xception.fea_part1(x)
# x = self.xception.block1(x)
# x = self.xception.block2(x)
# x = self.xception.block3(x)
# x = self.xception.fea_part3(x)
#
# # 特征融合
# fusion1 = self.dual_cma0(x, dctx4)
# f1 = self.fusion0(fusion1[0], fusion1[1])
# fusion2 = self.dual_cma1(x, xh)
# f2 = self.fusion1(fusion2[0], fusion2[1])
# f3 = torch.cat((f1, f2), dim=1)
# f = self.fusion2(f3)
# f = self.xception.fea_part4(f)
# f = self.xception.fea_part5(f)
# x = nn.ReLU(inplace=True)(x)
# x = F.adaptive_avg_pool2d(x, (1, 1))
# x = x.view(x.size(0), -1)
# x = self.xception.last_linear(x)


# fusion0和fusion1都是feature_fusion_function
# self.fusion0 = FeatureFusionModule(in_chan=728 * 2, out_chan=728)
# dual_cma0和dual_cma1都是DualCrossModalAttention
# self.dual_cma0 = DualCrossModalAttention(in_dim=728, ret_att=False)
# self.xception.last_linear = nn.Linear(2048, num_classes)
# self.fusion2=nn.Sequential(
#             nn.Conv2d(728*2, 728, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(728),
#             nn.ReLU(True),
#
#             nn.Conv2d(728, 728, kernel_size=3, stride=1, padding=1),  # 112
#             nn.BatchNorm2d(728),
#             nn.ReLU(True),
#         )


# 全局高频增强分支
# 小波变换与还原，从xfm1到xfm4都是一样的，J=1
# self.xfm1 = DWTForward(J=J, mode='zero', wave='haar')
# self.ifm1 = DWTInverse(mode='zero', wave='haar')

# 在第一阶段，这几个层用于处理高频的三个分量在特征融合后的再提取
# 在二、三、四阶段，这几个层（conv_x2、conv_x3、conv_x4）用于处理前一个阶段的低频特征中在小波变换后的三个高频分量特征融合后的总特征提取
# self.conv_x1 = nn.Sequential(
#             nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
#             nn.BatchNorm2d(9),
#             nn.ReLU(True),
#
#             nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(9),
#             nn.ReLU(True),
#
#             nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(9),
#             nn.ReLU(True),
#
#             nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1, groups=3),
#             nn.BatchNorm2d(9),
#             nn.ReLU(True),
#         )

# 第一阶段
# 小波变换：对输入图像 x 进行第一次小波变换，分解为低频分量 Yl_1 和高频分量 Yh_1。
# 高频分量处理：将 Yh_1 中的不同方向分量分离、重组并进行卷积处理，得到 x1。
# 合并与卷积：将处理后的结果 x1 按通道合并，并通过卷积层进一步加工，以增强特征表示。
# 特征拆分：在通道合并然后进行卷积进一步加工后，再将高频信息x1拆分成三个高频信息的维度

# 第二、三、四每个阶段的详细操作：
# 小波变换提取高频分量：对前一阶段的每个方向的高频输出进行小波变换。例如，在第二阶段，分别对 x1_1, x1_2, x1_3 进行处理。
# 每次变换都会得到新的高频（yh_2, yh_3, yh_4）和低频（yl_2, yl_3, yl_4）分量。
# 合并相同方向的高频分量并进行卷积处理：将同一方向上的高频分量（来自不同处理分支）合并，例如 yh_2[0][:, :, 0, :, :], yh_3[0][:, :, 0, :, :], yh_4[0][:, :, 0, :, :] 合并为 x1_2_1。
# 对合并后的高频分量通过卷积层如 self.conv_x1_2_1 进行处理，进一步提取特征。
# 小波变换处理前一阶段低频输出：对前一阶段的低频输出，如 Yl_1，进行小波变换，得到新一轮的低频 Yl_2 和高频 Yh_2。
# 提取新的高频分量，并调整形状以符合后续处理需求，如 Yh_2[0][:, :, 0, :, :] 调整为 x2_1。
# 合并新的高频分量并进行卷积处理：将提取的新高频分量合并，如 x2_1, x2_2, x2_3，并通过卷积层 self.conv_x2 进行处理，以增强特征表达。
# 特征分割与融合：分割合并后的结果，如 x2 分割为 x2_1, x2_2, x2_3，每部分包含合并后的不同方向特征。
# 将每部分与前面阶段的相应处理结果（x1_2_1, x1_2_2, x1_2_3）进行融合，以增强特征复杂性和有效性。
# 对融合后的特征再次应用卷积处理：对融合后的特征通过卷积层如 self.conv_x2_1, self.conv_x2_2, self.conv_x2_3 进行处理，以进一步提炼和加工这些特征，提高模型的识别能力。

# x1的都是这样
# self.conv_x1_2_1 = nn.Sequential(
#             nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
#             nn.BatchNorm2d(9),
#             nn.ReLU(True),
#
#             nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(9),
#             nn.ReLU(True),
#
#             nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(9),
#             nn.ReLU(True),
#
#             nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(9),
#             nn.ReLU(True),
#         )
# x2及以后的都是这样
# self.conv_x2_1 = nn.Sequential(
#             nn.Conv2d(12, 9, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(9),
#             nn.ReLU(True),
#
#             nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(9),
#             nn.ReLU(True),
#
#             nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(9),
#             nn.ReLU(True),
#
#             nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(3),
#             nn.ReLU(True),
#         )
# def dwt_transform(x, wavelet='haar', mode='zero'):
#     # 用于小波变换的函数，接受一个 TensorFlow 张量 x
#     def _dwt(x_np):
#         coeffs = pywt.dwt2(x_np, wavelet, mode)
#         cA, (cH, cV, cD) = coeffs
#         return np.stack([cA, cH, cV, cD], axis=-1)  # 将各分量合并回一个numpy数组
#
#     # 使用 tf.numpy_function 封装使其可以在 TensorFlow 图中使用
#     result = tf.numpy_function(_dwt, [x], tf.float32)
#     # 确保输出形状设置正确，特别是当使用 numpy_function 时
#     result.set_shape((None, (x.shape[1] + 1) // 2, (x.shape[2] + 1) // 2, 4))
#
#     return result
# 定义 Haar 小波
def create_haar_wavelet():
    sqrt2_inv = 1 / np.sqrt(2)

    # 创建一维 NumPy 数组系数
    coeffs_lp = np.array([sqrt2_inv, sqrt2_inv], dtype=np.float32)
    coeffs_hp = np.array([sqrt2_inv, -sqrt2_inv], dtype=np.float32)

    # 使用这些一维数组来初始化 Filter 对象
    decomp_lp = Filter(coeffs_lp, zero=0)
    decomp_hp = Filter(coeffs_hp, zero=1)
    recon_lp = Filter(coeffs_lp, zero=0)  # 对于 Haar 小波，重构和分解滤波器相同
    recon_hp = Filter(coeffs_hp, zero=1)

    return Wavelet(decomp_lp, decomp_hp, recon_lp, recon_hp)

haar_wavelet = create_haar_wavelet()

def build_model():
    # 定义输入层，指定输入形状
    inputs = Input(shape=(299, 299, 3))
    reshaped_inputs = tf.ensure_shape(inputs, (16, 299, 299, 3))  # 明确批量大小
    Yl, Yh = tf.keras.layers.Lambda(lambda x: dwt2d(x, wavelet=haar_wavelet, levels=1))(reshaped_inputs)
    # 封装 DWT 操作
    # Yl, Yh = tf.keras.layers.Lambda(lambda x: dwt2d(x, wavelet=haar_wavelet, levels=1))(inputs)

    # 从 Yh 中选择高频系数
    x1_1 = tf.keras.layers.Lambda(lambda x: x[:, :, :, :, 0])(Yh)
    x1_2 = tf.keras.layers.Lambda(lambda x: x[:, :, :, :, 1])(Yh)
    x1_3 = tf.keras.layers.Lambda(lambda x: x[:, :, :, :, 2])(Yh)

    # 提取并重构三个方向的高频分量
    # x1_1 = Yh_1[0][:, :, :, 0]  # 取第一个高频分量的第一个方向
    # x1_2 = Yh_1[0][:, :, :, 1]  # 取第一个高频分量的第二个方向
    # x1_3 = Yh_1[0][:, :, :, 2]  # 取第一个高频分量的第三个方向

    # 合并高频分量
    # x1 = Concatenate(axis=-1)([x1_1, x1_2, x1_3])
    #
    # # 通过卷积层进一步处理合并后的高频分量
    # x1 = Conv2D(64, (3, 3), padding='same')(x1)
    # x1 = BatchNormalization()(x1)
    # x1 = ReLU()(x1)

    # 创建模型
    model = tf.keras.Model(inputs=inputs, outputs=[x1_1, x1_2, x1_3])
    return model


# 创建模型
model = build_model()
model.summary()

# 测试模型
input_tensor = tf.random.normal([1, 299, 299, 3])
outputs = model(input_tensor)
print([o.shape for o in outputs])
