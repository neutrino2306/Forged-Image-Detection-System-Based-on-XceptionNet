from keras import backend as K, Input, regularizers
from keras.applications import Xception
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Flatten, Concatenate, Dropout, \
    BatchNormalization, Activation, Reshape, Add, Multiply, Lambda, Conv2D
from keras.optimizers import Adam
from keras.regularizers import l1_l2


# 反映了像素在所有通道上的平均强度，对每个像素位置的所有通道值求平均
def mean(x):
    return K.mean(x, axis=3, keepdims=True)


# 对每个像素位置的所有通道值求最大值，帮助模型在每个像素级别上识别颜色强度最大的特征
def max(x):
    return K.max(x, axis=3, keepdims=True)


def channel_attention(input_feature, ratio=8):
    channel_axis = -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

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

    # 原图的每一个像素在某一个通道的值都与通道注意力的对应通道的值相乘
    return Multiply()([input_feature, cbam_feature])


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
    return Multiply()([input_feature, cbam_feature])


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

    # 特征提取层
    x = base_model.output
    original_x = x

    # CBAM模块
    x = channel_attention(x, ratio=8)
    x = spatial_attention(x)

    # 在CBAM后添加跳层连接
    x = Add()([x, original_x])

    # 全局平均池化层
    x = GlobalAveragePooling2D()(x)

    # Dropout层
    x = Dropout(0.5)(x)

    # 预测层
    predictions = Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)

    # 构建完整模型
    model = Model(inputs=input_tensor, outputs=predictions)

    return model
