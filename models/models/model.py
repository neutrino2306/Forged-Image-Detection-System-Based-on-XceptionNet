from keras import Input
from keras.applications import Xception
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Flatten, Concatenate, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2


# 第一版[88]模型
def build_model():
    # 加载预训练的Xception模型
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(299, 299, 3)
    )

    # 在前三个epoch中，冻结预训练模型的权重
    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)  # 添加全局平均池化层，对每个特征图进行池化

    predictions = Dense(1, activation='sigmoid')(x)  # 添加全连接层，单神经元输出+sigmoid激活函数

    # 添加Dropout层，以减少过拟合
    x = Dropout(0.2)(x)

    # 添加全连接层，使用L2正则化
    predictions = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001))(x)

    model = Model(inputs=base_model.input, outputs=predictions)  # 构建完整模型

    adam = Adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-8)  # 设置Adam优化器

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])  # 编译模型

    return model

def build_model():

    # 加载预训练的Xception模型，不包括顶部的全连接层
    base_model = Xception(include_top=False, weights='imagenet')

    # 在前三个epoch中，冻结预训练模型的权重
    for layer in base_model.layers:
        layer.trainable = False

    # 获取模型的输出
    x = base_model.output

    # 添加MaxPooling2D层来减小特征图的尺寸
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # 添加GlobalMaxPooling2D层来提取特征
    global_max_pool = GlobalMaxPooling2D()(x)

    # 添加GlobalAveragePooling2D层来提取特征
    global_avg_pool = GlobalAveragePooling2D()(x)

    # 使用Flatten层将多维特征图转换为一维向量
    flat = Flatten()(x)

    # 使用Concatenate层合并全局池化层的输出和扁平化层的输出
    concatenated_features = Concatenate()([global_max_pool, global_avg_pool, flat])

    # 添加Dropout层减少过拟合，比例改为0.2
    concatenated_features = Dropout(0.2)(concatenated_features)

    # 添加Dense层进行分类，使用L2正则化
    predictions = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))(concatenated_features)

    # 构建最终的模型
    model = Model(inputs=base_model.input, outputs=predictions)

    # 设置Adam优化器
    adam = Adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    # 编译模型
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    return model




