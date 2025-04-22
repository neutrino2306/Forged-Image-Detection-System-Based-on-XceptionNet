import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


def random_color_adjustment(image):
    image = tf.image.random_hue(image, 0.08)  # 随机调整色调（hue）
    image = tf.image.random_saturation(image, 0.5, 1.5)  # 随机调整饱和度（saturation）
    image = tf.image.random_brightness(image, 0.05)  # 随机调整亮度（brightness）
    image = tf.image.random_contrast(image, 0.8, 1.2)  # 随机调整对比度（contrast）
    return image


def get_train_val_generators(train_data_path, val_data_path):
    train_data = ImageDataGenerator(
        rescale=1. / 255,
        preprocessing_function=random_color_adjustment,  # 使用自定义的预处理函数
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_data = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_data.flow_from_directory(
        train_data_path,
        target_size=(299, 299),
        batch_size=8,
        class_mode='binary'
    )

    validation_generator = val_data.flow_from_directory(
        val_data_path,
        target_size=(299, 299),
        batch_size=8,
        class_mode='binary'
    )

    return train_generator, validation_generator
