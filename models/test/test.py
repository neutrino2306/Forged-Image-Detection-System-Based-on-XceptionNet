import tensorflow as tf
import keras as keras
from keras.metrics import Precision, Recall, AUC
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from keras import backend as K


# 加载模型前确保自定义函数被正确引用
def custom_mean(x):
    return K.mean(x, axis=3, keepdims=True)

def custom_max(x):
    return K.max(x, axis=3, keepdims=True)

# 确保在加载模型时，将这些自定义函数传递给custom_objects参数
model_path = ('D:/Dataset/FF++_dataset_by_ls/jpg/c23/total/saved_model_further_phase6/my_model_further_epoch_04_val_loss_0.17.h5')
model = load_model(model_path, custom_objects={'K': K, 'custom_mean': custom_mean, 'custom_max': custom_max})

# 归一化操作，将图像像素值从[0, 255]缩放到[0, 1]
test_datagen = ImageDataGenerator(rescale=1. / 255)

# test_data_path = ('D:/Dataset/FF++_dataset_by_ls/jpg/c23/total/test')  # 调整为你的测试数据路径
test_data_path = ('D:/Dataset/celebaV2_test')

test_generator = test_datagen.flow_from_directory(
    test_data_path,
    target_size=(299, 299),  # 与训练时相同的图片尺寸
    batch_size=16,
    class_mode='binary',
    shuffle=False)

# 测试模型性能
test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")
print(f"Test AUC: {test_auc}")

# 预测
predictions = model.predict(test_generator)
predicted_classes = predictions > 0.5  # 二分类问题的阈值设置

# 真实标签
true_classes = test_generator.classes

# 性能报告
print(classification_report(true_classes, predicted_classes, target_names=test_generator.class_indices.keys()))

# 混淆矩阵
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print('Confusion Matrix:')
print(conf_matrix)
