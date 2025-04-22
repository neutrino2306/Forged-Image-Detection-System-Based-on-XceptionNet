import numpy as np
import tensorflow as tf
import keras as keras
from keras.metrics import Precision, Recall, AUC
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from keras import backend as K


# 加载模型前确保自定义函数被正确引用
def custom_mean(x):
    return K.mean(x, axis=3, keepdims=True)

def custom_max(x):
    return K.max(x, axis=3, keepdims=True)

# 计算EER的函数
def calculate_eer(y_true, y_scores):
    # 计算ROC曲线的FPR, TPR和阈值
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # 计算FAR为1-TPR，FRR为FPR
    far = 1 - tpr
    frr = fpr
    # 寻找EER：最小的点，此时FAR和FRR最为接近
    abs_diffs = abs(far - frr)
    min_index = np.argmin(abs_diffs)
    eer = (far[min_index] + frr[min_index]) / 2
    return eer, thresholds[min_index]

# 确保在加载模型时，将这些自定义函数传递给custom_objects参数
model_path = ('C:/Users/cissy/Desktop/models/my_model_further_epoch_04_val_accuracy_0.90.h5')
model = load_model(model_path, custom_objects={'K': K, 'custom_mean': custom_mean, 'custom_max': custom_max})

# 测试数据预处理
test_datagen = ImageDataGenerator(rescale=1. / 255)

# test_data_path = ('D:/Dataset/FF++_dataset_by_ls/jpg/c23/face2face/test')  # 调整为你的测试数据路径
# test_data_path = ('D:/Dataset/celebaV2_test')
test_data_path = ('C:/Users/cissy/Desktop/final/test')

test_generator = test_datagen.flow_from_directory(
    test_data_path,
    target_size=(299, 299),  # 与训练时相同的图片尺寸
    batch_size=32,
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

# 计算EER
eer, eer_threshold = calculate_eer(test_generator.classes, predictions)
print(f"EER: {eer:.3f} at threshold: {eer_threshold:.3f}")


# 性能报告
print(classification_report(true_classes, predicted_classes, target_names=test_generator.class_indices.keys()))

# 混淆矩阵
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print('Confusion Matrix:')
print(conf_matrix)
