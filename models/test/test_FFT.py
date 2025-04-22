import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from FFT_SRM import build_model
from keras.metrics import Precision, Recall, AUC
from keras.optimizers import Adam

# 定义计算等错误率（EER）的函数
def calculate_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_threshold = interp1d(fpr, thresholds)(eer)
    return eer, eer_threshold


# 加载保存的模型权重
model = build_model()
model.load_weights('C:/Users/cissy/Desktop/my_model_weights_initial_epoch_05_val_loss_0.64.h5')  # 更新为保存权重的正确路径
model.compile(optimizer=Adam(learning_rate=0.00002, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
                  loss='binary_crossentropy',
                  metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')])

# 定义测试数据生成器
test_datagen = ImageDataGenerator(rescale=1. / 255)

# test_data_path = ('D:/Dataset/FF++_dataset_by_ls/jpg/c23/total/test')  # 调整为你的测试数据路径
# test_data_path = ('D:/Dataset/celebaV2_test')
test_data_path = ('DC:/Users/cissy/Desktop/final/test')

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

# 计算EER
eer, eer_threshold = calculate_eer(test_generator.classes, predictions)
print(f"EER: {eer:.3f} at threshold: {eer_threshold:.3f}")

# 性能报告
print(classification_report(true_classes, predicted_classes, target_names=test_generator.class_indices.keys()))

# 混淆矩阵
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print('Confusion Matrix:')
print(conf_matrix)
