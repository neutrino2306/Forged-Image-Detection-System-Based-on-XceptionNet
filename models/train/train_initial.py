import tensorflow as tf
import keras as keras
from model_FFT import build_model
from data_preprocessing import get_train_val_generators
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


def initial_train(train_data_path, val_data_path, save_model_path):
    # 创建模型
    model = build_model()

    model.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # for layer in model.layers:
    #     output = layer.output
    #     print(f"Layer {layer.name} output shape: {output.shape}, dtype: {output.dtype}")
    #     if tf.as_dtype(output.dtype) == tf.complex64:
    #         raise ValueError(f"Layer {layer.name} has complex output")

    # 获取数据生成器
    train_generator, validation_generator = get_train_val_generators(train_data_path, val_data_path)

    # 设置模型保存点和早停策略
    checkpoint = ModelCheckpoint(
        filepath=save_model_path.replace('.h5', '_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5'),
        monitor='val_loss',
        save_best_only=False,
        mode='min',
        save_weights_only=False
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')

    # 设置TensorBoard回调
    tensorboard_log_dir = 'D:/Dataset/FF++_dataset_by_ls/jpg/c23/total/my_tensorboard_logs_initial_phase22'
    tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)

    # 训练前三个epoch
    model.fit(train_generator,
              epochs=5,
              batch_size=8,
              validation_data=validation_generator,
              callbacks=[checkpoint, early_stop, tensorboard_callback])

    # 保存模型状态用于后续训练
    model.save(save_model_path)


if __name__ == "__main__":
    initial_train('D:/Dataset/FF++_dataset_by_ls/jpg/c23/total/train',
                  'D:/Dataset/FF++_dataset_by_ls/jpg/c23/total/val',
                  'D:/Dataset/FF++_dataset_by_ls/jpg/c23/total/saved_model_initial_phase22/my_model_initial.h5')
