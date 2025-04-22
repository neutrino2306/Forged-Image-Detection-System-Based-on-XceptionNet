from model import build_model
from data_preprocessing import get_train_val_generators
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


def train_model(train_data_path, val_data_path, save_model_path):
    # 创建模型
    model = build_model()

    # 获取数据生成器
    train_generator, validation_generator = get_train_val_generators(train_data_path, val_data_path)

    # 设置模型保存点和早停策略
    checkpoint = ModelCheckpoint(
        filepath=save_model_path.replace('final.h5', 'epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5'),
        monitor='val_loss',
        save_best_only=False,
        mode='min',
        save_weights_only=False
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')

    # 设置TensorBoard回调
    tensorboard_log_dir = 'D:/Dataset/my_tensorboard_logs'
    tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)

    # 训练前三个epoch
    model.fit(train_generator,
              epochs=3,
              validation_data=validation_generator,
              callbacks=[checkpoint, early_stop, tensorboard_callback])

    # 在训练三个epoch后，解冻所有层，进行微调
    for layer in model.layers:
        layer.trainable = True

    # 重新编译模型
    model.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 训练后十五个epoch
    model.fit(train_generator,
              epochs=15,
              validation_data=validation_generator,
              callbacks=[checkpoint, early_stop, tensorboard_callback])

    # 保存最终模型
    model.save(save_model_path)


if __name__ == "__main__":
    train_model('path_to_train_data', 'path_to_val_data', 'path_to_save_model/my_model_final.h5')
