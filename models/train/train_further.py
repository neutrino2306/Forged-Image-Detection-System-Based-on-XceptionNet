import keras
from model import build_model
from data_preprocessing import get_train_val_generators
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


def further_train(train_data_path, val_data_path, save_model_path):

    # 加载之前保存的模型状态
    model = keras.models.load_model('D:/Dataset/my_tensorboard_logs_initial/my_model_initial.h5')

    # 在训练三个epoch后，解冻所有层，进行微调
    for layer in model.layers:
        layer.trainable = True

    # 重新编译模型
    model.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    train_generator, validation_generator = get_train_val_generators(train_data_path, val_data_path)

    checkpoint = ModelCheckpoint(
        filepath=save_model_path.replace('.h5', '_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5'),
        monitor='val_loss',
        save_best_only=False,
        mode='min',
        save_weights_only=False
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    tensorboard_log_dir = 'D:/Dataset/FF++_dataset_by_ls/jpg/c23/total/my_tensorboard_logs_further'
    tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)

    # 训练后十五个epoch
    model.fit(train_generator,
              epochs=15,
              batch_size=32,
              validation_data=validation_generator,
              callbacks=[checkpoint, early_stop, tensorboard_callback])

    # 保存最终模型状态
    model.save(save_model_path)


if __name__ == "__main__":
    further_train('D:/Dataset/FF++_dataset_by_ls/jpg/c23/total/train', 'D:/Dataset/FF++_dataset_by_ls/jpg/c23/total/val', 'D:/Dataset/FF++_dataset_by_ls/jpg/c23/total/saved_model_further/my_model_further.h5')
