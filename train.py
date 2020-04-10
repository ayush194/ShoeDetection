import numpy as np
import tensorflow as tf
import keras.applications
import keras.callbacks
from augmenter import *
from constants import *
from loss_funcs import *
from model import MobileNetv2
from threading import Thread

def train(x, y, n_outputs, loss_func, aug_func):
    n_train_dpts = int(split_ratio * 17000)
    # split x and y into train and validation sets
    x_train = x[:n_train_dpts, :]
    y_train = y[:n_train_dpts, :]
    x_valid = x[n_train_dpts:17000, :]
    y_valid = y[n_train_dpts:17000, :]
    # instantiate the pretrained model
    model = MobileNetv2(n_outputs, loss_func)
    # model savename upon callback
    filepath = "saved-model-{epoch:02d}-{val_loss:.2f}.h5"
    # callback to save model when loss reaches runnning minima
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    x_train_curr, y_train_curr = x_train, y_train
    x_valid_curr, y_valid_curr = x_valid, y_valid
    with tf.device('/gpu:0'):
        for i in range(10, 171, 10):
            # rotate x_train and y_train[:2] by i degrees
            x_train_next, y_train_next = np.zeros(x_train.shape, dtype=np.float32), np.zeros(y_train.shape, dtype=np.float32)
            x_valid_next, y_valid_next = np.zeros(x_valid.shape, dtype=np.float32), np.zeros(y_valid.shape, dtype=np.float32)
            data_aug_thread1 = Thread(target=aug_func, args=(x_train, y_train, x_train_next, y_train_next, i+10))
            data_aug_thread2 = Thread(target=aug_func, args=(x_valid, y_valid, x_valid_next, y_valid_next, i+10))
            data_aug_thread1.start()
            data_aug_thread2.start()
            # x_train_aug, y_train_aug = augment(x_train, y_train, i)
            # x_valid_aug, y_valid_aug = augment(x_valid, y_valid, i)
            model.fit(x_train_curr, y_train_curr, validation_data=(x_valid_curr, y_valid_curr), epochs=5, batch_size=64, verbose=2, callbacks=callbacks_list)
            data_aug_thread1.join()
            data_aug_thread2.join()
            x_train_curr, y_train_curr = x_train_next, y_train_next
            x_valid_curr, y_valid_curr = x_valid_next, y_valid_next
    model.save("bb.h5")
    valid_score = model.evaluate(x_valid, y_valid, verbose=0)
    print('Validation loss:', valid_score)


if __name__ == "__main__":
    # for regressing bounding boxes
    x = np.load("data/x.npy")
    y = np.load("data/y.npy")
    train(x, y, 10, myLoss, augment)

    # for regressing keypoints inside the bounding box
    # x_cropped = np.load("data/x_cropped.npy")
    # y_keypts = np.load("data/y_keypts.npy")
    # train(x_cropped, y_keypts[:, 2:], 50, 'mean_squared_error', augment2)
    




