import numpy as np
import tensorflow as tf
import keras.applications
import keras.callbacks
from augmenter import *
from constants import *
from loss_funcs import *
from model import MobileNetv2

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
    with tf.device('/gpu:0'):
        for i in range(0, 180, 10):
            # rotate x_train and y_train[:2] by i degrees
            x_train_aug, y_train_aug = aug_func(x_train, y_train, i)
            x_valid_aug, y_valid_aug = aug_func(x_valid, y_valid, i)
            model.fit(x_train_aug, y_train_aug, validation_data=(x_valid_aug, y_valid_aug), epochs=2, batch_size=32, verbose=2, callbacks=callbacks_list)
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
    




