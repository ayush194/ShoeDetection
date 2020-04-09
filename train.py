import numpy as np
import tensorflow as tf
import keras.applications
import keras.callbacks
from augmenter import augment
from constants import *
from loss_funcs import *
from model import MobileNetv2

def train():
    x = np.load("x.npy")
    y = np.load("y.npy")
    n_train_dpts = int(split_ratio * 17000)
    x_train = x[:n_train_dpts, :]
    y_train = y[:n_train_dpts, :]
    x_valid = x[n_train_dpts:17000, :]
    y_valid = y[n_train_dpts:17000, :]
    model = MobileNetv2(10, myLoss)
    filepath = "saved-model-{epoch:02d}-{val_loss:.2f}.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    with tf.device('/gpu:0'):
        for i in range(0, 180, 10):
            # rotate x_train and y_train[:2] by i degrees
            x_train_aug, y_train_aug = augment(x_train, y_train, i)
            x_valid_aug, y_valid_aug = augment(x_valid, y_valid, i)
            model.fit(x_train_aug, y_train_aug, validation_data=(x_valid_aug, y_valid_aug), epochs=2, batch_size=8, verbose=2, callbacks=callbacks_list)
    model.save("bb.h5")
    valid_score = model.evaluate(x_valid, y_valid, verbose=0)
    print('Validation loss:', valid_score)




