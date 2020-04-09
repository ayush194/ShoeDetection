import keras.layers
import keras.applications
from constants import *

def MobileNetv2(n_outputs, loss_func):
    model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=(img_size_red, img_size_red, 3), 
                                                        alpha=1.0, include_top=False, weights='imagenet', 
                                                        input_tensor=None, pooling=None, classes=100)
    output = keras.layers.Flatten()(model.output)
    output = keras.layers.Dense(100, activation='relu')(output)
    output = keras.layers.Dropout(0.5)(output)
    output = keras.layers.BatchNormalization()(output)
    output = keras.layers.Dense(n_outputs, activation='relu')(output)
    model = keras.models.Model(model.input, output)
    model.compile(loss=loss_func, optimizer='adam', metrics=['mse'])
    return model