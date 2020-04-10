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
    output1 = keras.layers.Dense(2, activation='softmax')(output)
    output2 = keras.layers.Dense(8, activation='relu')(output)
    output = keras.layers.concatenate([output1, output2])
    model = keras.models.Model(model.input, outputs=output)
    return model