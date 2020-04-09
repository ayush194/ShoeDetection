import keras.models
import numpy as np
from PIL import Image, ImageDraw
from losses import *
from constants import *

def eval2(id):
    # check how well the keypoint regression model performs
    x_cropped = np.load("data/x_cropped.npy")
    y_keypts = np.load("data/y_keypts.npy")
    model = keras.models.load_model("models/saved-model-11-0.00.h5")
    # note that in this case the indices of x_cropped may not correspond to the image id
    # since some images contain two shoes for which we have two datapoints
    img = x_cropped[id]
    y_keypts_pred = model.predict(img.reshape(1, img_size_cropped, img_size_cropped, 3))[0]
    img = Image.fromarray(np.uint8(img * 255.0))
    print(y_keypts_pred)
    print(y_keypts[id])
    y_keypts_pred = y_keypts_pred * img_size_cropped
    y_keypts_given = y_keypts[id] * img_size_cropped
    draw = ImageDraw.Draw(img)
    draw.point(y_keypts_pred, fill="red")
    draw.point(y_keypts_given, fill="green")
    img.save("test.png")


def eval1(img_id):
    # test how well the bounding box model performs
    y = np.load("data/y.npy")
    model = keras.models.load_model("models/saved-model-11-0.69.h5", custom_objects={'myLoss': myLoss})
    test_img_path = image_paths[img_id]
    img = Image.open(test_img_path)
    img_ = np.array(img.convert('RGB').resize((img_size_red, img_size_red)), dtype=np.float16).reshape((1, img_size_red, img_size_red, 3)) / 255.0
    y_pred = model.predict(img_)[0]
    print(y_pred)
    print(y[img_id])
    y_pred = y_pred * img_size
    y_given = y[img_id] * img_size
    bb_l = ((y_pred[2], y_pred[3]), (y_pred[4], y_pred[5]))
    bb_r = ((y_pred[6], y_pred[7]), (y_pred[8], y_pred[9]))
    bb_l_given = ((y_given[2], y_given[3]), (y_given[4], y_given[5]))
    bb_r_given = ((y_given[6], y_given[7]), (y_given[8], y_given[9]))
    img = img.resize((img_size, img_size))
    draw = ImageDraw.Draw(img)
    draw.rectangle(bb_l, outline ="red")
    draw.rectangle(bb_r, outline ="red")
    draw.rectangle(bb_l_given, outline="green")
    draw.rectangle(bb_r_given, outline="green")
    # img.show() 
    img.save("test.png")

if __name__ == "__main__":
    eval1(11000)
    # eval2(11000)