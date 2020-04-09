import keras.models
import numpy as np
from constants import *

def eval():
    # test how well the model performs
    model = keras.models.load_model("saved-model-11-0.69.h5", custom_objects={'myLoss': myLoss})
    test_img_path = image_paths[11000]
    img = Image.open(test_img_path)
    img_ = np.array(img.convert('RGB').resize((img_size_red, img_size_red)), dtype=np.float16).reshape((1, img_size_red, img_size_red, 3))
    y_pred = model.predict(img_)[0]
    print(y_pred)
    print(y[11000])
    y_pred = y_pred * img_size
    y_given = y[11000] * img_size
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