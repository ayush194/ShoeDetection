import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables import Keypoint, KeypointsOnImage
from imgaug.augmentables.batches import Batch
import numpy as np

def augment(x, y, deg):
    new_x, new_y = [], []
    # Batch(images=x, bounding_boxes=[[BoundingBox(*row[2:6]),  for row in y]);
    seq = iaa.Sequential([
        iaa.Affine(rotate=deg)
    ])
    for dpt_x, dpt_y in zip(x, y):
        bb1 = [t * img_size_red for t in dpt_y[2:6]]
        bb2 = [t * img_size_red for t in dpt_y[6:10]]
        img_aug, bbs_aug = seq(image=dpt_x, 
                                bounding_boxes=BoundingBoxesOnImage([
                                                 BoundingBox(*bb1), 
                                                 BoundingBox(*bb2)],
                                                 shape=dpt_x.shape))
        new_x.append(img_aug)
        a = bbs_aug.bounding_boxes[0]
        b = bbs_aug.bounding_boxes[1]
        new_y.append([dpt_y[0],dpt_y[1]] + [t / img_size_red for t in [a.x1,a.y1,a.x2,a.y2,b.x1,b.y1,b.x2,b.y2]])
    return np.array(new_x, dtype=np.float32), np.array(new_y, dtype=np.float32)

def augment2(x, y, deg):
    new_x, new_y = [], []
    seq = iaa.Sequential([
        iaa.Affine(rotate=deg)
    ])
    for dpt_x, dpt_y_norm in zip(x, y):
        dpt_y = [t * img_size_cropped for t in dpt_y_norm]
        keypts = KeypointsOnImage([Keypoint(x=dpt_y[i], y=dpt_y[i+1]) for i in range(0, 50, 2)])
        img_aug, keypts_aug = seq(image=dpt_x, keypoints=keypts)
        new_x.append(img_aug)
        new_y.append([t for x in [(keypt.x, keypt.y) for keypt in keypts_aug.keypoints] for t in x])
    return np.array(new_x, dtype=np.float32), np.array(new_y, dtype=np.float32)
