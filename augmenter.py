import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.batches import Batch
import numpy as np

def augment(x, y, deg):
    new_x, new_y = [], []
    # Batch(images=x, bounding_boxes=[[BoundingBox(*row[2:6]),  for row in y]);
    seq = iaa.Sequential([
        iaa.Affine(rotate=deg)
    ])
    for dpt_x, dpt_y in zip(x, y):
        img_aug, bbs_aug = seq(image=dpt_x, 
                                bounding_boxes=BoundingBoxesOnImage([
                                                 BoundingBox(*dpt_y[2:6]), 
                                                 BoundingBox(*dpt_y[6:10])],
                                                 shape=dpt_x.shape))
        new_x.append(img_aug)
        a = bbs_aug.bounding_boxes[0]
        b = bbs_aug.bounding_boxes[1]
        new_y.append([dpt_y[0],dpt_y[1],a.x1,a.y1,a.x2,a.y2,b.x1,b.y1,b.x2,b.y2])
    return np.array(new_x, dtype=np.float32), np.array(new_y, dtype=np.float32)