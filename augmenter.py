import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables import Keypoint, KeypointsOnImage
from imgaug.augmentables.batches import Batch
import numpy as np

def augment(x, y, x_new, y_new, deg):
    # Batch(images=x, bounding_boxes=[[BoundingBox(*row[2:6]),  for row in y]);
    seq = iaa.Sequential([
        iaa.Affine(rotate=deg)
    ])
    for dpt_x, dpt_y, dpt_x_new, dpt_y_new in zip(x, y, x_new, y_new):
        bb1 = [t * img_size_red for t in dpt_y[2:6]]
        bb2 = [t * img_size_red for t in dpt_y[6:10]]
        img_aug, bbs_aug = seq(image=dpt_x, 
                                bounding_boxes=BoundingBoxesOnImage([
                                                 BoundingBox(*bb1), 
                                                 BoundingBox(*bb2)],
                                                 shape=dpt_x.shape))
        dpt_x_new[:,:,:] = img_aug
        bbs_aug_clipped = bbs_aug.remove_out_of_image().clip_out_of_image()
        dpt_y_new[:2] = [dpt_y[0],dpt_y[1]]
        for i, bb in enumerate(bbs_aug_clipped):
            if (bb.is_out_of_image(img_aug.shape)):
                # boundingbox is fully out of image
                # remove it
                dpt_y_new[i] = 0.0
                dpt_y_new[4*i+2:4*i+6] = [0.0, 0.0, 0.0, 0.0]
            else:
                # boundingbox is completely or partially in the image
                dpt_y_new[4*i+2:4*i+6] = [t / img_size_red for t in [bb.x1, bb.y1, bb.x2, bb.y2]]
    # return np.array(new_x, dtype=np.float32), np.array(new_y, dtype=np.float32)


def augment2(x, y, x_new, y_new, deg):
    seq = iaa.Sequential([
        iaa.Affine(rotate=deg)
    ])
    for dpt_x, dpt_y, dpt_x_new, dpt_y_new in zip(x, y, x_new, y_new):
        keypts = KeypointsOnImage([Keypoint(x=dpt_y[i] * img_size_cropped, y=dpt_y[i+1] * img_size_cropped) for i in range(0, 50, 2)])
        img_aug, keypts_aug = seq(image=dpt_x, keypoints=keypts)
        dpt_x_new[:,:,:] = img_aug
        dpt_y_new[:] = [t / img_size_cropped for x in [(keypt.x, keypt.y) for keypt in keypts_aug.keypoints] for t in x]
    # return np.array(new_x, dtype=np.float32), np.array(new_y, dtype=np.float32)
