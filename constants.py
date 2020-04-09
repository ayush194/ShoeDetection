import os

# there are a total 17019 images
num_img = 17019
# each image has size 512 x 512
img_size = 512
# we will downscale the images to 224 x 224
img_size_red = 224
# training : validation datapoints
split_ratio = 0.8

curr_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(curr_path, "train")
image_paths = [os.path.join(data_path, str(i) + ".png") for i in range(num_img)]