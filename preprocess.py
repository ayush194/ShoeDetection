import os
from PIL import Image, ImageDraw
import numpy as np
from constants import *

def preprocess(save_file=True):
	train_data = np.load("data/train_data.npy")

	# format train_data
	y = []
	for i in range(len(train_data)):
	    y.append([train_data[i][-2].astype(np.float32), train_data[i][-1].astype(np.float32),
	             min(train_data[i][2:52:2].astype(np.float32)) + 0.5, 0.5 - max(train_data[i][3:52:2].astype(np.float32)), 
	             max(train_data[i][2:52:2].astype(np.float32)) + 0.5, 0.5 - min(train_data[i][3:52:2].astype(np.float32)),
	             min(train_data[i][52:102:2].astype(np.float32))+ 0.5, 0.5 - max(train_data[i][53:102:2].astype(np.float32)),
	             max(train_data[i][52:102:2].astype(np.float32))+ 0.5, 0.5 - min(train_data[i][53:102:2].astype(np.float32))])
	y = np.array(y, dtype=np.float32)

	x = []
	for i, image_path in enumerate(image_paths[:17000]):
	    x.append(np.array(Image.open(image_path).convert('RGB').resize((img_size_red, img_size_red)), 
	                      dtype=np.float32).reshape(img_size_red, img_size_red, 3) / 255.0)
	x = np.array(x, dtype=np.float32)
	if (save_file):
		np.save("x.npy", x)
		np.save("y.npy", y)
	return x, y

if __name__ == "__main__":
    preprocess()