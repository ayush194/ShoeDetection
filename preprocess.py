from PIL import Image, ImageDraw
import numpy as np
from constants import *

def preprocess(save_file=True):
	train_data = np.load("data/train_data.npy")

	# format train_data
	y = []
	y_keypts = []
	x = []
	x_cropped = []
	for i in range(num_img):
		# normalized bounding box coordinates
		# frame of reference has been shifted (top left -> (0, 0) and right, down are x, y axis)
		bb_coords_norm = 
			   [min(train_data[i][2:52:2].astype(np.float32)) + 0.5, 0.5 - max(train_data[i][3:52:2].astype(np.float32)), 
	            max(train_data[i][2:52:2].astype(np.float32)) + 0.5, 0.5 - min(train_data[i][3:52:2].astype(np.float32)),
	            min(train_data[i][52:102:2].astype(np.float32))+ 0.5, 0.5 - max(train_data[i][53:102:2].astype(np.float32)),
	            max(train_data[i][52:102:2].astype(np.float32))+ 0.5, 0.5 - min(train_data[i][53:102:2].astype(np.float32))]
	    y.append([train_data[i][-2].astype(np.float32), train_data[i][-1].astype(np.float32)] + bb_coords_norm)
	    x.append(np.array(Image.open(image_paths[i]).convert('RGB').resize((img_size_red, img_size_red)),
	    						dtype=np.float32).reshape(img_size_red, img_size_red, 3) / 255.0)

	    bb_coords = [t * img_size for i in bb_coords_norm]
	    for j in range(2):
	    	# j == 0 -> left shoe
	    	# j == 1 -> right shoe
		    if (train_data[i][-2+j].astype(np.float32)):
		    	# this image contains the jth shoe
		    	tmp = [train_data[i][-2].astype(np.float32), train_data[i][-1].astype(np.float32)]
		    	tmp[(j+1)%2] = 0.0
		    	for t in range(50):
		    		if (t % 2 == 0):
		    			# this is an x-coordinate of the jth foot
		    			tmp.append((train_data[i][2+j*50+t].astype(np.float32) + 0.5) * img_size - bb_coords[4*j]) / img_size_cropped
		    		else:
		    			# this is a y-coordinate of the jth foot
		    			tmp.append((0.5 - train_data[i][2+j*50+t].astype(np.float32)) * img_size - bb_coords[4*j+1]) / img_size_cropped
				y_keypts.append(tmp)
				img = Image.open(image_paths[i]).convert('RGB').crop(*bb_coords[4*j:4*(j+1)])
				img_new = Image.new('RGB', (img_size_cropped, img_size_cropped), (0, 0, 0))  # Black
				img_new.paste(img, img.getbbox())  # Not centered, top-left corner
		    	x_cropped.append(np.array(img_new, dtype=np.float32).reshape(img_size_cropped, img_size_cropped, 3) / 255.0)
	
	y = np.array(y, dtype=np.float32)
	y_keypts = np.array(y_keypts, dtype=np.float32)
	x = np.array(x, dtype=np.float32)
	x_cropped = np.array(x_cropped, dtype=np.float32)
	if (save_file):
		np.save("x.npy", x)
		np.save("y.npy", y)
		np.save("x_cropped.npy", x_cropped)
		np.save("y_keypts.npy", y_keypts)
	return x, y, x_cropped, y_keypts

if __name__ == "__main__":
    preprocess()