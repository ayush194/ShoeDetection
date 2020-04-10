# ShoeDetection
A simple Shoe Detection NeuralNet built on top of MobilenetV2

## Usage
1. First run the python script save_data.py to curl training images from the web. This may take some time.
2. Then run preprocess.py to preprocess the training data and store it in suitable forms as numpy matrices. Note that you need enough RAM (>16GB) to run this since the matrices are stored in memory before writing to file and can occupy a substantial amount of memory. 
3. Now run train.py to train the model. Inside train.py change the main function to train whichever model you want.
4. Finally run eval.py to evaluate the performance on some sample images from the test/validation set. eval.py has two evaluation functions one for the bounding-box regression and the other for the keypoint regression. Pass an image id as an argument to perform bounding-box/keypoint regression on the given image. After running eval.py, the image will be saved with the name "test.png".
