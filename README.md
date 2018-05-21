# objectDetection

Train a classifier to detect a specific object (binary response case) inside an image
using Histogram of Oriented Gradients (HOG) features extraction, and linear support vector machine (SVM)

The model is trained based on a dataset from UIUC Image Database for Car Detection, which can be found [here](http://cogcomp.org/Data/Car/) 

## To run the code:
1. Run "pip install -r requirements.txt"	==> install dependencies
2. Run "python svmTrain.py"		==> train a model
3. Run "python svmTest.py"		==> test the model, expect ~165 seconds for both scaled and un-scaled testing images

## Modify configuration for training:
Read the config.cfg file to have general ideas of free parameters:
1. The training images have dimension [height, width] = [40, 100], and object car is only smaller than the image size by tiny bit. I keep the default training window to be [40, 100] to speed up the training process (since hard-negative mining is not applied)
2. When changing the paths to training sets folder, test sets folder, model folder, and results folders, remember to change the paths accordingly inside config.cfg file
3. Mainly tweak the parameters:
	1. overlapThreshold: an area overlap threshold for NMS, over which we do not accept a bounding box. Typical value: between 0.3 and 0.5
	2. stepSize: a parameter for sliding window method, default to [10, 10], i.e. 10 steps horizontally (left to right) and 10 steps vertically (top to bottom)
	3. scale: re-scaling factor for the image pyramid method, default to 1.5
	4. pyramidMinSize: minimum image size after rescaling in image pyramid method, default to [height, width] = [20, 50]
	5. posProbThreshold: positive probability threshold for the object, attempting to decrease false positives, only those positive detections that have a probability larger than the parameter can be accepted, default to 0.85 

## Modules:
In this project, I used many modules to support the Car Image training: OpenCV (cv2), scikit-learn, scikit-image, numpy, etc. 
1. HOG is implemented in scikit-image
2. SVM is implemented in scikit-learn

## Outline the steps to train and test:
1. Extract HOG features from positive training set (labelled 1)
2. Extract HOG features from negative training set (labelled 0)
3. Train a linear SVM from the HOG features and corresponding labels
4. If training window is smaller than training image dimension, apply hard-negative mining on negative training set to produce a hard-negative sample (i.e. label false-positive detections as 0)
5. Re-train model with hard-negative sample => attempt to reduce the number of false-positive detections
6. Apply model to testing sets, and apply non-maximum suppression (NMS) technique to retain the most significant bounding boxes. Observations:
	1. Un-scaled testing set: good result
	2. Scaled testing set: a lot of false positives


