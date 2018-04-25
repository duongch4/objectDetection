# Import libraries
from skimage.feature import hog
from sklearn.externals import joblib

import cv2
import argparse
import time

import helpers
import config
import getHog

#--------------------SVM-Test------------------------

def svmTest():
	# Argument Parser
	parser = argparse.ArgumentParser()

	parser.add_argument("-i", "--image", help = "Path to test image", required = True)
	parser.add_argument("-v", "--visualise", help = "Visualise sliding window", action = "store_true")

	args = vars(parser.parse_args())

	# Read the input test image
	im = cv2.imread(args["image"])
	grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	# Get the other input
	visualise = args["visualise"]

	# Get min window size from the train image (very important to have same size so that hog can work!)
	minWindowSize = getHog.getTrainImageDimension()
	
	# Load the model
	model = joblib.load(config.modelPath)

	# Initialise detections (bounding boxes)
	boxes = []

	# Loop over image pyramid (downscale)
	for imResized in helpers.getImagePyramid(grey, config.scale, config.pyramidMinSize):
		
		# Loop over sliding window for each layer of the pyramid
		for (x, y, window) in helpers.getSlidingWindow(imResized, config.stepSize, minWindowSize):
			
			# If window does not meet our desired minWindowSize, ignore it
			if (window.shape[0] != minWindowSize[0] or window.shape[1] != minWindowSize[1]):
				continue

			# Extract HOG features for each window
			features = hog(window, config.orientations, 
				           config.pixels_per_cell, config.cells_per_block,
						   config.block_norm, transform_sqrt=config.transform_sqrt)
			#print(window.shape, len(features))
			# Get prediction (0:neg or 1:pos)
			prediction = model.predict(features.reshape(1,-1))
			
			if (prediction == 1):
				print("Got something at (x, y) = ({}, {})".format(x,y))
				rows, cols = window.shape
				lowerRight_x = x + cols #- 1
				lowerRight_y = y + rows #- 1
				boxes.append( (x, y, lowerRight_x, lowerRight_y) )

			# Draw window if --visualise is indicated
			if (visualise):
				clone = cv2.cvtColor(imResized, cv2.COLOR_GRAY2BGR).copy()
				cv2.rectangle(clone,
				              (x, y),
							  (x + window.shape[1] , y + window.shape[0] ),
							  color = (0, 255, 0), thickness = 2)
				cv2.imshow("Window", clone)
				cv2.waitKey(1)
				time.sleep(0.025)

	print( "Initial number of boxes: %d" % (len(boxes)) )
	imOrigin = im.copy()
			
	# Before NMS
	for (ul_x, ul_y, lr_x, lr_y) in boxes:
		cv2.rectangle(imOrigin, (ul_x, ul_y), (lr_x, lr_y), (0, 0, 255), 2)

	# Perform NMS
	picks = helpers.getNonMaxSuppression_fast(boxes, config.overlapThreshold)
		
	print( "After NMS, number of boxes: %d" % (len(picks)) )

	# After NMS
	for (ul_x, ul_y, lr_x, lr_y) in picks:
		cv2.rectangle(im, (ul_x, ul_y), (lr_x, lr_y), (0, 255, 0), 2)

	# Display
	cv2.imshow("Original image", imOrigin)
	cv2.imshow("After NMS", im)
	cv2.waitKey(0)

#----------------------Main--------------------------
if __name__ == "__main__":
	svmTest()