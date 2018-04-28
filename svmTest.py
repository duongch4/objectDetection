# Import libraries
from skimage.feature import hog
from sklearn.externals import joblib

import glob
import cv2
import argparse
import time
import os
import re

import helpers
import config
import svmTrain

#--------------------SVM-Test------------------------
def main():
	# Argument Parser
	parser = argparse.ArgumentParser()

	#parser.add_argument("-t", "--testpath", help = "Path to test images", required = True)
	parser.add_argument("-v", "--visualise", help = "Visualise sliding window", action = "store_true")
	
	args = vars(parser.parse_args())
	
	visualise = args["visualise"]

	print("Testing single-scale images")
	for imPath in glob.glob(os.path.join(config.testPath, "*")):
		svmTest(imPath, visualise, isScale = False)
	print("Single-scale results saved to {}".format(config.resultPath))

	print("Testing multi-scale images")
	for imPath in glob.glob(os.path.join(config.testScalePath, "*")):
		svmTest(imPath, visualise, isScale = True)
	print("Multi-scale results saved to {}".format(config.resultScalePath))

def svmTest(imPath, visualise, isScale):
	# Read the input test image
	image = cv2.imread(imPath)
	grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Get min window size from the train image (very important to have same size so that hog can work!)
	minWindowSize = config.trainWindowSize
	
	# Load the model
	model = joblib.load(config.modelPath)

	# Initialise detections (bounding boxes)
	boxes = []

	# For each scale, for each sliding window, decide whether to add the window to boxes 
	getBoxes(grey, minWindowSize, model, boxes, visualise)

	# NMS to get rid of overlapping boxes
	getNMS(image, boxes, visualise)
	
	# Save result
	# If results directory not exist, create it
	if ( not isScale and not os.path.isdir(config.resultPath) ):
		print("Create single-scale results directory")
		os.makedirs(config.resultPath)
	
	if ( isScale and not os.path.isdir(config.resultScalePath) ):
		print("Create multi-scale results directory")
		os.makedirs(config.resultScalePath)

	imName = os.path.split(imPath)[1]
	resultName = "result-" + re.split("\W", imName)[1] + ".png"
	if (not isScale):
		resultPath = os.path.join(config.resultPath, resultName)
		cv2.imwrite(resultPath, image)
	else:
		resultPath = os.path.join(config.resultScalePath, resultName)
		cv2.imwrite(resultPath, image)

def getBoxes(grey, minWindowSize, model, boxes, visualise):
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

			# Get prediction (0:neg or 1:pos)
			prediction = model.predict(features.reshape(1,-1))
			probability = model.predict_proba(features.reshape(1,-1))

			if (prediction == 1 and probability[:,1] > config.posProbThreshold):
				#print("Got something at (x, y) = ({}, {})".format(x,y))
				rows, cols = window.shape
				lowerRight_x = x + cols - 1
				lowerRight_y = y + rows - 1
				boxes.append( (x, y, lowerRight_x, lowerRight_y) )

			# Draw window if --visualise is indicated
			if (visualise):
				clone = cv2.cvtColor(imResized, cv2.COLOR_GRAY2BGR).copy()
				cv2.rectangle(clone,
				              (x, y),
							  (x + window.shape[1] - 1, y + window.shape[0] - 1),
							  color = (0, 255, 0), thickness = 2)
				cv2.imshow("Window", clone)
				cv2.waitKey(1)
				time.sleep(0.025)

def getNMS(image, boxes, visualise):
	#print( "Initial number of boxes: %d" % (len(boxes)) )

	imOrigin = image.copy()
	# Before NMS
	for (ul_x, ul_y, lr_x, lr_y) in boxes:
		cv2.rectangle(imOrigin, (ul_x, ul_y), (lr_x, lr_y), (0, 0, 255), 2)

	# Perform NMS
	picks = helpers.getNonMaxSuppression_fast(boxes, config.overlapThreshold)
		
	#print( "After NMS, number of boxes: %d" % (len(picks)) )

	# After NMS
	for (ul_x, ul_y, lr_x, lr_y) in picks:
		cv2.rectangle(image, (ul_x, ul_y), (lr_x, lr_y), (0, 255, 0), 2)

	# Display
	if (visualise):
		cv2.imshow("Original image", imOrigin)
		cv2.imshow("After NMS", image)
		cv2.waitKey(0)

#----------------------Main--------------------------
if __name__ == "__main__":
	startTime = time.time()
	main()
	print("--- %s seconds ---" % (time.time() - startTime))