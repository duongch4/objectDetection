# Import libraries - SVM from scikit-learn
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.externals import joblib

# Reading file names
import argparse
import glob
import os
import numpy as np
import cv2
import time

# Import own modules
import config
import helpers
import getHog

#--------------------Constant------------------------

def getTrainImageDimension():
	imPath = glob.glob(os.path.join(config.posTrainPath, "*"))[0]
	image = cv2.imread(imPath)
	grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return grey.shape

#--------------------SVM-Train----------------------

def main():
	# Argument Parser
	parser = argparse.ArgumentParser()

	#parser.add_argument("-p", "--posFeatPath", help = "Path to positive features from HOG", required = True)
	#parser.add_argument("-n", "--negFeatPath", help = "Path to negative features from HOG", required = True)
	parser.add_argument("-c", "--classifierType", help = "Classifier to be used", default = "LinSVM")

	args = vars(parser.parse_args())

	#posFeatPath = args["posFeatPath"]
	#negFeatPath = args["negFeatPath"]
	classifierType = args["classifierType"]

	# Initialise empty arrays
	featureList = []
	labelList = []

	# Extract HOG features and store correct labels
	print("Extracting HOG features...")
	setFeatureLabel(config.posTrainPath, featureList, labelList, 1)
	print("Done extracting HOG features for positive sample")
	setFeatureLabel(config.negTrainPath, featureList, labelList, 0)
	print("Done extracting HOG features for negative sample")

	# Train
	if classifierType is "LinSVM":
		print("Training a linear SVM Classifier...")
		
		model = CalibratedClassifierCV( LinearSVC() )
		model.fit(featureList, labelList)

		# Hard-negative mining
		model = hardNegMining(model, featureList, labelList)
		print("Done hard-negative mining")

		# Create feature directories if not exist
		if not os.path.isdir(os.path.split(config.modelPath)[0]):
			os.makedirs(os.path.split(config.modelPath)[0])
		
		joblib.dump(model, config.modelPath)
		print("Classifier saved to {}".format(config.modelPath))

def setFeatureLabel(samplePath, featureList, labelList, label):
	for imPath in glob.glob(os.path.join(samplePath, "*")):
		image = cv2.imread(imPath)
		grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		minWindowSize = config.trainWindowSize
		for imResized in helpers.getImagePyramid(grey, config.scale, config.pyramidMinSize):
			for (x, y, window) in helpers.getSlidingWindow(imResized, config.stepSize, minWindowSize):
				if (window.shape[0] != minWindowSize[0] or window.shape[1] != minWindowSize[1]):
					continue
				features = hog(window, config.orientations, config.pixels_per_cell,
				               config.cells_per_block, config.block_norm,
					           transform_sqrt = config.transform_sqrt)
				featureList.append(features)
				labelList.append(label)

def hardNegMining(model, featureList, labelList):
	for imPath in glob.glob(os.path.join(config.negTrainPath, "*")):
		image = cv2.imread(imPath)
		grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# Initialise false-positive set of windows (boxes)
		falsePosSample = []
		
		# Get min window size = train window size (very important to have same size so that hog can work!)
		minWindowSize = config.trainWindowSize

		# Loop over image pyramid (just once)
		for imResized in helpers.getImagePyramid(grey, config.scale, getTrainImageDimension()):
			#print("resized image: {}".format(imResized.shape))
			# Loop over sliding window for each layer of pyramid
			for (x, y, window) in helpers.getSlidingWindow(imResized, config.stepSize, minWindowSize):
				# If window does not meet our desired minWindowSize, ignore it
				if (window.shape[0] != minWindowSize[0] or window.shape[1] != minWindowSize[1]):
					continue
				
				# Extract hog features for each window
				features = hog(window, config.orientations,
				               config.pixels_per_cell, config.cells_per_block,
							   config.block_norm, transform_sqrt=config.transform_sqrt)
				
				# Get prediction
				prediction = model.predict(features.reshape(1,-1))
				#probability = model.predict_proba(features.reshape(1,-1))
				#print(prediction)

				# If prediction == 1, store features with the corresponding probability
				# into the false-positive set
				if (prediction == 1):
					#print("yo")
					# Add the associated features to false-positive sample and label 0 (negative)
					falsePosSample.append( features )
		
		featureList.extend(falsePosSample)			
		labelList.extend( [0] * len(falsePosSample) )
		model.fit(featureList, labelList)
	return model

#----------------------Main-------------------------
if __name__ == "__main__":
	startTime = time.time()
	main()
	print("--- %s seconds ---" % (time.time() - startTime))