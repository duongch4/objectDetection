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

	parser.add_argument("-c", "--classifierType", help = "Classifier to be used", default = "LinSVM")

	args = vars(parser.parse_args())

	classifierType = args["classifierType"]

	# Initialise empty arrays
	featureList = []
	labelList = []

	# Initialise training window size (very important to have same size so that hog can work!)
	trainWindowSize = config.trainWindowSize

	# Extract HOG features and store correct labels
	print("Extracting HOG features...")
	setFeatureLabel(config.posTrainPath, trainWindowSize, featureList, labelList, 1)
	print("Done extracting HOG features for positive sample")
	setFeatureLabel(config.negTrainPath, trainWindowSize, featureList, labelList, 0)
	print("Done extracting HOG features for negative sample")

	# Train
	if classifierType is "LinSVM":
		print("Training a (base) linear SVM Classifier...")
		
		model = CalibratedClassifierCV( LinearSVC() )
		model.fit(featureList, labelList)
		print("Done base training")

		# Hard-negative mining
		if (conditionHardNegMining(trainWindowSize)):
			print("Start hard-negative mining...")
			(model, falsePositiveThreshold) = hardNegMining(model, trainWindowSize, featureList, labelList)
			print("Done hard-negative mining")
			joblib.dump(falsePositiveThreshold, config.falsePositiveThresholdPath)

		# Create feature directories if not exist
		if not os.path.isdir(os.path.split(config.modelPath)[0]):
			os.makedirs(os.path.split(config.modelPath)[0])
		
		joblib.dump(model, config.modelPath)
		print("Classifier saved to {}".format(config.modelPath))

def setFeatureLabel(samplePath, trainWindowSize, featureList, labelList, label):
	for imPath in glob.glob(os.path.join(samplePath, "*")):
		image = cv2.imread(imPath)
		grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		for imResized in helpers.getImagePyramid(grey, config.scale, config.pyramidMinSize):
			for (x, y, window) in helpers.getSlidingWindow(imResized, config.stepSize, trainWindowSize):
				if (window.shape[0] != trainWindowSize[0] or window.shape[1] != trainWindowSize[1]):
					continue
				features = hog(window, config.orientations, config.pixels_per_cell,
				               config.cells_per_block, config.block_norm,
					           transform_sqrt = config.transform_sqrt)
				featureList.append(features)
				labelList.append(label)

def conditionHardNegMining(trainWindowSize): 
	return ( (trainWindowSize[0] < getTrainImageDimension()[0]) and
	         (trainWindowSize[1] < getTrainImageDimension()[1]) )

def hardNegMining(model, trainWindowSize, featureList, labelList):
	for imPath in glob.glob(os.path.join(config.negTrainPath, "*")):
		image = cv2.imread(imPath)
		grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# Initialise false-positive set of windows (boxes), and corresponding probability
		falsePositiveSample = []
		probabilityList = []

		# Loop over image pyramid (just once)
		for imResized in helpers.getImagePyramid(grey, config.scale, getTrainImageDimension()):
			#print("resized image: {}".format(imResized.shape))
			# Loop over sliding window for each layer of pyramid
			for (x, y, window) in helpers.getSlidingWindow(imResized, config.stepSize, trainWindowSize):
				# If window does not meet our desired minWindowSize, ignore it
				if (window.shape[0] != trainWindowSize[0] or window.shape[1] != trainWindowSize[1]):
					continue
				
				# Extract hog features for each window
				features = hog(window, config.orientations,
				               config.pixels_per_cell, config.cells_per_block,
							   config.block_norm, transform_sqrt=config.transform_sqrt)
				
				# Get prediction
				prediction = model.predict(features.reshape(1,-1))
				probability = model.predict_proba(features.reshape(1,-1))
				#print(prediction)

				# If prediction == 1, store features with the corresponding probability
				if (prediction == 1):
					# Store the associated features to false-positive sample
					falsePositiveSample.append( features )
					# Store the probability
					probabilityList.append( probability[0,1] )
		
		featureList.extend(falsePositiveSample)			
		labelList.extend( [0] * len(falsePositiveSample) )
		model.fit(featureList, labelList)

		if (not probabilityList):
			falsePositiveThreshold = config.posProbThreshold
		else:
			falsePositiveThreshold = max(probabilityList)
		
	return (model, falsePositiveThreshold)

#----------------------Main-------------------------
if __name__ == "__main__":
	startTime = time.time()
	main()
	print("--- %s seconds ---" % (time.time() - startTime))