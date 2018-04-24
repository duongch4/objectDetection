# Import libraries - SVM from scikit-learn
from skimage.feature import local_binary_pattern
from sklearn.svm import LinearSVC
from sklearn.externals import joblib

# Reading file names
import argparse
import glob
import os

# Import config.py
from config import *

#--------------------Helpers------------------------

def loadFeature(featurePath, featureArray, labelArray, intForLabel):
	for featPath in glob.glob(os.path.join(featurePath, "*.feat")):
		feature = joblib.load(featPath)
		featureArray.append(feature)
		labelArray.append(intForLabel)

#---------------------Train-------------------------

def svmTrain():
	# Argument Parser
	parser = argparse.ArgumentParser()

	parser.add_argument("-p", "--posFeatPath", help = "Path to positive features from HOG", required = True)
	parser.add_argument("-n", "--negFeatPath", help = "Path to negative features from HOG", required = True)
	parser.add_argument("-c", "--classifier", help = "Classifier to be used", default = "LinSVM")

	args = vars(parser.parse_args())

	posFeatPath = args["posFeatPath"]
	negFeatPath = args["negFeatPath"]
	classifierType = args["classifier"]

	# Initialise empty arrays
	features = []
	labels = []

	# Load positive and negative features
	loadFeature(posFeatPath, features, labels, 1)
	loadFeature(negFeatPath, features, labels, 0)

	# Train
	if classifierType is "LinSVM":
		classifier = LinearSVC()
		print("Training a linear SVM Classifier...")
		classifier.fit(features, labels)

		# Create feature directories if not exist
		if not os.path.isdir(os.path.split(modelPath)[0]):
			os.makedirs(os.path.split(modelPath)[0])
		
		joblib.dump(classifier, modelPath)
		print("Classifier saved to {}".format(modelPath))