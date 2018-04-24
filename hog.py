# Import functions from scikit-image and scikit-learn
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib

# Reading file names
import argparse
import glob
import os

# Import config.py
from config import *

#--------------------Helper--------------------------

def getHog(samplePath, featurePath):
	for imPath in glob.glob(os.path.join(samplePath, "*")):
		im = imread(imPath, as_grey = True)

		if desType == "HOG":
			featDes = hog(im, orientations, pixels_per_cell, cells_per_block)
		
		featDesName = os.path.split(imPath)[1].split(".")[0] + ".feat"
		featDesPath = os.path.join(featurePath, featDesName)
		joblib.dump(featDes, featDesPath)

#--------------------Main----------------------------

def main():
	# Argument Parser
	parser = argparse.ArgumentParser()

	parser.add_argument("-p", "--posPath", help = "Path to positive images", required = True)
	parser.add_argument("-n", "--negPath", help = "Path to negative images", required = True)
	parser.add_argument("-d", "--descriptor", help = "Descriptor to be used -- HOG", default = "HOG")

	args = vars(parser.parse_args())

	posPath = args["posPath"]
	negPath = args["negPath"]
	desType = args["descriptor"]

	# If feature dir dont exist, create them
	if not os.path.isdir(posFeatPath):
		print("Create pos feature dir")
		os.makedirs(posFeatPath)

	if not os.path.isdir(negFeatPath):
		print("Create neg feature dir")
		os.makedirs(negFeatPath)

	print("Descriptors: positive samples: processing...")
	getHog(posPath, posFeatPath)
	print("Positive features saved in {}".format(posFeatPath))

	print("Descriptors: negative samples: processing...")
	getHog(negPath, negFeatPath)
	print("Negative features saved in {}".format(negFeatPath))

#----------------------------------------------------
if __name__ == "__main__":
	main()