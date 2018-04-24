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
import config

#--------------------Helper--------------------------

def getHog(samplePath, featurePath, descriptorType):
	for imPath in glob.glob(os.path.join(samplePath, "*")):
		im = imread(imPath, as_grey = True)

		if descriptorType == "HOG":
			featDes = hog(im, config.orientations,
				          config.pixels_per_cell, config.cells_per_block,
						  block_norm = "L2-Hys")
		
		featDesName = os.path.split(imPath)[1].split(".")[0] + ".feat"
		featDesPath = os.path.join(featurePath, featDesName)
		joblib.dump(featDes, featDesPath)

#--------------------MainHog-------------------------

def mainHog():
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
	if not os.path.isdir(config.posFeatPath):
		print("Create pos feature dir")
		os.makedirs(config.posFeatPath)

	if not os.path.isdir(config.negFeatPath):
		print("Create neg feature dir")
		os.makedirs(config.negFeatPath)

	print("Describing positive samples...")
	getHog(posPath, config.posFeatPath, desType)
	print("Positive features saved in {}".format(config.posFeatPath))

	print("Describing negative samples...")
	getHog(negPath, config.negFeatPath, desType)
	print("Negative features saved in {}".format(config.negFeatPath))

#----------------------Main--------------------------
if __name__ == "__main__":
	mainHog()