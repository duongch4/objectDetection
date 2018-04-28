'''
Set the config variable.
'''

import configparser
import json

config = configparser.RawConfigParser()
config.read('./config.cfg')

overlapThreshold = config.getfloat("helpers", "overlapThreshold")
stepSize = json.loads(config.get("helpers", "stepSize"))
scale = config.getfloat("helpers", "scale")
pyramidMinSize = json.loads(config.get("helpers", "pyramidMinSize"))
trainWindowSize = json.loads(config.get("helpers", "trainWindowSize"))
posProbThreshold = config.getfloat("helpers", "posProbThreshold")

orientations = config.getint("hog", "orientations")
pixels_per_cell = json.loads(config.get("hog", "pixels_per_cell"))
cells_per_block = json.loads(config.get("hog", "cells_per_block"))
block_norm = config.get("hog", "block_norm")
transform_sqrt = config.getboolean("hog", "transform_sqrt")

posTrainPath = config.get("paths", "posTrainPath")
negTrainPath = config.get("paths", "negTrainPath")

posFeatPath = config.get("paths", "posFeatPath")
negFeatPath = config.get("paths", "negFeatPath")
modelPath = config.get("paths", "modelPath")

testPath = config.get("paths", "testPath")
testScalePath = config.get("paths", "testScalePath")
resultPath = config.get("paths", "resultPath")
resultScalePath = config.get("paths", "resultScalePath")