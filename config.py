'''
Set the config variable.
'''

import configparser
import json

config = configparser.RawConfigParser()
config.read('./config.cfg')

minWindowSize = json.loads(config.get("helpers", "minWindowSize"))
stepSize = json.loads(config.get("helpers", "stepSize"))
overlapThreshold = config.getfloat("helpers", "overlapThreshold")

orientations = config.getint("hog", "orientations")
pixels_per_cell = json.loads(config.get("hog", "pixels_per_cell"))
cells_per_block = json.loads(config.get("hog", "cells_per_block"))

posFeatPath = config.get("paths", "posFeatPath")
negFeatPath = config.get("paths", "negFeatPath")
modelPath = config.get("paths", "modelPath")