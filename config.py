'''
Set the config variable.
'''

import configparser
import json

config = configparser.RawConfigParser()
config.read('./config.cfg')

min_wdw_sz = json.loads(config.get("hog","min_wdw_sz"))
step_size = json.loads(config.get("hog", "step_size"))
orientations = config.getint("hog", "orientations")
pixels_per_cell = json.loads(config.get("hog", "pixels_per_cell"))
cells_per_block = json.loads(config.get("hog", "cells_per_block"))

posFeatPath = config.get("paths", "posFeatPath")
negFeatPath = config.get("paths", "negFeatPath")
modelPath = config.get("paths", "modelPath")

threshold = config.getfloat("nms", "threshold")
