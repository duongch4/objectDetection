# import the necessary packages
import numpy as np
import imutils

#----------------------------------------------------

def getNonMaxSuppression_fast(boxes, overlapThreshold = 0.3):
	'''
	Non-max suppression (fast method) based on Malisiewicz et al.
	Input: 
		1. boxes: Overlap boxes, defined by upper-left and lower-right points
		2. overlapThreshold: Overlap threshold, default to 0.3
	Output:
		return only the relevant bounding boxes
	'''

	# If boxes is not a numpy.array, turn it into one
	if ( type(boxes) is not np.ndarray ):
		boxes = np.array(boxes)

	# If no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# If the coordinates are integers, convert them to floats
	# => important because of divisions
	if ( boxes.dtype.kind == "i" ):
		boxes = boxes.astype("float")

	# Initialise the list of picked indexes	
	pick = []

	# Get the coordinates of the bounding boxes
	upperLeft_x = boxes[:,0]
	upperLeft_y = boxes[:,1]
	lowerRight_x = boxes[:,2]
	lowerRight_y = boxes[:,3]

	# Compute the area of the bounding boxes and
	# Sort the bounding boxes by the lower-right y-coordinate of the bounding box, return the indices of sorted array
	area = (lowerRight_x - upperLeft_x + 1) * (lowerRight_y - upperLeft_y + 1)
	idxs = np.argsort(lowerRight_y)

	# Keep looping while some indexes still remain in the indexes list
	while ( len(idxs) > 0 ):
		# Get the last index in the indexes list and
		# Add the index value to the list of picked indexes
		last_idx = len(idxs) - 1
		max_y_idx = idxs[last_idx]
		pick.append(max_y_idx)

		# Find the largest (x, y) coordinates array (array comparison!!) for the upper-left point
		# and the smallest (x, y) coordinates array (array comparison!!) for the lower-right point
		upperLeft_maxArr_x = np.maximum(upperLeft_x[max_y_idx], upperLeft_x[idxs[:last_idx]])
		upperLeft_maxArr_y = np.maximum(upperLeft_y[max_y_idx], upperLeft_y[idxs[:last_idx]])
		lowerRight_minArr_x = np.minimum(lowerRight_x[max_y_idx], lowerRight_x[idxs[:last_idx]])
		lowerRight_minArr_y = np.minimum(lowerRight_y[max_y_idx], lowerRight_y[idxs[:last_idx]])

		# Compute the width and height, based on max array comparison
		w = np.maximum(0, lowerRight_minArr_x - upperLeft_maxArr_x + 1)
		h = np.maximum(0, lowerRight_minArr_y - upperLeft_maxArr_y + 1)

		# Compute the ratio of overlap -> return a 1-d array
		overlap = (w * h) / area[idxs[:last_idx]]

		# Get indices of values > overlapThreshold (bad ones)
		# Since overlap is a 1-d array, the return tuple has empty second element (ie no cols)
		# => get the first element, ie array of row's indices of overlap
		overlap_badIdxs = np.where(overlap > overlapThreshold)[0]

		# Since we already added last_idx to pick[], put it inside the to-be-deleted-array
		del_tuple = ( [last_idx], overlap_badIdxs )
		del_array = np.concatenate(del_tuple)

		# Delete elements of idxs based on del_array
		idxs = np.delete(idxs, del_array)

	# Return only the bounding boxes that were picked as integer data type
	return boxes[pick].astype("int")

#----------------------------------------------------

def getSlidingWindow(image, stepSize = (10,10), windowSize = (20,20)):
    '''
	Get sliding window
	Input:
		1. image: Input Image
		2. stepSize: Incremented Size of Window
			Index (0, 1) for (y-direction step, x-direction step)
		3. windowSize: Size of Sliding Window, default to (20,20)
			as: windowSize[0] = nrows = height, windowSize[1] = ncols = width

	Output:
		yield a sequence of patches (windows) of the input image of windowSize
		The first window has upper-left co-ordinates (0, 0) 
		and are increment in both x and y directions by stepSize

		Type: tuple: (x, y, imWindow) where
			x: upper-left x-coordinate
			y: upper-left y-coordinate
			imWindow: the sliding window image
    '''

    for y in range(0, image.shape[0], stepSize[0]):
        for x in range(0, image.shape[1], stepSize[1]):
            yield (x, y, image[ y:(y + windowSize[0]), x:(x + windowSize[1]) ])

#----------------------------------------------------

def getImagePyramid(image, scale = 1.5, minSize = (20,20)):
	'''
	Build an image pyramid
	Input:
		1. image: Input image
		2. scale: scaling factor, default to 1.5
		3. minSize: minimum size threshold, default to (30,20)
			as: minSize[0] = nrows = height, minSize[1] = ncols = width
	Output:
		yield a sequense of images (this is a generator)
	'''
	# Yield original image
	yield image

	# Keep looping over the pyramid
	while True:

		# Resize 
		newWidth = int(image.shape[1] / scale)
		image = imutils.resize(image, width = newWidth)

		# If the resized image goes lower than the given minSize
		# then stop building pyramid
		if (image.shape[0] < minSize[0] or image.shape[1] < minSize[1]):
			break

		# Yield new image
		yield image