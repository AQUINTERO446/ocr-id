# USAGE
# python text_recognition.py --east frozen_east_text_detection.pb --image images/22.jpg --padding 0.20 --psm 7
# -*- coding: utf-8 -*-
# import the necessary packages
from imutils.object_detection import non_max_suppression
from imutils import rotate_bound
import numpy as np
import pytesseract
import argparse
import cv2
import re
import time
start = time.time()
def contrast_stretching(image):
    """Contrast stretching (normalization) to impruve contrast on image
        and get fastest and better OCR results.

    Parameters
    ----------
    image : numpy.array
        BGR image to be contrast streched.

    Returns
    -------
    image : numpy.array
        Preproced image.

    """
    channels = []
    for i in range(0, 3):
        I=image[:,:,i]
        Imin = np.amin(I)
        Imax = np.amax(I)
        Is = (255/(Imax-Imin))*(I.astype(np.float64) - Imin)
        channels.append(Is.astype(np.uint8))
    return cv2.merge(channels)

def histogram_equalization(image):
    """Histogram equalization to enchance contrast.

    Parameters
    ----------
    image : numpy.array
        BGR image to be equalized.

    Returns
    -------
    image : numpy.array
        Preproced image.

    """
    channels = []
    for i in range(0, 3):
        I=image[:,:,i]
        h,b = np.histogram(I,bins=256,range=[0,256])
        pdf =  h.astype(np.float64)/np.sum(h)
        # CDF
        cdf = np.zeros(256)
        cdf[0] = pdf[0]
        for i in range(256):
            cdf[i] = cdf[i-1] + pdf[i]
        # Equalization
        sz = I.shape
        Ie = np.zeros(shape=sz)
        for i in range(sz[0]):
            for j in range(sz[1]):
                Ie[i,j] = cdf[I[i,j]]

        Ie = 255*Ie
        channels.append(Ie.astype(np.uint8))

    return cv2.merge(channels)

def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < args["min_confidence"]:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            angles.append(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)

def merge_areas(boxes):
    """Merge de nearby boxes of text regions on a tolerance height
    so the output is each line of text on the original image.

    Parameters
    ----------
    boxes : list
        All depured boxes over a min confidence.

    Returns
    -------
    boxes_out : list
        list of boxes merged.

    """
    boxes_out = []
    banned_index = []
    rows, columns = boxes.shape
    endY_maxpop= 0
    index_maxY = 0
    index_aux = 0
    for index in range(0,rows):
        boxes_levels = []
        if index in banned_index:
            continue
        boxes_levels.append(boxes[index])
        for index_i in range(0,rows):
            if index is index_i or index_i in banned_index:
                continue
            ySimilarity=(abs(boxes[index][1] - boxes[index_i][1]))
            if (ySimilarity < 10):
                boxes_levels.append(boxes[index_i])
                banned_index.append(index_i)
            startX_max, startY_max, endX_max, endY_max = np.max(boxes_levels, axis=0)
            startX_min, startY_min, endX_min, endY_min = np.min(boxes_levels, axis=0)

            if endY_max > endY_maxpop:
                endY_maxpop = endY_max
                index_maxY = index_aux
        index_aux += 1
        boxes_out.append((startX_min, startY_min, endX_max, endY_max))
    boxes_out.pop(index_maxY)
    return boxes_out

def preprocess(img, gaussianBlur=True, contrastStretching = True, histogramEqualization =False):
    """Basic operetions to impruve features of image.

    Parameters
    ----------
    img : numpy.array
        BGR image to be preproced.
    gaussian_blur : bool
        Apply gaussian blur.
    contrast_stretching : bool
        Apply contrast_stretching to image.
    histogram_equalization : bool
        Apply histogram equalization to image.

    Returns
    -------
    image : numpy.array
        Preprocess image.

    """
    if gaussianBlur:
        img = cv2.GaussianBlur(img,(5,5),0)
    if contrastStretching:
        img = contrast_stretching(img)
    if histogramEqualization:
        img = histogram_equalization(img)
    return img

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
                help="path to input image")
ap.add_argument("-east", "--east", type=str,
                help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
                help="nearest multiple of 32 for resized width")
ap.add_argument("-e", "--height", type=int, default=320,
                help="nearest multiple of 32 for resized height")
ap.add_argument("-p", "--padding", type=float, default=0.0,
                help="amount of padding to add to each border of ROI")
ap.add_argument("-o", "--psm", type=int, default=7,
                help="Page segmentation mode")
args = vars(ap.parse_args())

# load the input image and grab the image dimensions
image = preprocess(cv2.imread(args["image"]))
cv2.imwrite('/home/images/preprocess.png', image)
orig = image.copy()
#List of angles of all regiongs with text
angles = []
(origH, origW) = image.shape[:2]
#Configuration to write on output image depending on height.
if origH >600:
    fontSize = 1.2
    fontborder = 3
    lineborder = 2
else:
    fontSize = 0.5
    fontborder = 1
    lineborder = 1

# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (args["width"], args["height"])
rW = origW / float(newW)
rH = origH / float(newH)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                             (123.68, 116.78, 103.94),
                             swapRB=True,
                             crop=False)
start_detection = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
print('Detection Running time: ', time.time() - start_detection, ' seconds')
# decode the predictions, then  apply non-maxima suppression to
# suppress weak, overlapping bounding boxes
(rects, confidences) = decode_predictions(scores, geometry)
angle = sum(angles)/len(angles)
boxes = non_max_suppression(np.array(rects), probs=confidences)
boxes = merge_areas(boxes)
print('Number of regions detected: ', len(boxes))

# initialize the list of results
results = []

# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
    # scale the bounding box coordinates based on the respective
    # ratios
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    # in order to obtain a better OCR of the text we can potentially
    # apply a bit of padding surrounding the bounding box -- here we
    # are computing the deltas in both the x and y directions
    dX = int((endX - startX) * args["padding"])
    dY = int((endY - startY) * args["padding"])

    # apply padding to each side of the bounding box, respectively
    startX = max(0, startX - dX)
    startY = max(0, startY - dY)
    endX = min(origW, endX + (dX * 2))
    endY = min(origH, endY + (dY * 2))

    # extract the actual padded ROI
    roi = rotate_bound(orig[startY:endY, startX:endX], angle)

    # in order to apply Tesseract v4 to OCR text we must supply
    # (1) a language, (2) an OEM flag of 4, indicating that the we
    # wish to use the LSTM neural net model for OCR, and finally
    # (3) an OEM value, in this case, 7 which implies that we are
    # treating the ROI as a single line of text
    config = ("-l spa --oem 1 --psm "+str(args["psm"]))
    text = pytesseract.image_to_string(roi, config=config)

    # strip out non-ASCII text and to upper case
    pattern = re.compile('([^\s\w]|_)+', re.UNICODE)
    text = pattern.sub('', text).upper()
    #Ignore all empty ocr
    if not text:
        continue
    # cv2.imwrite("/home/images/TextDetection{}.png".format(text), roi)

    # add the bounding box coordinates and OCR'd text to the list
    # of results
    results.append(((startX, startY, endX, endY), text))

# sort the results bounding box coordinates from top to bottom
results = sorted(results, key=lambda r:r[0][1])
output = orig.copy()
# loop over the results
for ((startX, startY, endX, endY), text) in results:
    # display the text OCR'd by Tesseract
    print("OCR TEXT")
    print("========")
    print("{}\n".format(text))
    output = cv2.rectangle(output,
                           (startX, startY),
                           (endX, endY),
                           (0, 0, 255),
                           lineborder)
    output = cv2.putText(output,
                         text,
                         (startX, startY - 20),
                         cv2.FONT_HERSHEY_SIMPLEX,
                         fontSize,
                         (0, 0, 255),
                         fontborder)

# show the output image
cv2.imwrite("/home/images/TextDetection.png", output)
print('Running time: ', time.time() - start, ' seconds')
