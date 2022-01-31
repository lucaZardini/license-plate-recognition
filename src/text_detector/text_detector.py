import math

import PIL.Image
import time
import cv2
import numpy as np

def funziona(image):
    min_confidence = 0.5
    eastpath = "/home/luca/Scaricati/frozen_east_text_detection/frozen_east_text_detection.pb"

    (H, W) = image.shape[:2]
    # proportion_H = H/32
    # proportion_W = W/32
    # multiple_H = int(math.ceil(proportion_H))
    # multiple_W = int(math.ceil(proportion_W))
    # new_width = multiple_W*32
    # new_height = multiple_H*32
    #
    # image = PIL.Image.new("RGB", (new_width, new_width), (0,0,255))
    #
    # image.paste(image, (0,0))
    #
    # image = np.array(image)
    # print(image.shape)
    image = np.resize(image, (128,128))
    # rW = W / float(new_height)
    # rH = H / float(new_height)

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    # TODO: if does not work, use the following commented line instead of the one which raises the problem.
    net = cv2.dnn.readNetFromTensorflow(eastpath)
    # net = cv2.dnn.readNet(eastpath)
    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()
    # show timing information on text prediction
    print("[INFO] text detection took {:.6f} seconds".format(end - start))

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        # xData0 = geometry[0, 0, y]
        # xData1 = geometry[0, 1, y]
        # xData2 = geometry[0, 2, y]
        # xData3 = geometry[0, 3, y]
        # anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] >= min_confidence:
                return True
    return False
#         # compute the offset factor as our resulting feature maps will
#         # be 4x smaller than the input image
#         (offsetX, offsetY) = (x * 4.0, y * 4.0)
#         # extract the rotation angle for the prediction and then
#         # compute the sin and cosine
#         angle = anglesData[x]
#         cos = np.cos(angle)
#         sin = np.sin(angle)
#         # use the geometry volume to derive the width and height of
#         # the bounding box
#         h = xData0[x] + xData2[x]
#         w = xData1[x] + xData3[x]
#         # compute both the starting and ending (x, y)-coordinates for
#         # the text prediction bounding box
#         endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
#         endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
#         startX = int(endX - w)
#         startY = int(endY - h)
#         # add the bounding box coordinates and probability score to
#         # our respective lists
#         rects.append((startX, startY, endX, endY))
#         confidences.append(scoresData[x])
#
# #  TODO: up to here to is enough for us.
#
# # apply non-maxima suppression to suppress weak, overlapping bounding
# # boxes
# boxes = non_max_suppression(np.array(rects), probs=confidences)
# # loop over the bounding boxes
# for (startX, startY, endX, endY) in boxes:
#     # scale the bounding box coordinates based on the respective
#     # ratios
#     startX = int(startX * rW)
#     startY = int(startY * rH)
#     endX = int(endX * rW)
#     endY = int(endY * rH)
#     # draw the bounding box on the image
#     cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
# # show the output image
# cv2.imshow("Text Detection", orig)
# cv2.waitKey(0)
