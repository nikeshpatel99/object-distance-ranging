import cv2
import numpy as np

# calculates the distance of an object within a bounding box using sparse disparity methods
# left, top, right, bottom: the location of the corners of the bounding box
# sparseDisparity: dictionary of sparse disparity values index by y,x location in the left image
def getSparseDistance(left, top, right, bottom, sparseDisparity):
    camera_focal_length_px = 399.9745178222656  # focal length in pixels
    stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres
    points = []
    # loop over all the pixels within the bounding box, check if feature points exist
    for i in range(top,bottom):
        for j in range(left,right):
            disparity = sparseDisparity.get((i,j),0)
            if disparity == 0: # if the feature point doesn't exist or isn't useful, skip the pixel
                continue
            points.append(disparity) # if the keypoint exists and is useful - append to our array of useful points
    # convert to numpy filestructure to facilitate numpy operations
    points = np.array(points)
    # if the median is <= 0, the distance calculation will incorrect, so we return out with an error value
    if np.median(points) <= 0:
        return -1
    # return the distance
    return (camera_focal_length_px * stereo_camera_baseline_m) / np.nanmedian(points)

def sparseStereo(grayL,grayR):
    # initialise an ORB detector
    orb = cv2.ORB_create(5000)

    # detect feature points and descriptors in left and right images
    keyPointsL, descriptorsL = orb.detectAndCompute(grayL,None)
    keyPointsR, descriptorsR = orb.detectAndCompute(grayR,None)
    
    # initalise a brute force matcher that will match our descriptors together
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # match descriptors
    matches = matcher.knnMatch(descriptorsR, trainDescriptors = descriptorsL, k = 2)

    # filter out poor matches, keep the goods ones (taken from surf_detection.py - Toby Breckon)
    goodMatches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            goodMatches.append(m)

    sparseDisparity = {}
    # calculate and store disparity for each feature point
    for match in goodMatches:
        # get the indexes for the left and right keypoints of the match 
        keyPointIndexL = match.trainIdx
        keyPointIndexR = match.queryIdx
        # get the pixel location in the left and right images of the keypoints
        keyPointLCoords = keyPointsL[keyPointIndexL].pt
        keyPointRCoords = keyPointsR[keyPointIndexR].pt
        # calculate the disparity value of the keypoint and store it in a dictionary indexed by the location of the keypoint in the left image
        sparseDisparity[(int(keyPointLCoords[1]),int(keyPointLCoords[0]))] = ((keyPointLCoords[0]-keyPointRCoords[0])**2  + (keyPointLCoords[1]-keyPointRCoords[1])**2)**0.5

#    featurePointImg = cv2.drawKeypoints(grayL,keyPointsL,None,flags=2)
#    cv2.imshow('featurePointImg',featurePointImg)
    return sparseDisparity
