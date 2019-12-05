import cv2
import numpy as np

max_disparity = 128

# create a dense stereo disparity map from a pair of greyscale images
# grayL, grayR: grayscale input images, right image is rectified
def denseStereo(grayL, grayR):
    # setup the disparity stereo processor to find a maximum of 128 disparity values
    # (adjust parameters if needed - this will effect speed to processing)

    # uses a modified H. Hirschmuller algorithm [Hirschmuller, 2008] that differs (see opencv manual)
    # parameters can be adjusted, current ones from [Hamilton / Breckon et al. 2013]

    # FROM manual: stereoProcessor = cv2.StereoSGBM(numDisparities=128, SADWindowSize=21);

    # From help(cv2): StereoBM_create(...)
    #        StereoBM_create([, numDisparities[, blockSize]]) -> retval
    #
    #    StereoSGBM_create(...)
    #        StereoSGBM_create(minDisparity, numDisparities, blockSize[, P1[, P2[,
    # disp12MaxDiff[, preFilterCap[, uniquenessRatio[, speckleWindowSize[, speckleRange[, mode]]]]]]]]) -> retval
    stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);
    disparity = stereoProcessor.compute(grayL,grayR);

    # filter out noise and speckles (adjust parameters as needed)

    dispNoiseFilter = 5; # increase for more agressive filtering
    cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);

    # filter the disparity image using the Weighted Least Squares technique to obtain
    # a smoother dispartiy map
    wls = cv2.ximgproc.createDisparityWLSFilter(stereoProcessor)
    right = cv2.ximgproc.createRightMatcher(stereoProcessor)
    right = right.compute(grayR, grayL)
    disparity = wls.filter(disparity, grayL, None, right)

    # scale the disparity to 8-bit for viewing
    # divide by 16 and convert to 8-bit image (then range of values should
    # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
    # so we fix this also using a initial threshold between 0 and max_disparity
    # as disparity=-1 means no disparity available

    _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO);
    disparity_scaled = (disparity / 16.).astype(np.uint8);
    width = np.size(disparity_scaled, 1)
    disparity_scaled = disparity_scaled[:,135:width]

    # display image (scaling it to the full 0->255 range based on the number
    # of disparities in use for the stereo part)
    
    cv2.imshow("disparity", (disparity_scaled * (256. / max_disparity)).astype(np.uint8));
    return disparity_scaled

# calculates the distance of an object within a bounding box on an image
def getDenseDistance(left, top, right, bottom, disparity_scaled):
    # error correction
    left = max(left,0)
    top = max(top,0)
    right = max(right,0)
    bottom = max(bottom,0)
    
    camera_focal_length_px = 399.9745178222656  # focal length in pixels
    stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

    # slice the disparity map to only contain the values within the box, then flatten the array
    bounded_disparity = disparity_scaled[top:bottom,left:right].ravel()
    # filter out zero values - these serve no use in the distance calculation
    bounded_disparity = bounded_disparity[bounded_disparity > 0]

    # if we have no meaningful disparity values, return an error value
    if len(bounded_disparity) == 0:
        return -1
    # we calculate a threshold value using OTSU thresholding to seperate background and foreground values in the disparity map
    thresh, _ = cv2.threshold(bounded_disparity, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # filter out the background values
    bounded_disparity = bounded_disparity[bounded_disparity > thresh]
    # if the median is <= 0, the percentile function will crash out, so if this is the case we return the error value
    if np.median(bounded_disparity) <= 0:
        return -1
    # calculate the final distance
    return (camera_focal_length_px * stereo_camera_baseline_m) / np.percentile(bounded_disparity,75)
