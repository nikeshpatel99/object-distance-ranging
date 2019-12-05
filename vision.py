import cv2
import argparse
import sys
import math
import numpy as np
import os

########### elements of YOLO object detection - by Toby Breckon ################

keep_processing = True

################################################################################
# dummy on trackbar callback function
def on_trackbar(val):
    return


def sparseStereo(grayL,grayR):
    # initialise an ORB detector
    orb = cv2.ORB_create()

    # detect feature points and descriptors in left and right images
    keyPointsL, descriptorsL = orb.detectAndCompute(grayL,None)
    keyPointsR, descriptorsR = orb.detectAndCompute(grayR,None)

    
    # initalise a brute force matcher that will match our descriptors together
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # match descriptors
    matches = matcher.match(descriptorsL,descriptorsR)

    sparseDisparity = {}
    # calculate and store disparity for each feature point
    for match in matches:
        keyPointIndexL = match.trainIdx
        keyPointIndexR = match.queryIdx
        keyPointLCoords = keyPointsL[keyPointIndexL].pt
        keyPointRCoords = keyPointsR[keyPointIndexR].pt
        sparseDisparity[(int(keyPointLCoords[0]),int(keyPointLCoords[1]))] = abs(keyPointLCoords[1] - keyPointRCoords[1])

    featurePointImg = cv2.drawKeypoints(grayL,keyPointsL,None,flags=2)
    cv2.imshow('featurePointImg',featurePointImg)
    return sparseDisparity

def getSparseDistance(left, top,right,bottom):
    camera_focal_length_px = 399.9745178222656  # focal length in pixels
    stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres
    #sumOfDisparity = 0
    #numOfFeaturePoints = 0
    points = []
    for i in range(top,bottom):
        for j in range(left,right):
            disparity = sparseDisparity.get((i,j),-1)
            if disparity == -1:
                continue
            points.append(disparity)
            #sumOfDisparity += disparity
            #numOfFeaturePoints += 1
    #if numOfFeaturePoints == 0:
    #    return -1
    #disparityValue = sumOfDisparity / numOfFeaturePoints
    print(np.median(np.array(points)))
    if np.median(np.array(points)) < 0:
        return -1
    return (camera_focal_length_px * stereo_camera_baseline_m) / np.median(np.array(points))

def getDistance(left, top, right, bottom):
    left = max(left,0)
    top = max(top,0)
    right = max(right,0)
    bottom = max(bottom,0)
    camera_focal_length_px = 399.9745178222656  # focal length in pixels
    stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres
    
    bounded_disparity = disparity_scaled[top:bottom,left:right].ravel()
    bounded_disparity = bounded_disparity[bounded_disparity > 0]

    if len(bounded_disparity) == 0:
        return -1
    
    thresh, _ = cv2.threshold(bounded_disparity, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bounded_disparity = bounded_disparity[bounded_disparity > thresh]
    if np.median(bounded_disparity) <= 0:
        return -1
    print(np.percentile(bounded_disparity,75))
    return (camera_focal_length_px * stereo_camera_baseline_m) / np.percentile(bounded_disparity,75)

#####################################################################
# Draw the predicted bounding box on the specified image
# image: image detection performed on
# class_name: string name of detected object_detection
# left, top, right, bottom: rectangle parameters for detection
# colour: to draw detection rectangle in

useful_classes = ['person', 'car', 'bus', 'truck']

def drawPred(image, class_name, confidence, left, top, right, bottom, colour):
    # Get distance value. If the distance isn't useful, do not draw this bounding box.
    if sparse:
        distance = getSparseDistance(left, top, right, bottom)
    else:
        distance = getDistance(left, top, right, bottom)
    if distance <= 0: return
    if class_name in useful_classes: colour = (34, 181, 44)
##    if class_name == 'person':
##        grayImg = cv2.cvtColor(image[top:bottom,left:right],cv2.COLOR_BGR2GRAY)
##        contours, _ = cv2.findContours(grayImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
##        coutours = contours[0]
##        ellipse = cv2.fitEllipse(contours)
##        cv2.ellipse(image, ellipse, colour, 2)
##    else:
    # Draw a bounding box
    cv2.rectangle(image, (left, top), (right, bottom), colour, 3)

    # construct label
    label = '%s:%.2fm' % (class_name, distance)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(image, (left, top - round(1.5*labelSize[1])),
        (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

#####################################################################
# Remove the bounding boxes with low confidence using non-maxima suppression
# image: image detection performed on
# results: output from YOLO CNN network
# threshold_confidence: threshold on keeping detection
# threshold_nms: threshold used in non maximum suppression

def postprocess(image, results, threshold_confidence, threshold_nms):
    frameHeight = image.shape[0]
    frameWidth = image.shape[1]

    # Scan through all the bounding boxes output from the network and..
    # 1. keep only the ones with high confidence scores.
    # 2. assign the box class label as the class with the highest score.
    # 3. construct a list of bounding boxes, class labels and confidence scores

    classIds = []
    confidences = []
    boxes = []
    for result in results:
        for detection in result:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > threshold_confidence:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences
    classIds_nms = []
    confidences_nms = []
    boxes_nms = []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold_confidence, threshold_nms)
    for i in indices:
        i = i[0]
        classIds_nms.append(classIds[i])
        confidences_nms.append(confidences[i])
        boxes_nms.append(boxes[i])

    # return post processed lists of classIds, confidences and bounding boxes
    return (classIds_nms, confidences_nms, boxes_nms)

################################################################################
# Get the names of the output layers of the CNN network
# net : an OpenCV DNN module network object

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

################################################################################


################################################################################

# init YOLO CNN object detection model

confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image

config_file = 'yolov3.cfg'
weights_file = 'yolov3.weights'

# Load names of classes from file

classesFile = 'coco.names'
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# load configuration and weight files for the model and load the network using them

net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
output_layer_names = getOutputsNames(net)

 # defaults DNN_BACKEND_INFERENCE_ENGINE if Intel Inference Engine lib available or DNN_BACKEND_OPENCV otherwise
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)

# change to cv2.dnn.DNN_TARGET_CPU (slower) if this causes issues (should fail gracefully if OpenCL not available)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

################################################################################

################################################################################

############# elements of YOLO object detection - by Toby Breckon ##############

############# dense stero (TODO change to sparse) ###############

# data set paths

master_path_to_dataset = "./TTBB-durham-02-10-17-sub10"; # ** need to edit this **
directory_to_cycle_left = "left-images";     # edit this if needed
directory_to_cycle_right = "right-images";   # edit this if needed

# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns

skip_forward_file_pattern = ""; # set to timestamp to skip forward to

pause_playback = False; # pause until key press after each image

#####################################################################

# resolve full directory location of data set for left / right images

full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);

# get a list of the left image files and sort them (by timestamp in filename)

left_file_list = sorted(os.listdir(full_path_directory_left));

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

max_disparity = 128;
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);
sparse = True
for filename_left in left_file_list:

    # skip forward to start a file we specify by timestamp (if this is set)

    if ((len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename_left)):
        continue;
    elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left)):
        skip_forward_file_pattern = "";

    # from the left image filename get the correspondoning right image

    filename_right = filename_left.replace("_L", "_R");
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left);
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right);

    # for sanity print out these filenames

    print(full_path_filename_left);
    print(full_path_filename_right);
    print();

    # check the file is a PNG file (left) and check a correspondoning right image
    # actually exists

    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        ####################################
        #      Disparity Calculations      #
        ####################################

        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel
        # RGB images so load both as such

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgL = imgL[0:390,:]
        cv2.imshow('left image',imgL)

        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
        imgR = imgR[0:390,:]
        cv2.imshow('right image',imgR)

        print("-- files loaded successfully");
        print();

        # remember to convert to grayscale (as the disparity matching works on grayscale)
        # N.B. need to do for both as both are 3-channel images

        grayL_unfiltered = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY);
        grayR_unfiltered = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY);

        # perform preprocessing - raise to the power, as this subjectively appears
        # to improve subsequent disparity calculation
        # TODO add other preprocessing: histogram equalisation, noise removal (bilateral filter, cv2 processing etc.)

        grayL = np.power(grayL_unfiltered, 0.75).astype('uint8');
        grayL = cv2.bilateralFilter(grayL,11,50,50)
        grayL = cv2.equalizeHist(grayL)
        grayR = np.power(grayR_unfiltered, 0.75).astype('uint8');
        grayR = cv2.bilateralFilter(grayR,11,50,50)
        grayR = cv2.equalizeHist(grayR)

        disparity_scaled = []
        def denseStereo():
            global disparity_scaled
            # compute disparity image from undistorted and rectified stereo images
            # that we have loaded
            # (which for reasons best known to the OpenCV developers is returned scaled by 16)

            disparity = stereoProcessor.compute(grayL,grayR);

            # filter out noise and speckles (adjust parameters as needed)

            dispNoiseFilter = 5; # increase for more agressive filtering
            cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);


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

        #denseStereo()
        print("Entering sparse test")
        sparseDisparity = sparseStereo(grayL_unfiltered, grayR_unfiltered)
        
        ############################################
        #             Object Detection             #
        ############################################

        width = np.size(imgL, 1)
        imgL = imgL[:,135:width]

        # image preprocessing to optimise YOLO
        # convert image to HSV colour space in order to access luminence channel V
        hsv_imgL = cv2.cvtColor(imgL,cv2.COLOR_BGR2HSV)
        # split channels
        H, S, V = cv2.split(hsv_imgL)
        # create a Contrast Limited Adaptive Histogram Equalization instance and apply to our input image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrastCorrectedV = clahe.apply(V)
        # merge channels back together and convert image back to a BGR format
        imgL_adjusted = cv2.merge((H,S,contrastCorrectedV))
        imgL_adjusted = cv2.cvtColor(imgL_adjusted,cv2.COLOR_HSV2BGR)
        

        # define display window name + trackbar

        windowName = 'YOLOv3 object detection: ' + weights_file
        cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

        # start a timer (to see how long processing and display takes)
        start_t = cv2.getTickCount()

        # create a 4D tensor (OpenCV 'blob') from image frame (pixels scaled 0->1, image resized)
        tensor = cv2.dnn.blobFromImage(imgL_adjusted, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # set the input to the CNN network
        net.setInput(tensor)

        # runs forward inference to get output of the final output layers
        results = net.forward(output_layer_names)

        # remove the bounding boxes with low confidence using confidence threshold 0.5
        classIDs, confidences, boxes = postprocess(imgL_adjusted, results, 0.5, nmsThreshold)

        # draw resulting detections on image
        for detected_object in range(len(boxes)):
            box = boxes[detected_object]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            drawPred(imgL, classes[classIDs[detected_object]], confidences[detected_object], left, top, left + width, top + height, (255, 178, 50))

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(imgL, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # display image
        cv2.imshow(windowName,imgL)
        
        # stop the timer and convert to ms. (to see how long processing and display takes)
        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000

                # keyboard input for exit (as standard), save disparity and cropping
        # exit - x
        # crop - c
        # pause - space

        key = cv2.waitKey(40 * (not(pause_playback))) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if (key == ord('x')):       # exit
            break; # exit
        elif (key == ord(' ')):     # pause (on next frame)
            pause_playback = not(pause_playback);
    else:
            print("-- files skipped (perhaps one is missing or not PNG)")
            print()


# close all windows

cv2.destroyAllWindows()

#####################################################################


