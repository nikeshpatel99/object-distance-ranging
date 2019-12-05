import cv2
import argparse
import sys
import math
import numpy as np
import os
from sparse_stereo import sparseStereo
from dense_stereo import denseStereo
import yolo

keep_processing = True
# if you want to use sparse stereo, set to True, for dense set to False
sparse = False

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

# define display window name
windowName = 'YOLOv3 object detection'
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

# define network from YOLO
net, output_layer_names, classes = yolo.initialise()

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
        # we also perform a bilateral filter to smooth the images
        # and we equalise the image histograms to aim with the brightness in the image

        grayL = np.power(grayL_unfiltered, 0.75).astype('uint8');
        grayL = cv2.bilateralFilter(grayL,11,50,50)
        grayL = cv2.equalizeHist(grayL)
        grayR = np.power(grayR_unfiltered, 0.75).astype('uint8');
        grayR = cv2.bilateralFilter(grayR,11,50,50)
        grayR = cv2.equalizeHist(grayR)

        # calculate the disparity map dependent on the method selected
        if sparse:
            disparity_map = sparseStereo(grayL_unfiltered, grayR_unfiltered)
        else:
            disparity_map = denseStereo(grayL,grayR)
        
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

        # run YOLO object detection on the image
        imgL = yolo.run(net, output_layer_names, classes, imgL, imgL_adjusted, disparity_map, sparse)
        
        # display image
        cv2.imshow(windowName,imgL)
        
        # keyboard input for exit (as standard), save disparity and cropping
        # exit - x
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


