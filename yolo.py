import cv2
import numpy as np
import math
from sparse_stereo import getSparseDistance 
from dense_stereo import getDenseDistance

# contains elements from yolo.py by Toby Breckon

# init YOLO CNN object detection model

confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4   # Non-maximum suppression threshold
inpWidth = 416       # Width of network's input image
inpHeight = 416      # Height of network's input image

config_file = 'yolov3.cfg'
weights_file = 'yolov3.weights'

# Load names of classes from file

classesFile = 'coco.names'


#####################################################################
# Draw the predicted bounding box on the specified image
# image: image detection performed on
# class_name: string name of detected object_detection
# left, top, right, bottom: rectangle parameters for detection
# colour: to draw detection rectangle in

useful_classes = ['person', 'car', 'bus', 'truck', 'motorbike', 'train', 'bicycle']

def drawPred(image, class_name, confidence, left, top, right, bottom, colour, disparityMap, sparse):
    # Get distance value. If the distance isn't useful, do not draw this bounding box.
    if sparse:
        distance = getSparseDistance(left, top, right, bottom, disparityMap)
    else:
        distance = getDenseDistance(left, top, right, bottom, disparityMap)
    if not distance > 0: return 99999
    if class_name in useful_classes: colour = (34, 181, 44)
    else: return 99999
    
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
    return distance

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


def initialise():
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

    return net, output_layer_names, classes

def run(net, output_layer_names, classes, imgL, imgL_adjusted, disparity_map, sparse):
    # create a 4D tensor (OpenCV 'blob') from image frame (pixels scaled 0->1, image resized)
    tensor = cv2.dnn.blobFromImage(imgL_adjusted, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # set the input to the CNN network
    net.setInput(tensor)

    # runs forward inference to get output of the final output layers
    results = net.forward(output_layer_names)

    # remove the bounding boxes with low confidence using confidence threshold 0.5
    classIDs, confidences, boxes = postprocess(imgL_adjusted, results, 0.5, nmsThreshold)

    # draw resulting detections on image
    minDistance = 99999
    for detected_object in range(len(boxes)):
        box = boxes[detected_object]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        distance = drawPred(imgL, classes[classIDs[detected_object]], confidences[detected_object], left, top, left + width, top + height, (255, 178, 50), disparity_map, sparse)
        minDistance = min(distance,minDistance)

    if minDistance == 99999:
        minDistance = 0

    return imgL, minDistance


