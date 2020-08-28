
# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import time
import threading
#calculate frames:
videoframeNum = 0;#output how many frames
class webcamCapture:
    def __init__(self, filename):
        self.Frame = []
        self.status = False
        self.isstop = False
        self.cameraframeNum = 0 #camera get how many frames
		
	# 攝影機連接。
        self.capture = cv.VideoCapture(1)
	# 設定擷取影像的尺寸大小
        self.capture.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
	# 使用 X264 編碼
        fourcc = cv.VideoWriter_fourcc(*'X264')

	# 建立 VideoWriter 物件，輸出影片至 output.mp4
	# FPS 值為 23.976，解析度為 1280x720
        self.out = cv.VideoWriter(filename, fourcc, 23.976, (1280, 720))

    def start(self):
	# 把程式放進子執行緒，daemon=True 表示該執行緒會隨著主執行緒關閉而關閉。
        print('webcam started!')
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
	# 記得要設計停止無限迴圈的開關。
        self.isstop = True
        print('webcam stopped!')
        print("input frames: ",self.cameraframeNum)
        
    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()
            if (self.status == True):
            	self.cameraframeNum += 1;
            	self.out.write(self.Frame)
        
        self.capture.release()
        self.out.release()

filename = "output.mp4"

# 連接攝影機
webcam = webcamCapture(filename)

# 啟動子執行緒
webcam.start()

# 暫停1秒，確保影像已經填充
time.sleep(1)

#set GPU
def set_gpus(gpu_index):
    if type(gpu_index) == list:
        gpu_index = ','.join(str(_) for _ in gpu_index)
    if type(gpu_index) ==int:
        gpu_index = str(gpu_index)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
gpu_index = []
for i in range(1,255):
    gpu_index.append(i)
#set_gpus(gpu_index)

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()
        
# Load names of classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#set calculate Lines
calculateRange_y = 1
calculateLine1 = []
calculateLine2 = []
#Initialize count variables
count_Truck1 = 0
count_Car1 = 0
count_Bus1 = 0
count_Truck2 = 0
count_Car2 = 0
count_Bus2 = 0

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Display the counts.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
	
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

#輸入車輛的bbox（bounding box），判斷是否進入南下的熱區，若在熱區，則將該車的ID、bbox、中心點、車子種類、是否已計算過等資訊，放入儲存目前frame data的array中。

def in_range_N2S(bbox):

    #only calculate the cars (from north to south) run across the line and not over Y +- calculateRange_y

    x = bbox[0]

    y = bbox[1]

    w = bbox[2]

    h = bbox[3]

    cx = int(x+(w/2))

    cy = int(y+(h/2))

    if(cx>=calculateLine2[0] and cx<=calculateLine2[1] and \

        cy>calculateLine2[2]-calculateRange_y and cy<=(calculateLine2[3]+calculateRange_y) ):

        return True

    else:

        return False
def in_range_S2N(bbox):

    #only calculate the cars (from south to north) run across the line and not over Y +- calculateRange_y


    x = bbox[0]

    y = bbox[1]

    w = bbox[2]

    h = bbox[3]

    cx = int(x+(w/2))

    cy = int(y+(h/2))

    if(cx>=calculateLine1[0] and cx<=calculateLine1[1] and \

        cy>calculateLine1[2]-calculateRange_y and cy<=(calculateLine1[3]+calculateRange_y) ):

        return True

    else:

        return False



#傳入上下兩個frame的車子bbox list，以上個frame的list為基準，從上一個frame熱區中車子裹，找出與目前frame熱區中距離最短的車子，放入新的list中，我們可以透過該list找到上下兩個frame中每台車子的對應。
obj_target = []
#兩點(x1,y1), (x2, y2) 距離的計算是使用歐式距離（又稱歐幾里得距離），指的就是兩點之間的直線最短距離。
def count_Object(centroid_last, centroid_now):

    distances = []

    for cent_last in centroid_last:

        smallist = 99999.0

        smallist_id = 0

        dist = 0.0
        

        for i, cent_now in enumerate(centroid_now):

            dist = distance(cent_now, cent_last)

            if(dist<=smallist):

                smallist = dist

                smallist_id = i

        distances.append(smallist_id)

    return distances

def counts(last_CENTROIDS,now_CENTROIDS):
	
	for i, now_id in enumerate(obj_target):
	    #print("counts:",i)
	    #if last Y is under the line and now Y is above or on the line, then count += 1

	    print(last_CENTROIDS[i][1], calculateLine2[2], now_CENTROIDS[now_id][1], calculateLine2[2])

	    UP = last_CENTROIDS[i][1]>calculateLine1[2] and now_CENTROIDS[now_id][1]<=calculateLine1[2]

	    DOWN = last_CENTROIDS[i][1]<calculateLine2[2] and now_CENTROIDS[now_id][1]>=calculateLine2[2]

	    if( UP is True):
	    	if(now_LABELS[now_id]=="truck"):
	    		count_Truck1 += 1

	    	elif(now_LABELS[now_id]=="car"):
	    		count_Car1 += 1

	    	elif(now_LABELS[now_id]=="bus"):
	    		count_Bus1 += 1

	    elif( DOWN is True):
	    	if(now_LABELS[now_id]=="truck"):
	    		count_Truck2 += 1

	    	elif(now_LABELS[now_id]=="car"):
	    		count_Car2 += 1

	    	elif(now_LABELS[now_id]=="bus"):
	    		count_Bus2 += 1


# Remove the bounding boxes with low confidence using non-maxima suppression

now_IDs = []	
now_BBOXES = []	
now_CENTROIDS = []	
now_LABELS = []	
now_COUNTED = []
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    global count_Truck1 
    global count_Car1 
    global count_Bus1 
    global count_Truck2 
    global count_Car2 
    global count_Bus2 
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if (classId == 2 or classId == 5 or classId == 7):
            	if confidence > confThreshold:
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
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
        S2N = in_range_S2N(box)
        N2S = in_range_N2S(box)
        classId = classIds[i]
        if(S2N==True or N2S==True):
        	if(S2N==True):
        		if(classId == 7):
        			count_Truck1 += 1

        		elif(classId == 2):
        			count_Car1 += 1

        		elif(classId == 5):
        			count_Bus1 += 1	
        	elif(N2S==True):
        		if(classId == 7):
        			count_Truck2 += 1

        		elif(classId == 2):
        			count_Car2 += 1

        		elif(classId == 5):
        			count_Bus2 += 1	
        	labelName = '%s' % classes[classId]
        	now_IDs.append(i)
        	now_BBOXES.append(box)
        	now_CENTROIDS.append([left,top])
        	now_LABELS.append(labelName)
        	now_COUNTED.append(False)
        	cv.rectangle(frame, (left, top), (left+width, top+height), (0,255,0), 2)
	
    #obj_target =  count_Object(last_CENTROIDS, now_CENTROIDS)	
    #counts(last_CENTROIDS,now_CENTROIDS)
    # Display the counts.
    cv.putText(frame, "truck: ", (0, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv.putText(frame, str(count_Truck1), (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv.putText(frame, "car: ", (0, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv.putText(frame, str(count_Car1), (100, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv.putText(frame, "bus: ", (0, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv.putText(frame, str(count_Bus1), (100, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv.putText(frame, "truck: ", (frameWidth-200, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0,204,0), 2)
    cv.putText(frame, str(count_Truck2), (frameWidth-100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0,204,0), 2)
    cv.putText(frame, "car: ", (frameWidth-200, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0,204,0), 2)
    cv.putText(frame, str(count_Car2), (frameWidth-100, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0,204,0), 2)
    cv.putText(frame, "bus: ", (frameWidth-200, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (0,204,0), 2)
    cv.putText(frame, str(count_Bus2), (frameWidth-100, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (0,204,0), 2)

# Process inputs
winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.mp4"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yolo_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    if cap.isOpened():
        print("read video success!")
    else:
        print("read video failed!")
    outputFile = args.video[:-4]+'_yolo_out_py.mp4'
else:
    # Webcam input
    cap = cv.VideoCapture(filename)
    if cap.isOpened():
        print("read video success!")
    else:
        print("read video failed!")

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:
    
    # get frame from the video
    hasFrame, frame = cap.read()
    videoframeNum += 1;
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]	
    calculateLine1.append(0)
    calculateLine1.append(int(frameWidth/2))
    calculateLine1.append(int(frameHeight*5/12))
    calculateLine1.append(int(frameHeight*5/12+1))

    calculateLine2.append(int(frameWidth/2+1))
    calculateLine2.append(int(frameWidth-1))
    calculateLine2.append(int(frameHeight*5/12))
    calculateLine2.append(int(frameHeight*5/12+1))

# Draw the calculate lines.
    #cv.rectangle(frame, (calculateLine1[0], calculateLine1[2]-calculateRange_y), (calculateLine1[1], calculateLine1[3]+calculateRange_y), (255, 0, 0), 5)
    #cv.rectangle(frame, (calculateLine2[0], calculateLine2[2]-calculateRange_y), (calculateLine2[1], calculateLine2[3]+calculateRange_y), (0, 204, 0), 5)

    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        print("output frames: ",videoframeNum)
		print("— %s seconds —" % (time.time() - start_time))
        cv.waitKey(3000)
        # Release device
        cap.release()
        break

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Write the frame with the detection boxes
    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8))
    else:
        vid_writer.write(frame.astype(np.uint8))

    cv.imshow(winName, frame)

webcam.stop()
print("output frames: ",videoframeNum)


