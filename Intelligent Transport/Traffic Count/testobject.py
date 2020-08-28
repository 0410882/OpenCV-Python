from yoloPydarknet import pydarknetYOLO
import cv2 as cv
import imutils
import time
import argparse
import sys
import numpy as np

yolo = pydarknetYOLO(obdata="../darknet/cfg/coco.data", weights="../darknet/yolov3.weights",

cfg="../darknet/cfg/yolov3.cfg")
#輸入車輛的bbox（bounding box），判斷是否進入南下的熱區，若在熱區，則將該車的ID、bbox、中心點、車子種類、是否已計算過等資訊，放入儲存目前frame data的array中。

def in_range_N2S(bbox,op):

    #only calculate the cars (from north to south) run across the line and not over Y +- calculateRange_y

    x = bbox[0]

    y = bbox[1]

    w = bbox[2]

    h = bbox[3]

    cx = int(x+(w/2))

    cy = int(y+(h/2))
    if op == 1:
    	if(cx>=calculateLine2[0] and cx<=calculateLine2[1] and \

        cy>calculateLine2[2]-calculateRange_y1 and cy<=(calculateLine2[3]+calculateRange_y1) ):

        	return True

    	else:

        	return False
    if op == 2:
    	if(cx>=calculateLine4[0] and cx<=calculateLine4[1] and \

        cy>calculateLine4[2]-calculateRange_y2 and cy<=(calculateLine4[3]+calculateRange_y2) ):

        	return True

    	else:

        	return False
def in_range_S2N(bbox,op):

    #only calculate the cars (from south to north) run across the line and not over Y +- calculateRange_y


    x = bbox[0]

    y = bbox[1]

    w = bbox[2]

    h = bbox[3]

    cx = int(x+(w/2))

    cy = int(y+(h/2))
    if op == 1:
    	if(cx>=calculateLine1[0] and cx<=calculateLine1[1] and \

        cy>calculateLine1[2]-calculateRange_y1 and cy<=(calculateLine1[3]+calculateRange_y1) ):

        	return True

    	else:

        	return False
    if op == 2:
    	if(cx>=calculateLine3[0] and cx<=calculateLine3[1] and \

        cy>calculateLine3[2]-calculateRange_y2 and cy<=(calculateLine3[3]+calculateRange_y2) ):

        	return True

    	else:

        	return False
def postprocess(frame, bbox):
    # Scan through all the bounding boxes output from the network and keep only the
    global count_CarL1
    global count_CarL2
    global count_CarR1
    global count_CarR2
    for i, box in enumerate(bbox):
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            S2N_1 = in_range_S2N(box,1)
            N2S_1 = in_range_N2S(box,1)
            S2N_2 = in_range_S2N(box,2)
            N2S_2 = in_range_N2S(box,2)
            if(S2N_1==True or N2S_1==True or S2N_2==True or N2S_2==True):
                if(S2N_1==True):
                        count_CarL1 += 1
                if(N2S_1==True):
                        count_CarR1 += 1
                if(S2N_2==True):
                        count_CarL2 += 1
                if(N2S_2==True):
                        count_CarR2 += 1
                cv.rectangle(frame, (left, top), (left+width, top+height), (0,255,0), 2)

    # Display the counts.
    cv.putText(frame, "carL1: ", (0, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv.putText(frame, str(count_CarL1), (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv.putText(frame, "carL2: ", (0, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv.putText(frame, str(count_CarL2), (100, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv.putText(frame, "carR1: ", (frameWidth-200, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0,204,0), 2)
    cv.putText(frame, str(count_CarR1), (frameWidth-100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0,204,0), 2)
    cv.putText(frame, "carR2: ", (frameWidth-200, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0,204,0), 2)
    cv.putText(frame, str(count_CarR2), (frameWidth-100, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0,204,0), 2)

#set calculate Lines
calculateRange_y1 = 8
calculateRange_y2 = 1
calculateLine1 = []
calculateLine2 = []
calculateLine3 = []
calculateLine4 = []
#Initialize count variables
count_CarL1 = 0
count_CarL2 = 0
count_CarR1 = 0
count_CarR2 = 0

#start timer
start_time = time.time()

if __name__ == "__main__":

    VIDEO_IN = cv.VideoCapture("output.avi")
    if VIDEO_IN.isOpened():
        print("read video success!")
    else:
        print("read video failed!")
    VIDEO_IN.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    VIDEO_IN.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    # 使用 XVID 編碼
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    outputFile = "yolo_out.avi"
    # 建立 VideoWriter 物件，輸出影片至 output.avi
    # FPS 值為 25.0，解析度為 1280x720
    vid_writer = cv.VideoWriter(outputFile, fourcc, 25.0, (1280, 720))
    #vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(VIDEO_IN.get(cv.CAP_PROP_FRAME_WIDTH)),round(VIDEO_IN.get(cv.CAP_PROP_FRAME_HEIGHT))))
# Get the video writer initialized to save the output video
    isSetLine = False
    frameID = 0
    last_CarL1 = 0
    last_CarL2 = 0
    last_CarR1 = 0
    last_CarR2 = 0
    last_time = start_time
    while True:
        hasFrame, frame = VIDEO_IN.read()
        # Stop the program if reached end of video
        if(isSetLine == False):
            frameHeight = frame.shape[0]
            frameWidth = frame.shape[1]
            calculateLine1.append(0)
            calculateLine1.append(int(frameWidth/2))
            calculateLine1.append(int(frameHeight*7/12))
            calculateLine1.append(int(frameHeight*7/12+1))

            calculateLine3.append(0)
            calculateLine3.append(int(frameWidth/2))
            calculateLine3.append(int(frameHeight*3/8))
            calculateLine3.append(int(frameHeight*3/8+1))

            calculateLine2.append(int(frameWidth/2+1))
            calculateLine2.append(int(frameWidth-1))
            calculateLine2.append(int(frameHeight*7/12))
            calculateLine2.append(int(frameHeight*7/12+1))

            calculateLine4.append(int(frameWidth/2+1))
            calculateLine4.append(int(frameWidth-1))
            calculateLine4.append(int(frameHeight*3/8))
            calculateLine4.append(int(frameHeight*3/8+1))
            isSetLine = True
        if not hasFrame:
            print("Done processing !!!")
            print("— %s seconds —" % (time.time() - start_time))
            cv.waitKey(3000)
            break
        
        yolo.getObject(frame, labelWant=("car","truck","bus"), drawBox=True)
        #print ("Object counts:", yolo.objCounts)
        #yolo.listLabels()
        
        #count process:
        postprocess(frame, yolo.bbox)
        if(time.time() - last_time > 60):
            print("last 60 seconds' carL1 : %s" % (count_CarL1 - last_CarL1))
            print("last 60 seconds' carL2 : %s" % (count_CarL2 - last_CarL2))
            print("last 60 seconds' carR1 : %s" % (count_CarR1 - last_CarR1))
            print("last 60 seconds' carR2 : %s" % (count_CarR2 - last_CarR2))
            last_CarL1 = count_CarL1
            last_CarL2 = count_CarL2
            last_CarR1 = count_CarR1
            last_CarR2 = count_CarR2
            last_time = time.time()
#        print("ID #1:", yolo.list_Label(1))
# Draw the calculate lines.
        cv.rectangle(frame, (calculateLine1[0], calculateLine1[2]-calculateRange_y1), (calculateLine1[1], calculateLine1[3]+calculateRange_y1), (255, 0, 0), 1)
        cv.rectangle(frame, (calculateLine2[0], calculateLine2[2]-calculateRange_y1), (calculateLine2[1], calculateLine2[3]+calculateRange_y1), (0, 204, 0), 1)
        cv.rectangle(frame, (calculateLine3[0], calculateLine3[2]-calculateRange_y2), (calculateLine3[1], calculateLine3[3]+calculateRange_y2), (255, 0, 0), 1)
        cv.rectangle(frame, (calculateLine4[0], calculateLine4[2]-calculateRange_y2), (calculateLine4[1], calculateLine4[3]+calculateRange_y2), (0, 204, 0), 1)

        cv.imshow("Frame", imutils.resize(frame, width=850))
        vid_writer.write(frame)
        #vid_writer.write(frame.astype(np.uint8))
        k = cv.waitKey(1)
        if k == 0xFF & ord("q"):
                VIDEO_IN.release()
                break
