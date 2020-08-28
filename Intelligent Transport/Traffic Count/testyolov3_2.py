from yoloPydarknet import pydarknetYOLO
import cv2
import imutils
import time

yolo = pydarknetYOLO(obdata="../darknet/cfg/coco.data", weights="../darknet/yolov3.weights",

cfg="../darknet/cfg/yolov3.cfg")
#start_time = time.time()

if __name__ == "__main__":

    VIDEO_IN = cv2.VideoCapture("2.avi")

    frameID = 0
    while True:
        hasFrame, frame = VIDEO_IN.read()
        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            print("— %s seconds —" % (time.time() - start_time))
            cv2.waitKey(3000)
            break

        yolo.getObject(frame, labelWant=("car"), drawBox=True)
        print ("Object counts:", yolo.objCounts)
        yolo.listLabels()
#        print("ID #1:", yolo.list_Label(1))
        cv2.imshow("Frame", imutils.resize(frame, width=850))

        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
                VIDEO_IN.release()
                break
