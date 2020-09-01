import matplotlib.pyplot as plt
import cv2
import numpy as np


class TrafficCounting:

    def show_images(self, images, cmap=None):
        cols = 2
        rows = (len(images)+1)//cols
        
        plt.figure(figsize=(15, 12))
        for i, image in enumerate(images):
            plt.subplot(rows, cols, i+1)
            cmap = 'gray' if len(image.shape)==2 else cmap
            plt.imshow(image, cmap=cmap)
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        plt.show()
		
    # show image
    def cv_show(self, name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # convert rgb to gray
    def convert_gray_scale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # build hotspots
    def filter_region(self, image, vertices1, vertices2):
        """
				剔除掉不需要的地方
		"""
        mask1 = np.zeros_like(image)
        mask2 = np.zeros_like(image)
        if len(mask1.shape) == 2:
            cv2.fillPoly(mask1, vertices1, 255)  # 填充mask
        if len(mask2.shape) == 2:
            cv2.fillPoly(mask2, vertices2, 255)  # 填充mask
        mask1 = cv2.bitwise_or(mask1, mask2)
        return cv2.bitwise_and(image, mask1)

    def select_region(self, image):
        """
				手动选择区域
		"""
        # first, define the polygon by vertices
        rows, cols = image.shape[:2]
        pt_1l = [cols * 0.15, rows * 0.45]
        pt_2l = [cols * 0.15, rows * 0.55]
        pt_3l = [cols * 0.49, rows * 0.55]
        pt_4l = [cols * 0.49, rows * 0.45]
		
        pt_1r = [cols * 0.51, rows * 0.45]
        pt_2r = [cols * 0.51, rows * 0.55]
        pt_3r = [cols * 0.85, rows * 0.55]
        pt_4r = [cols * 0.85, rows * 0.45]

        vertices1 = np.array([[pt_1l, pt_2l, pt_3l, pt_4l]], dtype=np.int32)
        vertices2 = np.array([[pt_1r, pt_2r, pt_3r, pt_4r]], dtype=np.int32)
        
        return self.filter_region(image, vertices1, vertices2)



    def distance(self, cent_now, cent_last):
        return math.sqrt((cent_now[0] - cent_last[0])**2 + (cent_now[1] - cent_last[1])**2)
			
    def count_Object(self, centroid_last, centroid_now):
        distances = []
        for cent_last in centroid_last:
            smallist = 99999.0
            smallist_id = 0
            dist = 0.0

            for id, cent_now in enumerate(centroid_now):
                dist = self.distance(cent_now, cent_last)
                if(dist<=smallist):
                    smallist = dist
                    smallist_id = id
            distances.append(smallist_id)
			
        return distances
		

    def get_frame_data(self, boxes, classes):
        centroids = []
        for box in boxes:
            x1, y1, x2, y2 = box
            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)
            centroids.append((cx, cy))		
        return centroids, boxes, classes        	
		

			
    def count_vehicles(self, frame, now_frame_data, last_frame_data):
        now_CENTROIDS = now_frame_data[0]
        last_CENTROIDS = last_frame_data[0]	
        now_BOXES = now_frame_data[1]
        now_CLASSES = now_frame_data[2]
        frame_h, frame_w = frame.shape[:2]		
        distances = self.count_Object(last_CENTROIDS, now_CENTROIDS)
        #print("Line1_X:" + str(self.calculateLine1[1][0]))
        #print("Line2_X:" + str(self.calculateLine2[0][0]))
        for id, cent_last in enumerate(last_CENTROIDS):
            now_id = distances[id]
            if len(now_CENTROIDS) and len(now_CENTROIDS[now_id]):	
                #print(cent_last)
                #print(now_CENTROIDS[now_id])				
                N2S = cent_last[1]<self.calculateLine1[0][1] and now_CENTROIDS[now_id][1]>=self.calculateLine1[0][1] and cent_last[0]<self.calculateLine1[1][0] and now_CENTROIDS[now_id][0]<self.calculateLine1[1][0]
                S2N = cent_last[1]>self.calculateLine2[0][1] and now_CENTROIDS[now_id][1]<=self.calculateLine2[0][1] and cent_last[0]>self.calculateLine2[0][0] and now_CENTROIDS[now_id][0]>self.calculateLine2[0][0]
                if(N2S is True):
                    #print("N2S")		
                    if(now_CLASSES[now_id]==2 or now_CLASSES[now_id]==5 or now_CLASSES[now_id]==7):
                        self.count_Car1 += 1						

                    x1 = max(0, np.floor(now_BOXES[now_id][0] + 0.5).astype(int))
                    y1 = max(0, np.floor(now_BOXES[now_id][1] + 0.5).astype(int))
                    x2 = min(frame_w, np.floor(now_BOXES[now_id][2] + 0.5).astype(int))
                    y2 = min(frame_h, np.floor(now_BOXES[now_id][3] + 0.5).astype(int))			
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)

                elif(S2N is True):
                    #print("s2N")			
                    if(now_CLASSES[now_id]==2 or now_CLASSES[now_id]==5 or now_CLASSES[now_id]==7):
                        self.count_Car2 += 1	
						
                    x1 = max(0, np.floor(now_BOXES[now_id][0] + 0.5).astype(int))
                    y1 = max(0, np.floor(now_BOXES[now_id][1] + 0.5).astype(int))
                    x2 = min(frame_w, np.floor(now_BOXES[now_id][2] + 0.5).astype(int))
                    y2 = min(frame_h, np.floor(now_BOXES[now_id][3] + 0.5).astype(int))						
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
				

