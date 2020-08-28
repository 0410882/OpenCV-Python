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
