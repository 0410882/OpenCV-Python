from __future__ import division
import matplotlib.pyplot as plt
import os
import glob
from TrafficCounting import TrafficCounting

cwd = os.getcwd()


def img_process(my_test_images, tf_count):
     #灰度图
    gray_images = list(map(tf_count.convert_gray_scale, my_test_images))
    tf_count.show_images(gray_images)
     #选择区域
    roi_images = list(map(tf_count.select_region, gray_images))
    tf_count.show_images(roi_images)


if __name__ == '__main__':
    test_images = [plt.imread(path) for path in glob.glob('test_images/*.png')]
    tf_count = TrafficCounting()
    #tf_count.show_images(test_images)
    img_process(test_images, tf_count)
    