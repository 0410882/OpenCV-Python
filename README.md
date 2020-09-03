# OpenCV-Python project —— Intelligent Traffic Light 智慧交通信號燈

#### Goal 目的：
> By setting up cameras at the intersection, the traffic flow in various directions is calculated, so as to adjust the length of time the traffic lights are in different directions and improve traffic congestion.
> 通過架設在十字路口的攝影機，計算出各個方向的車流量，以此調控交通信號燈在不同方向上的時間長短，改善交通擁堵的情況。

#### Flow Chart 流程圖：
![FlowChart](/FlowChart.jpg)

## First Part —— Traffic Count 車流計算
### Environment configuration 環境配置
> my environment : Windows 10 + Python 3.7.7 + OpenCV 4.4.0 + Tensorflow 2.1.0 + YOLOv4
### How to use on the command line
> python demo_video.py --video=car.mp4
### What improved 改進部分
* 影像處理：   
我們不需要針對整張圖計算車子流量及方向，因此在用YOLO 偵測前，將取得的frame中熱區以外的區域都覆蓋掉，以提升檢測的速度
* 車輛偵測定位及分類:  
從YOLOv3升級為YOLOv4，直接使用YOLO提供的pre-trained yolov4.weights來進行車輛的定位偵測及分類。因為這個pre-trained model是使用Coco dataset所訓練，可偵測多達80種物件，其中也包含了數種車輛類型，如：car、truck、bus、bicycle、motorbike等，打算自行訓練只辨識車輛的模型，不過需要用到GPU才行。
YOLOv3和YOLOv4速度對比如下圖：
![YOLOv3 and YOLOv4](https://user-images.githubusercontent.com/4096485/82835867-f1c62380-9ecd-11ea-9134-1598ed2abc4b.png)
* 提高偵測的精度：   
當我們得到車輛的類型以及位置之後，接著，我們將每一個frame取得的車輛與上一個frame的車輛進行比對，計算該車中心點與上一個frame所有車輛的中心點距離最短是那一台。
由於我們的熱區是定義於道路中間位置，因此理論上該熱區不會有突然新出現的車輛，每一台車應該能找到其一個frame的所在位置。
經由上下frame得到該車的行徑方向，並將該方向的車輛數目加1。
### Sample output video 範例輸出視頻
[車流計算demo1](https://youtu.be/lHy2d0_w_XU)
[車流計算demo2]（https://youtu.be/TYbNcwF694Y）
### Reference 參考資料
[YOLOv4]( https://github.com/AlexeyAB/darknet)  
[如何計算道路及十字路口的車流](https://chtseng.wordpress.com/2018/11/03/%E5%A6%82%E4%BD%95%E8%A8%88%E7%AE%97%E9%81%93%E8%B7%AF%E5%8F%8A%E5%8D%81%E5%AD%97%E8%B7%AF%E5%8F%A3%E7%9A%84%E8%BB%8A%E6%B5%81/)  
[Keras-YOLOv4]( https://github.com/miemie2013/Keras-YOLOv4)  

## Second Part —— Traffic Light Control 信號燈控制
undone

