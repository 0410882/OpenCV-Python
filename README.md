# Cross disciplinary Project I —— Intelligent Traffic Light 智慧交通信號燈

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
[車流計算demo2](https://youtu.be/TYbNcwF694Y)  
### Reference 參考資料
[YOLOv4]( https://github.com/AlexeyAB/darknet)  
[如何計算道路及十字路口的車流](https://chtseng.wordpress.com/2018/11/03/%E5%A6%82%E4%BD%95%E8%A8%88%E7%AE%97%E9%81%93%E8%B7%AF%E5%8F%8A%E5%8D%81%E5%AD%97%E8%B7%AF%E5%8F%A3%E7%9A%84%E8%BB%8A%E6%B5%81/)  
[Keras-YOLOv4]( https://github.com/miemie2013/Keras-YOLOv4)  

## Second Part —— Traffic Light Control 信號燈控制
#### 目的：
> 通過信號燈的控制使路口總的車輛等待時間最短（先考慮只有一個路口的情況）
#### 模擬環境建立：
* 考慮汽車啟動時的加速度，先不考慮剎車需要的時間；
* 假設汽車最後都會以最高限速行駛；
* 假設汽車等待綠燈或行駛時，互相之間會隔開一定的安全距離；
* 假設汽車們的車長一致；
* 假設汽車從紅燈變綠燈時都會同時啟動；

第一次嘗試思路：
每過一個單位時間，計算一次東西和南北向車輛的數量，如果綠燈方向車
輛較少，則改變信號燈，或是綠燈時間太久也會改變信號燈。

#### 第一次嘗試架構圖：
![Structure](https://github.com/Lvma-0323/Cross-disciplinary-Projects/blob/master/structure.PNG)
#### 結果討論：
從總的等待時間這個衡量標準來看，設置最長綠燈時間沒有任何效果，不
過如果輸出每個單位時間道路上的車輛狀況，可以發現新增車流量較少的那個
方向會有一些車輛很久都通過不了。
因為考慮了汽車啟動到最高速度需要一定的時間，所以並不是每過 1 個單
位時間就決定是否變燈時等待時間最短，以上述特定條件下的模擬結果來看，
大約間隔 6～10 個單位時間再決定比較好。經測試，調整車輛加速度、最高限
速、車間距的大小都不會影響到上述結論。
#### 接下來的目標：
> 增加模擬環境的複雜度;
> 想一個更好的算法來決定是否改變信號燈;
#### 模擬環境建立：
* 考慮汽車啟動時的加速度，先不考慮剎車需要的時間；
* 假設汽車最後都會以最高限速行駛；
* 假設汽車等待綠燈或行駛時，互相之間會隔開一定的安全距離；
* 假設汽車們的車長一致；
* 假設汽車從紅燈變綠燈時都會同時啟動；

新增細節：
紅燈時，排在最前面的尚未通過的車輛會在路口停下，後續車輛會依照理
論間距依序停好。
若是 update 的時間間隔過長，可能導致計算剛啟動的車輛的車速時，一下
就超過最高限速了。所以計算時改為先計算車速，如車速超高最高限速，就以
最高限速的速度計算開出去的距離(假設車輛剛啟動時是勻加速運動，到達最高
限速後是勻速運動)
#### 結果討論：
下圖是在同樣 seed 隨機出的數據下，不同的道路容量以及不同的決定紅綠燈變
化單位時間的情況下的車輛總等待時間狀況。
從以下數據可以看出，如要車輛總等待時間最少，決定紅綠燈變化的間隔時間
大概是 15s 左右比較好。
![Result](https://github.com/Lvma-0323/Cross-disciplinary-Projects/blob/master/result1.PNG)

下面三圖是不同 seed 時(即道路上初始和新增的車輛數量不同)，道路容量和最
短等待時間時所用的決定紅綠燈變化的間隔時間的關係。
大概來說，道路的容量越大，使車輛總等待時間最短的紅綠燈變換間隔也越大
![Result](https://github.com/Lvma-0323/Cross-disciplinary-Projects/blob/master/result2.PNG)
![Result](https://github.com/Lvma-0323/Cross-disciplinary-Projects/blob/master/result3.PNG)
![Result](https://github.com/Lvma-0323/Cross-disciplinary-Projects/blob/master/result4.PNG)

至於每輛車的最大等待時長，和兩個方向上新增車輛的多少有關
若是每个方向上等概率的新增車輛，則每个方向上車子的最大等待時長也會比
較平均
<img src = "https://github.com/Lvma-0323/Cross-disciplinary-Projects/blob/master/situation1.PNG" width = "375">
![Result](https://github.com/Lvma-0323/Cross-disciplinary-Projects/blob/master/situation1.PNG)
![Result](https://github.com/Lvma-0323/Cross-disciplinary-Projects/blob/master/output1.PNG)

若是某個方向的車輛新增比較多(如東西向)，則相對應的方向(如南北向)車子的
等待時間就會比較長
![Result](https://github.com/Lvma-0323/Cross-disciplinary-Projects/blob/master/situation2.PNG)
![Result](https://github.com/Lvma-0323/Cross-disciplinary-Projects/blob/master/output2.PNG)

若是只有一個方向新增車輛，則會出現對應方向的車輛一直無法通過的情況
![Result](https://github.com/Lvma-0323/Cross-disciplinary-Projects/blob/master/situation3.PNG)
![Result](https://github.com/Lvma-0323/Cross-disciplinary-Projects/blob/master/output3.PNG)
