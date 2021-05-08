# Megengine Object Detection Models

## 介绍

本目录包含了采用MegEngine实现的如下经典网络结构，并提供了在COCO2017数据集上的完整训练和测试代码：

- [RetinaNet](https://arxiv.org/abs/1708.02002)
- [Faster R-CNN](https://arxiv.org/abs/1612.03144)
- [FreeAnchor](https://arxiv.org/abs/1909.02466)
- [FCOS](https://arxiv.org/abs/1904.01355)
- [ATSS](https://arxiv.org/abs/1912.02424)

## 模型说明
##### 本baseline模型采用的是“检测-跟踪”两部分分离的跟踪架构，在此对目录结构进行简要的介绍和说明。
##### 在检测方面，baseline使用的天元框架实现的经典目标检测结构：
- [RetinaNet](https://arxiv.org/abs/1708.02002)
- [Faster R-CNN](https://arxiv.org/abs/1612.03144)
- [FreeAnchor](https://arxiv.org/abs/1909.02466)
- [FCOS](https://arxiv.org/abs/1904.01355)
- [ATSS](https://arxiv.org/abs/1912.02424)

##### 在跟踪方面，baseline使用的是卡尔曼滤波和匈牙利匹配算法：
- [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763)

##### 项目基本文件结构如下：
```
/path/to/
    |->dataset 
    |->workspace
    |    |data                                        ：用于解压和存放比赛数据，后续跟踪的结果也将存储在该目录下
    |    |    |coco             
    |    |    |    |annotations                       ：step1中生成的coco格式的数据
    |    |    |train                                  ：track1,2,3,5,11 为MOT格式的标注数据，用于训练。在step1中，将每一段数据从中间划分成训练集和验证集，用于参赛选手验证模型
    |    |    |    |track1
    |    |    |    |      |img                        :图像文件夹
    |    |    |    |      |gt                         : gt.txt：MOT格式的标注数据
                                                      : gt_1_train_half.txt：在该段数据中，用于训练的 MOT格式标注数据
                                                      : gt_1_val_half.txt：在该段数据中，用于验证的 MOT格式标注数据
                                                      : gt_5_train_half.txt：在该段数据中，用于验证的 MOT格式标注数据 (在提供的标准数据中，为25帧/秒，该文件中抽取为 5帧/秒)
                                                      : gt_5_val_half.txt：在该段数据中，用于验证的 MOT格式标注数据   (在提供的标准数据中，为25帧/秒，该文件中抽取为 5帧/秒)
                                                      : track_1_val_half.txt：fish_tracker 生成的MOT格式标注数据
                                                      : track_5_val_half.txt：fish_tracker 生成的MOT格式标注数据 (在提供的标准数据中，为25帧/秒，该文件中抽取为 5帧/秒)                                                      
    |    |    |test                                   ：track 4,10 用于初赛测试数据。在step1中，测试结果需提交至平台，用于得分排名计算
    |    |    |    |track10|img                       : 图像文件夹
    |    |detector                                    ：目标检测器相关实现代码
    |    |tools                                       ：在代码运行过程中可能需要用的到一些功能函数
    |    |tracker                                     ：目标跟踪器相关实现代码
    |    det_inference.ipynb                          ：目标检测图片推理入口文件
    |    det_test.ipynb                               ：目标检测精度测试入口文件
    |    det_train.ipynb                              ：目标检测图片训练入口文件
    |    fish_tracker.ipynb                           ：鱼类多目标跟踪入口文件
    |    tracker_eval.ipynb                           ：用于计算多目标跟踪评价指标的文件，
                                                        最后返回结果不抽帧与抽5帧的调和平均得分。
    |    ipynb_importer.py                            ：在Jupyter平台中调用其他文件的函数
```
### 运行步骤
#### 1.  格式化数据
为使用天元目标检测框架，需将MOT17格式数据转换成为COCO格式数据，转换代码见```data/convert_mot_to_coco.py```
#### 2. 查看标注数据
```data/visulization.py```
#### 3. 训练检测器
```det_train.py```
#### 4. MOT跟踪
```fish_tracker.py```
#### 5. 跟踪指标计算
```python tracker_eval.py --track_result_path '/path/to/track_result_filepath --gt_path '/path/to/gt_filepath```

```
/path/to/
    |->track_result_filepath 
    |    |{seq_name}_track_s1_test_no1.txt     ：模型跟踪结果（不抽帧的情况）
    |    |{seq_name}_track_s5_test_no1.txt     ：模型跟踪结果（抽帧的情况）
    |->gt_filepath
    |    |{seq_name}_gt_s1_test+no1.txt        ：真值数据（不抽帧的情况）
    |    |{seq_name}_gt_s5_test+no1.txt        ：真值数据（抽帧的情况）
```

调和平均比重：w_MOTA=1, w_IDF1=1

gt:

    original          dataset get scores: 2.002636991145807
    selected-5-frames dataset get scores: 2.0
    
example:

    original          dataset get scores: 1.596286314214157
    selected-5-frames dataset get scores: 0.8543985845843052
                                                            