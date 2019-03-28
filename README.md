# -Jinnan-Faster-R-CNN-tensorflow
# [津南数字制造算法挑战赛](https://tianchi.aliyun.com/competition/entrance/231703/introduction) Faster R-CNN
---
## 开发环境
- 基于Faster R-CNN 实现
- win10 + python3.5 + tensorflow-gpu1.5.0
- cuda9.0 + cndnn7.0.5

## 数据集
- 格式兼容coco数据集
- 训练图片共4000张，其中normal2540,restricted1460

## 运行结果
- 训练速度：每张图片的训练时间约1.2s
- 检测速度：每张图片约0.5s
- mAP：训练10000轮的模型，IOU阈值0.05/0.5/0.95的平均mAP约为0.574
