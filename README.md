# Heartbeat-sequence-prediction
天池上的一场长期赛，非常简单朴素的实现， ~~长期赛榜单第21名（432.47分）（暂时）~~ 长期赛榜单第8名（258.9817分）

主要是用来练一练手，单独手动完成了所有工作。没有使用一些trick，例如K折交叉验证、模型融合、启发式寻参等常用trick（人太懒了），有兴趣可以继续用此模型来优化

使用的分类模型参考了 ~~**DenseNet121**~~   **DenseNet264**（二维卷积改为了一维卷积，卷积核大小不变），识别成功率为：~~98.9%~~ A榜数据集为99.4%

## 使用教程

可以直接用DenseNet.ipynb来训练和测试（需要自己去天池比赛页面下载训练和测试文件：trian.csv和testA.csv）；也可以直接下载预训练模型

长期赛的提交结果为submit.csv ~~和submit1.csv~~

天池比赛地址：https://tianchi.aliyun.com/competition/entrance/531883/introduction

## 预训练模型

百度网盘地址为：https://pan.baidu.com/s/12ODnFTmBcXaWETdHI-6Ohg 
提取码：m96j 

## 重构版本

更新了一份重构版本，代码比之前看起来舒服了一点（可能
