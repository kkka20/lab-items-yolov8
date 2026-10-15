---
title: 130 张图玩转 YOLOv8：实验室物品检测从零微调与网页部署实战
slug: yolov8-lab-items-detection
description: 用 130 张图片微调 YOLOv8n 识别烧杯、锥形瓶、试管，从数据标注、训练评估到摄像头与网页部署的完整踩坑记录。
publishedAt: 2026-10-02
category: 机器人
tags: [YOLOv8, 目标检测, ROS2, 深度学习]
draft: false
---

# 130 张图玩转 YOLOv8：实验室物品检测从零微调与网页部署实战

> 一个机器人方向、正在学习 ROS2 的学生，为了给机器人视觉节点打基础，从零走通「数据标注 → 微调训练 → 摄像头/网页部署」全流程的真实记录。

2026.10.02 · 个人项目实战 · 最终验证集 mAP50 = 0.978

## 一、为什么做这个

最近在学 ROS2，方向是机器人。机器人要通过摄像头"看懂"周围环境，视觉感知是绕不开的一环，而 YOLO 正是 ROS 生态里做目标检测最常用的模型之一。为了给以后写 ROS2 视觉节点打基础，我给自己定了个小目标：**让电脑认出烧杯、锥形瓶、试管三种实验室物品**，并且要用真实器材在摄像头前检验效果，而不是只在数据集里刷分。

一开始我以为这种项目要很大的数据量、很贵的显卡，做完才发现：YOLOv8n + 迁移学习，130 张图、一张笔记本 4060，训练只花了两分钟。真正花时间的不是训练，而是**数据和踩坑**。这篇文章把完整流程、结果解读和四个坑都记录下来，项目代码已开源：https://github.com/kkka20/lab-items-yolov8

## 二、最终效果先看为敬

验证集（模型训练时没见过的 26 张图、77 个目标）上的实测：

![检测效果-烧杯](https://raw.githubusercontent.com/kkka20/lab-items-yolov8/main/runs/lab_items/predict_val/beaker_41a5ccb6-beaker_2.jpg)

![检测效果-锥形瓶](https://raw.githubusercontent.com/kkka20/lab-items-yolov8/main/runs/lab_items/predict_val/erlenmeyer_flask_03cce82a-flask_25.jpg)

![检测效果-试管](https://raw.githubusercontent.com/kkka20/lab-items-yolov8/main/runs/lab_items/predict_val/test_tube_0b71b0b3-testtube_14.jpg)

最终指标：

| 类别 | 精确率 P | 召回率 R | mAP50 |
| --- | --- | --- | --- |
| beaker 烧杯 | 0.990 | 1.000 | 0.995 |
| erlenmeyer_flask 锥形瓶 | 1.000 | 0.987 | 0.995 |
| test_tube 试管 | 0.989 | 0.773 | 0.944 |
| **总体** | **0.993** | **0.920** | **0.978** |

除了 OpenCV 弹窗，我还做了一个 **Flask 网页版**：浏览器打开页面调用摄像头，画面上实时显示检测框和置信度，演示时比弹窗直观得多。

## 三、整体流程

```
收集图片 → LabelImg 标注 → 数据质检 → 按 8:2 划分 → YOLOv8n 迁移学习 → 评估解读 → 部署实测
```

技术栈很简单：`ultralytics`（YOLOv8 官方库）+ `opencv-python` + `flask`，一张 RTX 4060 Laptop。

## 四、第一步：准备数据集

### 4.1 收集图片

三类各下载 30~50 张，共 130 张。这里有一个**后来让我付出代价的教训**：为了省事，图片大部分是电商商品主图（棚拍、灰白背景、带广告文字）。它让验证集分数非常漂亮，但真实器材面对摄像头时置信度明显下降——这个后面细说。

**建议**：如果你要做实际能用的检测器，从第一天起就要混入真实场景照片（手持、实验台、多个混放、不同光线）。

### 4.2 认识 YOLO 标签格式

每张图片配一个同名 `.txt`，每行一个目标，5 个数字：

```text
类别编号  中心点x  中心点y  宽  高
```

后四个数都是 0~1 的归一化值（除以图片宽高），所以换分辨率也不影响。例如：

```text
0 0.512 0.488 0.305 0.612
```

### 4.3 LabelImg 标注

LabelImg 左侧栏切到 **YOLO 模式**，输出的就是上面这种 txt。先写好 `classes.txt` 固定类别顺序：

```text
beaker
erlenmeyer_flask
test_tube
```

标注三原则：框紧贴物体边缘；图里每个出现的目标都要框；严重遮挡、看不清的宁可不框。试管架上一排十根试管，就老老实实画十个框。

### 4.4 数据质检（小数据集最关键的一步）

数据量越小，脏数据的伤害越大。我写脚本自动检查了：图文是否一一配对、标签数值是否合法（5 个字段、0~1 范围、框不越界）、有没有完全重复的图片。结果抓出三类问题：

1. **6 对完全重复的图片**（同一张广告图下载了两次）。重复图最危险的地方在于：万一一张进训练集、一张进验证集，mAP 会虚高，等于自己骗自己。
2. **17 张试管图标错**：斜放的试管被框成了竖着的，直接删掉这批。
3. **越界框**：小试管密集时框画出了图片边界，脚本裁剪到合法范围。

质检后剩 130 张：烧杯 50、锥形瓶 50、试管 30。

### 4.5 划分数据集

脚本 `split_dataset.py` 做了四件事：统一类别编号（烧杯=0 / 锥形瓶=1 / 试管=2，标注时写的编号不一致也没关系）、文件名加类别前缀防重名、按 8:2 **分层划分**（保证每类在训练/验证集比例一致）、生成 `data.yaml`：

```yaml
path: dataset
train: images/train
val: images/val

names:
  0: beaker
  1: erlenmeyer_flask
  2: test_tube
```

最终 104 张训练、26 张验证。

## 五、第二步：训练（核心代码只有 10 行）

用官方最轻量的 yolov8n（约 300 万参数、6MB），在预训练权重上做迁移学习——底层特征（边缘、纹理、形状）直接继承 COCO，只需要让检测头学会三个新类别：

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # 首次运行自动下载约 6MB 的预训练权重

model.train(
    data="dataset/data.yaml",
    epochs=60,       # 小数据集 50~80 轮足够，配合早停
    imgsz=640,       # 输入分辨率
    batch=16,        # 显存不够就降到 8
    device=0,        # 没有 N 卡改成 "cpu"
    patience=20,     # 连续 20 轮无提升自动停止
    project="runs",
    name="lab_items",
)
```

RTX 4060 上训练**约 2 分钟**。我在脚本里还加了个 `Tee` 类，把终端输出同步写进 `train_log.txt`（tqdm 的 `\r` 刷新要转成换行，否则日志里只剩一行），这样训练日志自动落盘，不用每次记得重定向命令。

## 六、第三步：看懂训练输出的一堆图

训练完 `runs/lab_items/` 里会生成一堆文件，这是我以前最懵的部分，逐个解释。

**1. results.png —— 训练体检报告**

![results 训练曲线](https://raw.githubusercontent.com/kkka20/lab-items-yolov8/main/runs/lab_items/results.png)

左列三条 loss（box 定位误差、cls 分类误差、dfl 分布误差）应该**稳步下降**；右列的 mAP 应该稳步上升。如果 loss 不降、暴涨或变成 NaN，就是训练崩了。

**2. confusion_matrix_normalized.png —— 混淆矩阵，看模型"把谁认成了谁"**

![混淆矩阵](https://raw.githubusercontent.com/kkka20/lab-items-yolov8/main/runs/lab_items/confusion_matrix_normalized.png)

对角线越亮越好（自己就是自己），非对角线亮代表混淆，背景列（background）表示漏检。

**3. BoxPR_curve.png —— P-R 曲线**

![PR曲线](https://raw.githubusercontent.com/kkka20/lab-items-yolov8/main/runs/lab_items/BoxPR_curve.png)

解释两个最容易混的指标：

- **精确率 P（Precision）**：模型报出的目标里，有多少是真的。P 低 = 误检多（把试管架认成烧杯）。
- **召回率 R（Recall）**：真实目标里，有多少被找出来了。R 低 = 漏检多（试管在画面里却没框）。
- **mAP50**：IoU 阈值 0.5 时 P-R 曲线下面积，综合分，0~1 越接近 1 越好。

我的结果里有个明显的短板：**试管召回率只有 0.773**。原因很清楚——试管样本最少（30 张），又删掉了 17 张错标难例，模型于是学会了"对不确定的小目标保守处理"。这不是玄学，数据分布直接决定了模型的短板。

## 七、第四步：部署，用真家伙检验

### 7.1 OpenCV 摄像头实时检测

```python
import cv2
from ultralytics import YOLO

model = YOLO("runs/lab_items/weights/best.pt")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ok, frame = cap.read()
    results = model(frame, conf=0.35, verbose=False)
    cv2.imshow("detector", results[0].plot())
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        break
```

### 7.2 Flask 网页版（演示效果更好）

OpenCV 弹窗比较朴素，网页版的思路是：**浏览器负责采集摄像头帧，POST 给后端，YOLO 推理画框后回传，页面显示**。后端核心代码：

```python
@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json()["image"].split(",", 1)[-1]
    raw = np.frombuffer(base64.b64decode(data), np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    results = model(frame, conf=0.35, verbose=False)[0]
    detections = [{
        "name": model.names[int(box.cls)],
        "conf": round(float(box.conf), 3),
        "box": [round(v) for v in box.xyxy[0].tolist()],
    } for box in results.boxes]

    ok, buf = cv2.imencode(".jpg", results.plot(), [cv2.IMWRITE_JPEG_QUALITY, 80])
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    return jsonify({"image": img_b64, "detections": detections})
```

页面每 200ms 抓一帧发请求（约 5 FPS），用上一帧没推理完就跳过的策略避免帧堆积。`python app.py` 后浏览器打开 `http://127.0.0.1:5000` 即可（注意浏览器只允许 localhost/HTTPS 页面调用摄像头）。

### 7.3 真实器材测试的落差

这是整个项目给我上的最深刻的一课：广告图训练的模型，面对真实器材时置信度从 0.95 掉到 0.5~0.7，还出现过把蓝色试管架认成烧杯的情况。应对办法：

- 阈值 `conf` 按场景调：误检多调高到 0.6，漏检多调低到 0.25；
- 物品离镜头 30~50cm、占画面 1/4 以上、背景简洁、光线充足；
- 根治办法只有一个：**补真实场景照片重训**。分数好看 ≠ 真的能用，验证集如果和训练集同分布，高 mAP 是会骗人的。

## 八、踩坑记录

1. **预训练权重下载超时**：GitHub 连接不稳定，反复 `Download failure, retrying`。重试后成功，且权重只下载一次、本地有缓存，之后训练不再联网。
2. **`can't open file 'train.py'`**：终端停在用户目录，脚本在项目盘。先 `cd` 到项目目录再运行；IDE 里右键文件夹「在集成终端中打开」最省心。
3. **摄像头弹窗全黑**：一度怀疑是代码问题，用 Windows 自带「相机」App 一测也是黑的——最终发现是笔记本的**镜头物理挡片**没拨开。排查硬件问题要先用系统自带工具隔离，别上来就改代码。
4. **data.yaml 写死绝对路径**：最初 yaml 里是我电脑的绝对路径，这种仓库一换机器训练就报错。改成相对路径后，别人 clone 即可复现。

## 九、收获与下一步

**收获：**

- 完整走通了目标检测工程闭环，明白"训练只占 20%，数据质量和结果分析才是大头"；
- 学会了读 loss 曲线、混淆矩阵、P/R/mAP，能对着指标说出模型的短板和原因；
- 理解了一个通用的视觉部署架构：采集帧 → 模型推理 → 解析框 → 画框/发布，OpenCV 版和网页版只是"帧从哪来、结果给谁看"的区别。

**下一步：** 补真实场景数据重训，把泛化能力补上；然后把 `best.pt` 封装成 ROS2 Python 节点——订阅相机图像话题、推理、发布检测结果话题。回头看，这跟 Flask 网页版是同一个套路，只不过传输层从 HTTP 换成了 ROS2 话题。

完整代码、130 张数据集、训练日志和权重都在仓库里，欢迎 star 和交流：

**https://github.com/kkka20/lab-items-yolov8**
