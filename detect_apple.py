# -*- coding: utf-8 -*-
"""
苹果成熟度检测-效果验证脚本
用训练好的 best.pt 对验证集图片做预测, 保存带框结果图
用法: python detect_apple.py
"""
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"e:\文档\trae\训练极小数据集\runs\detect\apple_ripeness\weights\best.pt")
    results = model.predict(
        source=r"E:\apple_dataset\val\images",
        conf=0.25,
        save=True,
        name="apple_detect",
    )
    print(f"检测完成, 结果保存到 runs/detect/apple_detect/")
