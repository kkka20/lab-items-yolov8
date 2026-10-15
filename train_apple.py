# -*- coding: utf-8 -*-
"""
苹果成熟度检测训练脚本
数据: dataset_apple (ripe=0, unripe=1)
用法: python train_apple.py
"""
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(
        data="E:/apple_dataset/data.yaml",
        epochs=60,
        imgsz=640,
        batch=16,
        device=0,
        patience=20,
        workers=0,
        amp=False,
        exist_ok=True,
        name="apple_ripeness",
    )
