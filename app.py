"""
网页版实时检测 —— 浏览器打开页面调用摄像头，YOLOv8 在服务端推理

启动：
    pip install -r requirements.txt
    python app.py
然后浏览器访问 http://127.0.0.1:5000
"""
import base64

import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# 优先用 GPU，没有 GPU 自动回退 CPU
DEVICE = 0 if torch.cuda.is_available() else "cpu"
model = YOLO("best.pt")
model.to(DEVICE)
CONF = 0.35  # 置信度阈值


@app.route("/")
def index():
    return render_template("index.html", device="GPU" if DEVICE == 0 else "CPU")


@app.route("/detect", methods=["POST"])
def detect():
    # 前端传来 base64 的摄像头帧
    data = request.get_json()["image"].split(",", 1)[-1]
    raw = np.frombuffer(base64.b64decode(data), np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    results = model(frame, conf=CONF, verbose=False)[0]
    detections = [{
        "name": model.names[int(box.cls)],
        "conf": round(float(box.conf), 3),
        "box": [round(v) for v in box.xyxy[0].tolist()],
    } for box in results.boxes]

    # 画好框的帧编码回 base64 给前端显示
    ok, buf = cv2.imencode(".jpg", results.plot(), [cv2.IMWRITE_JPEG_QUALITY, 80])
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    return jsonify({"image": img_b64, "detections": detections})


if __name__ == "__main__":
    print("浏览器访问 http://127.0.0.1:5000 （Ctrl+C 停止）")
    app.run(host="127.0.0.1", port=5000, threaded=True)
