"""
YOLOv8 微调训练脚本
任务：识别 3 种实验室物品（烧杯 beaker / 锥形瓶 erlenmeyer_flask / 试管 test_tube）
运行方式：
    python train.py
训练产物（提交用）：
    runs/lab_items/weights/best.pt   最优权重
    runs/lab_items/results.csv       每个 epoch 的损失与指标
    train_log.txt                    完整训练日志（脚本自动保存，无需额外命令）
"""
import os
import shutil
import sys
from pathlib import Path

# 强制按宽终端渲染，避免 tqdm 把逐轮 loss 数值截断成省略号（须在 import ultralytics 前）
os.environ.setdefault("COLUMNS", "250")
shutil.get_terminal_size = lambda *args, **kwargs: os.terminal_size((250, 60))

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


class Tee:
    """把 stdout/stderr 同时输出到控制台和日志文件。"""

    def __init__(self, log_path: Path):
        self.log = open(log_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, msg):
        self.stdout.write(msg)
        # tqdm 用 \r 刷新进度，写文件时转成换行，保留每轮训练表格
        self.log.write(msg.replace("\r", "\n"))
        self.log.flush()

    def flush(self):
        self.stdout.flush()
        self.log.flush()

    def isatty(self):
        # 伪装成终端，ultralytics 才会打印逐轮表格
        return getattr(self.stdout, "isatty", lambda: False)()


def main():
    sys.stdout = sys.stderr = Tee(ROOT / "train_log.txt")
    # yolov8n.pt：官方 nano 预训练权重，首次运行自动下载（约 6MB）
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(ROOT / "dataset" / "data.yaml"),
        epochs=60,       # 极小数据集 50~80 轮足够，配合早停
        imgsz=640,       # 输入图片分辨率
        batch=16,        # RTX 4060 8GB 无压力；若显存报错就降到 8
        device=0,        # 0 = 第一块 NVIDIA GPU；无独显时改为 "cpu"
        workers=4,
        patience=20,     # 连续 20 轮无提升自动早停
        project=str(ROOT / "runs"),
        name="lab_items",
        exist_ok=True,
    )

    # 训练结束后在验证集上评估，mAP 会打印到控制台（属于训练日志的一部分）
    metrics = model.val()
    print(f"mAP50    = {metrics.box.map50:.4f}")
    print(f"mAP50-95 = {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
