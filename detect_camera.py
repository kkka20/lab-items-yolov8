"""
实时摄像头检测 —— 用微调后的 best.pt 识别 烧杯 / 锥形瓶 / 试管

用法（在项目目录下）:
    python detect_camera.py            # 打开默认摄像头，实时检测
    python detect_camera.py 照片.jpg    # 检测单张图片（如手机拍的实物照）

实时窗口中按 Q 或 ESC 退出。
"""
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "runs" / "lab_items" / "weights" / "best.pt"
CONF = 0.35  # 置信度阈值：误检太多就调高(如0.6)，漏检太多就调低(如0.25)
CAM = 0      # 摄像头编号：默认0打不开就改成1


def main():
    model = YOLO(str(MODEL_PATH))
    target = sys.argv[1] if len(sys.argv) > 1 else None

    # ---------- 模式一：单张图片 ----------
    if target is not None:
        results = model.predict(source=target, conf=CONF, device=0, save=True)
        cv2.imshow("result (按任意键关闭)", results[0].plot())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("带框图已保存到 runs/detect/predict 目录")
        return

    # ---------- 模式二：摄像头实时检测 ----------
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("摄像头打开失败：请检查①是否被其他软件占用 ②是否允许桌面应用使用摄像头 "
              "③把脚本顶部 CAM = 0 改成 1 再试")
        return

    print("摄像头已开启，对准烧杯/锥形瓶/试管，按 Q 或 ESC 退出")
    prev_t = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("警告：读取摄像头画面失败（ok=False）")
            break

        results = model(frame, conf=CONF, device=0, verbose=False)
        boxes = results[0].boxes

        # 终端调试：检测到目标就打印类别+置信度（约每0.5秒一次，避免刷屏）
        now = time.time()
        if now - prev_t > 0.5:
            prev_t = now
            if len(boxes) > 0:
                msg = ", ".join(
                    f"{model.names[int(c)]} {float(x):.2f}"
                    for c, x in zip(boxes.cls, boxes.conf))
                print(f"检出 {len(boxes)} 个目标: {msg}")
            else:
                print("未检出目标（当前阈值 CONF = %.2f）" % CONF)

        annotated = results[0].plot()
        cv2.putText(annotated, f"conf>={CONF} | objects: {len(boxes)}",
                    (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow("lab items detector (Q quit)", annotated)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
