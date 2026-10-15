# -*- coding: utf-8 -*-
"""
苹果成熟度-摄像头实时检测
用法: python detect_camera_apple.py
按 q 退出
"""
import cv2
from ultralytics import YOLO

MODEL_PATH = r"e:\文档\trae\训练极小数据集\runs\detect\apple_ripeness\weights\best.pt"

# 颜色 (BGR): ripe=红, unripe=绿
COLORS = {"ripe": (0, 0, 255), "unripe": (0, 200, 0)}

def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)  # 0=默认摄像头, 不行就试 1
    if not cap.isOpened():
        print("摄像头打开失败, 试试把 0 改成 1")
        return

    print("摄像头已启动, 按 q 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取画面失败")
            break

        result = model.predict(frame, conf=0.7, verbose=False)[0]
        ripe_count = 0
        unripe_count = 0

        for box in result.boxes:
            cls_id = int(box.cls[0])
            name = result.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = COLORS.get(name, (200, 200, 200))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if name == "ripe":
                ripe_count += 1
            else:
                unripe_count += 1

        # 顶部统计条
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(frame, f"Ripe(pickable): {ripe_count}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Unripe(wait): {unripe_count}", (350, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

        cv2.imshow("Apple Ripeness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
