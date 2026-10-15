# -*- coding: utf-8 -*-
r"""
苹果成熟度数据集划分脚本
输入: C:\Users\lenovo\Desktop\apple  (YOLO 平铺结构: images/ + labels/)
输出: 项目目录下 dataset_apple/  (train/val 各含 images/ + labels/ + data.yaml)
用法: python split_apple.py
"""
import os, random, shutil, sys

SRC = r"C:\Users\lenovo\Desktop\apple"
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_apple")
RATIO = 0.8          # 训练集比例
SEED = 42            # 随机种子(保证可复现)
NAMES = ["ripe", "unripe"]   # 类别顺序对应 classes.txt

img_dir = os.path.join(SRC, "images")
lab_dir = os.path.join(SRC, "labels")

def main():
    random.seed(SEED)
    images = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"总图片数: {len(images)}")

    # 校验配对
    ok = []
    miss = 0
    for f in images:
        stem = os.path.splitext(f)[0]
        if os.path.exists(os.path.join(lab_dir, stem + ".txt")):
            ok.append(f)
        else:
            miss += 1
    print(f"配对成功: {len(ok)}, 缺标签: {miss}")

    # 打乱 + 划分
    random.shuffle(ok)
    n_train = int(len(ok) * RATIO)
    train_imgs = ok[:n_train]
    val_imgs = ok[n_train:]
    print(f"train={len(train_imgs)}  val={len(val_imgs)}")

    # 重建输出目录
    if os.path.exists(DST):
        shutil.rmtree(DST)
    for split in ["train", "val"]:
        os.makedirs(os.path.join(DST, split, "images"))
        os.makedirs(os.path.join(DST, split, "labels"))

    # 复制
    for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
        for f in imgs:
            stem = os.path.splitext(f)[0]
            shutil.copy2(os.path.join(img_dir, f), os.path.join(DST, split, "images", f))
            shutil.copy2(os.path.join(lab_dir, stem + ".txt"), os.path.join(DST, split, "labels", stem + ".txt"))

    # 写 data.yaml (用绝对路径, 避免 ultralytics 解析歧义)
    yaml = (
        f"path: {DST}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(NAMES)}\n"
        f"names: {NAMES}\n"
    )
    with open(os.path.join(DST, "data.yaml"), "w", encoding="utf-8") as fp:
        fp.write(yaml)
    print(f"划分完成 -> {DST}")
    print("data.yaml:")
    print(yaml)

if __name__ == "__main__":
    main()
