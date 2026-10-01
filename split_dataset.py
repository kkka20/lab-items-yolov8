"""
三类数据合并 + 训练/验证集划分脚本
数据源（桌面上三个已质检的文件夹）：
    训练图片/beaker   -> 类别 0  beaker
    训练图片/flask    -> 类别 1  erlenmeyer_flask
    训练图片/shiguan  -> 类别 2  test_tube（只取主 images/labels，不含 _relabel）
动作：
    1. 逐张校验图片与 txt 配对、框数值合法；
    2. 把每个标签里的类别编号统一重映射为 0/1/2（源文件里写 0 也没关系）；
    3. 每类内部按 8:2 分层划分（保证验证集三类都有），复制到 dataset/；
    4. 生成 dataset/data.yaml。
运行：python split_dataset.py
"""
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DESKTOP_DATA = Path(r"C:\Users\lenovo\Desktop\训练图片")
# (源文件夹, 目标类别编号, 类别名)
SOURCES = [
    ("beaker", 0, "beaker"),
    ("flask", 1, "erlenmeyer_flask"),
    ("shiguan", 2, "test_tube"),
]
VAL_RATIO = 0.2
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_valid_boxes(txt: Path, class_id: int):
    """读取并重映射类别；返回 (行列表, 问题列表)，顺带做数值校验。"""
    rows, problems = [], []
    for i, line in enumerate(txt.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            problems.append(f"{txt.name} 第{i}行字段数={len(parts)}")
            continue
        try:
            _, x, y, w, h = map(float, parts)
        except ValueError:
            problems.append(f"{txt.name} 第{i}行有非数字内容")
            continue
        if not all(0 <= v <= 1 for v in (x, y, w, h)) or w <= 0 or h <= 0:
            problems.append(f"{txt.name} 第{i}行数值非法: {parts}")
            continue
        if not (0 <= x - w / 2 and x + w / 2 <= 1 and 0 <= y - h / 2 and y + h / 2 <= 1):
            problems.append(f"{txt.name} 第{i}行框越界")
            continue
        rows.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return rows, problems


def main():
    dataset = ROOT / "dataset"
    if dataset.exists():
        shutil.rmtree(dataset)  # 每次重建，保证可重复运行
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_problems = []
    counts = {"train": [0, 0, 0], "val": [0, 0, 0]}
    random.seed(42)

    for folder, class_id, class_name in SOURCES:
        src = DESKTOP_DATA / folder
        img_dir, lbl_dir = src / "images", src / "labels"
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        samples = []
        for img in imgs:
            txt = lbl_dir / (img.stem + ".txt")
            if not txt.exists():
                all_problems.append(f"[{class_name}] 缺标签: {img.name}")
                continue
            rows, problems = read_valid_boxes(txt, class_id)
            all_problems.extend(f"[{class_name}] {p}" for p in problems)
            if not rows:
                all_problems.append(f"[{class_name}] 无有效框，跳过: {img.name}")
                continue
            samples.append((img, rows))

        random.shuffle(samples)
        n_val = max(1, round(len(samples) * VAL_RATIO))
        for split, items in (("val", samples[:n_val]), ("train", samples[n_val:])):
            for img, rows in items:
                # 加类名前缀，彻底避免三类间文件名撞车
                stem = f"{class_name}_{img.stem}"
                shutil.copy2(img, dataset / "images" / split / f"{stem}{img.suffix.lower()}")
                (dataset / "labels" / split / f"{stem}.txt").write_text(
                    "\n".join(rows) + "\n", encoding="utf-8")
                counts[split][class_id] += 1
        print(f"{class_name:18s} 共 {len(samples):3d} 张 -> "
              f"train {len(samples) - n_val:3d} / val {n_val:3d}")

    if all_problems:
        print(f"\n[发现 {len(all_problems)} 个问题，已跳过对应图片]")
        for p in all_problems[:30]:
            print("  -", p)

    yaml_text = (
        "# 路径相对于项目根目录，请在项目根目录执行 python train.py\n"
        "path: dataset\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: beaker\n"
        "  1: erlenmeyer_flask\n"
        "  2: test_tube\n"
    )
    (dataset / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"\n合计 train={sum(counts['train'])} val={sum(counts['val'])}")
    print("各类别 train 数量 [beaker, flask, test_tube]:", counts["train"])
    print("各类别 val   数量 [beaker, flask, test_tube]:", counts["val"])
    print("data.yaml 已生成 ->", dataset / "data.yaml")


if __name__ == "__main__":
    main()
