# -*- coding: utf-8 -*-
"""YOLO-CAST 训练入口：注册自定义模块 + 打补丁 + 按论文 Table 4 协议训练。

用法：
    python train_yolocast.py --yaml yolocast_full.yaml --name yolocast_full
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ultralytics.nn.tasks as tasks
import yolocast_modules  # noqa: F401  (先导入，向 tasks 注册)
from yolocast_modules import C2f_CILA, GSFuse, P2TSCDetect  # noqa: F401
from patched_parse import apply_patch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="yolocast_full.yaml")
    ap.add_argument("--data", default="G:/workbuddy任务/麦yolo/train_pkg/dataset_gpu.yaml")
    ap.add_argument("--project", default="G:/workbuddy任务/麦yolo/yolocast_runs")
    ap.add_argument("--name", default="yolocast_full")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--pretrained", default="yolov8s.pt",
                    help="预训练权重用于层迁移；传空字符串则从零训练")
    args = ap.parse_args()

    # 向 tasks 全局注册自定义模块（parse_model 通过 globals()[m] 查找）
    for nm in ("C2f_CILA", "GSFuse", "P2TSCDetect"):
        setattr(tasks, nm, vars(__import__("yolocast_modules"))[nm])
    apply_patch()

    from ultralytics import YOLO

    model = YOLO(args.yaml)
    if args.pretrained:
        n_load = model.load(args.pretrained)
        print(f"[pretrained] transferred matching layers from {args.pretrained} (intersect={n_load})")
    model.info(verbose=True)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=0,
        workers=8,
        seed=42,
        patience=100,
        project=args.project,
        name=args.name,
        exist_ok=True,
        optimizer="AdamW",
        lr0=3e-4,
        lrf=0.05,
        momentum=0.90,
        weight_decay=2.5e-4,
        warmup_epochs=2.0,
        hsv_h=0.008, hsv_s=0.22, hsv_v=0.16,
        degrees=1.0, translate=0.04, scale=0.15,
        fliplr=0.50, mosaic=0.15, mixup=0.02,
        close_mosaic=10,
        amp=True,
    )
    print("TRAINING DONE ->", args.name)


if __name__ == "__main__":
    main()
