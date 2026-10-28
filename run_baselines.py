# -*- coding: utf-8 -*-
"""Retrain YOLOv5su and YOLO11n under the unified protocol for Table 6/7 rows."""
import os, sys, time, traceback

BASE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(BASE, "runs")
DATA = os.path.join(BASE, "dataset_local.yaml")

os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(BASE, "uconfig"))
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

from ultralytics import YOLO  # noqa: E402

JOBS = [
    ("yolov5su_base", os.path.join(BASE, "yolov5su.pt")),
    ("yolo11n_base", os.path.join(BASE, "yolo11n.pt")),
]

KW = dict(
    data=DATA, epochs=100, imgsz=640, batch=6, device=0, workers=6,
    seed=42, patience=100, project=RUNS, exist_ok=True,
    optimizer="AdamW", lr0=3e-4, lrf=0.05, momentum=0.90, weight_decay=2.5e-4,
    warmup_epochs=2.0, hsv_h=0.008, hsv_s=0.22, hsv_v=0.16,
    degrees=1.0, translate=0.04, scale=0.15, fliplr=0.50,
    mosaic=0.15, mixup=0.02, close_mosaic=10, amp=True,
)


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(BASE, "queue_progress.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    for name, weights in JOBS:
        try:
            log(f"START {name}")
            m = YOLO(weights)
            m.train(name=name, **KW)
            log(f"DONE {name}")
        except Exception:
            log(f"FAILED {name}: {traceback.format_exc(limit=3)}")


if __name__ == "__main__":
    main()
