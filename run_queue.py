# -*- coding: utf-8 -*-
"""YOLO-CAST 补充实验串行队列（本机 RTX 4060 Ti）。

按论文 Table 4 统一协议跑 100 epochs：
  1. smoke_test        1 epoch 环境验证
  2. cila_p2tsc        两两组合 E
  3. cila_gsbfpn       两两组合 D
  4. gsbfpn_p2tsc      两两组合 F
  5. full_wa           宽度自适应修复版完整模型
每个实验完成后把关键指标追加到 experiments/queue_progress.log。
"""
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(BASE), "mayolo", "麦yolo", "yolocast_impl"))

EXP = os.path.join(BASE)
RUNS = os.path.join(BASE, "runs")
DATA = os.path.join(EXP, "dataset_local.yaml")

os.makedirs(RUNS, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(EXP, "uconfig"))
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

import torch  # noqa: E402
import ultralytics.nn.tasks as tasks  # noqa: E402
import yolocast_modules  # noqa: E402
from yolocast_modules import C2f_CILA, GSFuse, P2TSCDetect  # noqa: E402
from patched_parse import apply_patch  # noqa: E402

for nm in ("C2f_CILA", "GSFuse", "P2TSCDetect"):
    setattr(tasks, nm, vars(yolocast_modules)[nm])
apply_patch()

from ultralytics import YOLO  # noqa: E402

JOBS = [
    # (yaml, name, epochs, pretrained)
    (None, "smoke_test", 1, "yolov8n.pt"),
    ("yolocast_cila_p2tsc.yaml", "cila_p2tsc", 100, "yolov8s.pt"),
    ("yolocast_cila_gsbfpn.yaml", "cila_gsbfpn", 100, "yolov8s.pt"),
    ("yolocast_gsbfpn_p2tsc.yaml", "gsbfpn_p2tsc", 100, "yolov8s.pt"),
    ("yolocast_full_wa.yaml", "full_wa", 100, "yolov8s.pt"),
]


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(EXP, "queue_progress.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    log(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    for yaml_name, name, epochs, pre in JOBS:
        yaml_path = os.path.join(EXP, yaml_name) if yaml_name else "yolov8n.pt"
        try:
            log(f"START {name} ({yaml_name or 'yolov8n.pt'}, {epochs} epochs)")
            model = YOLO(yaml_path)
            if pre and yaml_name:
                n = model.load(pre)
                log(f"{name}: pretrained transfer intersect={n}")
            model.train(
                data=DATA,
                epochs=epochs,
                imgsz=640,
                batch=6,
                device=0,
                workers=6,
                seed=42,
                patience=epochs,
                project=RUNS,
                name=name,
                exist_ok=True,
                # ===== 论文 Table 4 统一协议 =====
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
            log(f"DONE {name}")
        except Exception as e:  # noqa: BLE001
            log(f"FAILED {name}: {type(e).__name__}: {e}")
            raise


if __name__ == "__main__":
    main()
