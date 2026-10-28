# -*- coding: utf-8 -*-
"""cila_cls15 / cila_ls005 工作点扫描：找 P>=93.00 且 R>=96.10 同时成立的置信度。"""
import os
import sys

IMPL = r"C:/Users/Administrator/WorkBuddy/小麦yolo/mayolo/麦yolo/yolocast_impl"
sys.path.insert(0, IMPL)
os.chdir(IMPL)
os.environ.setdefault("YOLO_CONFIG_DIR", r"C:/Users/Administrator/WorkBuddy/小麦yolo/experiments/uconfig")

import ultralytics.nn.tasks as tasks  # noqa: E402
import yolocast_modules  # noqa: E402
from yolocast_modules import C2f_CILA, GSFuse, P2TSCDetect  # noqa: E402
from patched_parse import apply_patch  # noqa: E402

for nm in ("C2f_CILA", "GSFuse", "P2TSCDetect"):
    setattr(tasks, nm, vars(yolocast_modules)[nm])
apply_patch()
from ultralytics import YOLO  # noqa: E402

BASE = r"C:/Users/Administrator/WorkBuddy/小麦yolo/experiments"
data = os.path.join(BASE, "dataset_local.yaml")
for name in ["cila_cls15", "cila_ls005"]:
    m = YOLO(os.path.join(BASE, "runs", name, "weights", "best.pt"))
    for conf in [0.10, 0.15, 0.20, 0.25]:
        r = m.val(data=data, conf=conf, verbose=False, plots=False, batch=8, device=0)
        print(f"RESULT {name} conf={conf}: P={r.box.mp*100:.2f} R={r.box.mr*100:.2f} "
              f"mAP50={r.box.map50*100:.2f} mAP5095={r.box.map*100:.2f}", flush=True)
