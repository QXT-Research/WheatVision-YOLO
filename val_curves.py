# -*- coding: utf-8 -*-
"""Extract curve statistics of the final model for Figure 9/10 paragraph updates."""
import os, sys, json

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(BASE), "mayolo", "麦yolo", "yolocast_impl"))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(BASE, "uconfig"))
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

import ultralytics.nn.tasks as tasks
import yolocast_modules
from yolocast_modules import C2f_CILA, GSFuse, P2TSCDetect
from patched_parse import apply_patch
for nm in ("C2f_CILA", "GSFuse", "P2TSCDetect"):
    setattr(tasks, nm, vars(yolocast_modules)[nm])
apply_patch()

from ultralytics import YOLO

def main():
    m = YOLO(os.path.join(BASE, "runs", "cila_lowaug", "weights", "last.pt"))
    r = m.val(data=os.path.join(BASE, "dataset_local.yaml"), plots=True, verbose=False, half=True,
              project=os.path.join(BASE, "runs", "val_final"), name="curve_stats", exist_ok=True)
    names = r.names
    ap50 = r.box.ap50
    order = sorted(range(len(ap50)), key=lambda i: ap50[i])
    low = [(names[i], round(ap50[i] * 100, 1)) for i in order[:3]]
    # F1 curve
    import numpy as np
    fc = np.asarray(r.box.f1_curve)          # (nc, 1000)
    x = np.linspace(0, 1, 1000)
    y = fc.mean(0)                            # all-classes curve as plotted
    k = int(y.argmax())
    above = x[y >= 0.85]
    out = {
        "map50_all": round(r.box.map50 * 100, 2),
        "lowest_ap50": low,
        "f1_peak": round(float(y[k]), 4),
        "f1_peak_conf": round(float(x[k]), 3),
        "f1_above_085_band": [round(float(above.min()), 3), round(float(above.max()), 3)] if len(above) else None,
    }
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(BASE, "curve_stats.json"), "w"), indent=1)
    print("SAVED")

if __name__ == "__main__":
    main()
