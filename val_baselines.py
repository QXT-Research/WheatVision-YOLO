# -*- coding: utf-8 -*-
"""Unified-protocol val of retrained YOLOv5su / YOLO11n baselines (same as paper val: default + half=True)."""
import os, sys, json

BASE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(BASE, "uconfig"))
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

from ultralytics import YOLO

DATA = os.path.join(BASE, "dataset_local.yaml")
def main():
    out = {}
    for name in ("yolov5su_base", "yolo11n_base"):
        m = YOLO(os.path.join(BASE, "runs", name, "weights", "best.pt"))
        r = m.val(data=DATA, plots=True, verbose=False, half=True,
                  project=os.path.join(BASE, "runs", "val_final"), name=name, exist_ok=True)
        info = (None,
                sum(p.numel() for p in m.model.parameters()),
                None,
                __import__("ultralytics.utils.torch_utils", fromlist=["get_flops"]).get_flops(m.model, 640))
        sp = r.speed
        fps = 1000.0 / (sp["preprocess"] + sp["inference"] + sp["postprocess"])
        out[name] = {
            "P": round(r.box.mp * 100, 2), "R": round(r.box.mr * 100, 2),
            "mAP50": round(r.box.map50 * 100, 2), "mAP50-95": round(r.box.map * 100, 2),
            "params_M": round(info[1] / 1e6, 2), "GFLOPs": round(info[3], 1),
            "FPS": round(fps, 1),
        }
        print(name, out[name], flush=True)

    json.dump(out, open(os.path.join(BASE, "baselines_final.json"), "w"), indent=1)
    print("SAVED")


if __name__ == "__main__":
    main()
