# -*- coding: utf-8 -*-
"""Per-class AP@0.5 of the final YOLO-CAST (cila_lowaug) + standard val plots."""
import os, sys, json


def main():
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
    m = YOLO(os.path.join(BASE, "runs", "cila_lowaug", "weights", "last.pt"))
    r = m.val(data=os.path.join(BASE, "dataset_local.yaml"), plots=True, verbose=False,
              half=True,
              project=os.path.join(BASE, "runs", "val_final"), name="cila_lowaug_last", exist_ok=True)
    names = r.names
    out = {"overall": {"P": r.box.mp * 100, "R": r.box.mr * 100,
                       "mAP50": r.box.map50 * 100, "mAP5095": r.box.map * 100},
           "per_class": {}}
    for i, c in enumerate(r.box.ap50):
        out["per_class"][names[i]] = round(c * 100, 1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    json.dump(out, open(os.path.join(BASE, "final_perclass.json"), "w"), ensure_ascii=False, indent=1)
    print("SAVED")


if __name__ == "__main__":
    main()
