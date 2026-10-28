## Project Structure

```text

WheatVision-YOLO-CAST/
│
├── yolocast_modules.py
├── patched_parse.py
├── train_yolocast.py
├── smoke_test.py
│
├── yolocast_full.yaml
├── yolocast_cila_only.yaml
├── yolocast_gsbfpn_only.yaml
├── yolocast_p2tsc_only.yaml
│
├── experiments/
│   ├── run_baselines.py
│   ├── val_baselines.py
│   ├── val_perclass.py
│   ├── val_curves.py
│   ├── conf_sweep.py
│   ├── ...
│   └── results/*.json
│
├── README.md
├── requirements.txt
└── LICENSE
