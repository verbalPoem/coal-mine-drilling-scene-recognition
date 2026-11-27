import argparse
from pathlib import Path

from ultralytics import YOLOv10


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "ultralytics/cfg/models/v10/YOLOv10n-HBB.yaml"


def run(mode):
    parser = argparse.ArgumentParser(description=f"YOLOv10n-HBB {mode}")
    parser.add_argument("--device", default="", help="cpu or CUDA device index")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default=None, help="output directory")
    parser.add_argument("--name", default=None, help="run name")
    parser.add_argument("--weights", type=Path, required=mode != "train")
    if mode in {"train", "val"}:
        parser.add_argument("--data", type=Path, required=True, help="local YOLO dataset YAML")
        parser.add_argument("--batch", type=int, default=32)
        parser.add_argument("--workers", type=int, default=4)
    if mode == "train":
        parser.add_argument("--model", type=Path, default=MODEL)
        parser.add_argument("--resume", type=Path, help="training checkpoint")
        parser.add_argument("--epochs", type=int, default=100)
    elif mode == "val":
        parser.add_argument("--split", choices=("val", "test"), default="val")
    else:
        parser.add_argument("--source", required=True, help="image, video, or directory")
        parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    for key in ("data", "weights", "resume", "model"):
        path = getattr(args, key, None)
        if path is not None and not path.is_file():
            parser.error(f"--{key} file does not exist: {path}")

    options = {k: v for k, v in vars(args).items() if k in {"device","imgsz","project","name","batch","workers"} and v is not None}
    if mode == "train":
        if args.resume and args.weights:
            parser.error("choose either --resume or --weights")
        model = YOLOv10(str(args.resume or args.model))
        if args.weights:
            model.load(str(args.weights.resolve()))
        model.train(
            data=str(args.data.resolve()), epochs=args.epochs, resume=bool(args.resume),
            seed=42, optimizer="AdamW", lr0=0.001, lrf=0.01, momentum=0.9,
            weight_decay=0.0005, warmup_bias_lr=0.0, nbs=32, cos_lr=True,
            patience=0, plots=False, **options,
        )
    elif mode == "val":
        model = YOLOv10(str(args.weights.resolve())).one_to_one()
        model.val(data=str(args.data.resolve()), split=args.split, rect=False, half=False, **options)
    else:
        model = YOLOv10(str(args.weights.resolve())).one_to_one()
        model.predict(source=args.source, conf=args.conf, save=True, **options)
