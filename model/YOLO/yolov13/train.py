import argparse
from ultralytics import YOLO
from ultralytics.utils.torch_utils import init_seeds
from weights import from_yaml, load_weights


def main():
    parser = argparse.ArgumentParser(
        description="Train an oriented bounding box detector."
    )
    parser.add_argument(
        "--model", default="YOLOv13n-OBB.yaml", help="Model YAML filename or path"
    )
    parser.add_argument(
        "--data", required=True, help="Path to a YOLO-format OBB dataset YAML"
    )
    parser.add_argument(
        "--weights", help="Initial weights (Ultralytics checkpoint or state dict)"
    )
    parser.add_argument(
        "--resume", metavar="LAST_PT", help="Resume an interrupted training run"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    if args.resume and args.weights:
        parser.error("Choose either --resume or --weights.")
    if args.resume:
        YOLO(args.resume, task="obb").train(
            resume=True, data=args.data, device=args.device
        )
        return
    init_seeds(args.seed, deterministic=True)
    detector = from_yaml(args.model)
    if args.weights:
        load_weights(detector, args.weights, strict=False)
    detector.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        nbs=32,
        imgsz=args.imgsz,
        workers=args.workers,
        seed=args.seed,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=False,
        patience=0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        momentum=0.9,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.0,
        cos_lr=True,
        close_mosaic=10,
        rect=False,
        plots=False,
    )


if __name__ == "__main__":
    main()
