import argparse

import torch

from ultralytics import YOLO
from ultralytics.models.yolo.model import MODEL_CFG


def main():
    parser = argparse.ArgumentParser(description="Train a horizontal bounding-box detector")
    parser.add_argument("--data", required=True, help="Dataset YAML")
    parser.add_argument("--model", default=str(MODEL_CFG), help="Model YAML")
    parser.add_argument("--weights", help="Optional local initialization checkpoint")
    parser.add_argument("--resume", help="Continue a native training checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on supported GPUs")
    args = parser.parse_args()
    if args.resume:
        YOLO(args.resume).train(resume=True, data=args.data, device=args.device)
        return
    torch.manual_seed(42)
    model = YOLO(args.model)
    if args.weights:
        model.load(args.weights)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                workers=args.workers, device=args.device, amp=args.amp, seed=42, nbs=32,
                optimizer="AdamW", lr0=0.001, lrf=0.01, momentum=0.9, warmup_bias_lr=0.0,
                weight_decay=0.0005, cos_lr=True, patience=0, pretrained=False)


if __name__ == "__main__":
    main()
