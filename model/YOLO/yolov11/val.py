import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Evaluate a horizontal bounding-box detector")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True, help="Dataset YAML")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    YOLO(args.weights).val(data=args.data, split=args.split, imgsz=args.imgsz,
                           batch=args.batch, device=args.device, rect=False, half=False)


if __name__ == "__main__":
    main()
