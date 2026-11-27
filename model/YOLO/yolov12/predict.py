import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Predict horizontal bounding boxes")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="Image, directory or video")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    YOLO(args.weights).predict(source=args.source, imgsz=args.imgsz,
                               conf=args.conf, device=args.device, save=True)


if __name__ == "__main__":
    main()
