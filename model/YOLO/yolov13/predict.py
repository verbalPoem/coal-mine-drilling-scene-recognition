import argparse

from weights import for_inference


def main():
    parser = argparse.ArgumentParser(
        description="Predict oriented boxes in images or video."
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model", help="Architecture YAML for a tensor state_dict")
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    results = for_inference(args.weights, args.model).predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        stream=True,
    )
    for _ in results:
        pass


if __name__ == "__main__":
    main()
