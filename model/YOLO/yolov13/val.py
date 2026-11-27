import argparse

from weights import for_inference


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an oriented bounding box detector."
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model", help="Architecture YAML for a tensor state_dict")
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("val", "test", "train"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    for_inference(args.weights, args.model).val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        rect=False,
        half=False,
    )


if __name__ == "__main__":
    main()
