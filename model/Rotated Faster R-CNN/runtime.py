"""Local training and inference for rotated object detectors."""
import argparse
import json
import logging
from pathlib import Path


CONFIG = Path(__file__).resolve().parent / "configs" / "model.py"


def pipelines(size):
    normalize = dict(type="Normalize", mean=[123.675, 116.28, 103.53],
                     std=[58.395, 57.12, 57.375], to_rgb=True)
    pad = dict(type="Pad", size=(size, size))
    train = [
        dict(type="LoadImageFromFile"), dict(type="LoadAnnotations", with_bbox=True),
        dict(type="RResize", img_scale=(size, size)),
        dict(type="RRandomFlip", flip_ratio=[0.25, 0.25, 0.25],
             direction=["horizontal", "vertical", "diagonal"], version="le90"),
        normalize, pad, dict(type="DefaultFormatBundle"),
        dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
    ]
    test = [
        dict(type="LoadImageFromFile"),
        dict(type="MultiScaleFlipAug", img_scale=(size, size), flip=False,
             transforms=[dict(type="RResize"), normalize, pad,
                         dict(type="DefaultFormatBundle"), dict(type="Collect", keys=["img"])]),
    ]
    return train, test


def configure(args):
    from mmcv import Config

    cfg = Config.fromfile(str(args.config))
    train, test = pipelines(args.imgsz)
    cfg.train_pipeline = cfg.get("train_pipeline", train)
    cfg.test_pipeline = cfg.get("test_pipeline", test)
    for pipeline in (cfg.train_pipeline, cfg.test_pipeline):
        for step in pipeline:
            if step["type"] in ("RResize", "MultiScaleFlipAug"):
                step["img_scale"] = (args.imgsz, args.imgsz)
            for transform in [step, *step.get("transforms", [])]:
                if transform["type"] == "Pad":
                    transform.pop("size_divisor", None)
                    transform["size"] = (args.imgsz, args.imgsz)
    cfg.data = dict(test=dict(pipeline=cfg.test_pipeline))
    return cfg


def resolve_device(name):
    import torch

    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(f"cuda:{name}" if name.isdecimal() else name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use --device cpu or install the CUDA runtime.")
        torch.cuda.set_device(device.index or 0)
    elif device.type != "cpu":
        raise ValueError("Use cpu or a CUDA device index.")
    return device


def build_model(cfg, weights=None, backbone_weights=None):
    from mmcv import ConfigDict
    from mmcv.runner import load_checkpoint
    from mmrotate.models import build_detector

    if backbone_weights:
        backbone = cfg.model.backbone
        if "pretrained" in backbone:
            backbone.pretrained = str(backbone_weights)
        else:
            backbone.init_cfg = ConfigDict(type="Pretrained", checkpoint=str(backbone_weights))
    model = build_detector(cfg.model)
    model.cfg = cfg
    if weights:
        checkpoint = load_checkpoint(model, str(weights), map_location="cpu", strict=True)
        model.CLASSES = (checkpoint.get("meta") or {}).get("CLASSES")
    else:
        model.init_weights()
    return model


def parallel_model(model, device):
    from mmcv.parallel import MMDataParallel
    from torch import nn

    if device.type == "cuda":
        return MMDataParallel(model.to(device), device_ids=[device.index or 0])

    class CPUDataParallel(MMDataParallel):
        def __init__(self, module):
            # Keep MMCV's CPU DataContainer handling even on a CUDA-capable host.
            nn.Module.__init__(self)
            self.module = module
            self.device_ids = []
            self.dim = 0

    return CPUDataParallel(model.cpu())


def loader(cfg, args, split, *, train=False):
    from detector_data import dataset_config
    from mmdet.datasets import build_dataloader, build_dataset

    spec = dataset_config(args.data, split, cfg.train_pipeline if train else cfg.test_pipeline,
                          test_mode=not train)
    dataset = build_dataset(spec)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=args.batch if train else 1, workers_per_gpu=args.workers,
        num_gpus=1, dist=False, shuffle=train, seed=getattr(args, "seed", None),
        persistent_workers=False,
    )
    return dataset, data_loader


def train(args, cfg, device):
    from mmcv.runner import build_optimizer
    from mmdet.apis import set_random_seed
    from mmrotate.utils import get_root_logger
    from training import TrainingRunner, BestValidationHook

    set_random_seed(args.seed, deterministic=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    dataset, train_loader = loader(cfg, args, "train", train=True)
    _, val_loader = loader(cfg, args, "val")
    model = build_model(cfg, args.weights, args.backbone_weights)
    if args.weights and model.CLASSES and tuple(model.CLASSES) != tuple(dataset.CLASSES):
        raise ValueError("Dataset class order differs from the checkpoint.")
    model.CLASSES = tuple(dataset.CLASSES)
    wrapped = parallel_model(model, device)
    runner = TrainingRunner(
        wrapped, optimizer=build_optimizer(wrapped, cfg.optimizer),
        work_dir=str(args.work_dir), logger=get_root_logger(log_level=logging.INFO),
        max_epochs=args.epochs, meta={"CLASSES": model.CLASSES},
    )
    runner.register_training_hooks(
        cfg.get("lr_config", dict(policy="step", step=[4, 5], warmup="linear",
                                  warmup_iters=500, warmup_ratio=1.0/3)),
        cfg.get("optimizer_config", dict(grad_clip=dict(max_norm=35, norm_type=2))),
        dict(interval=1, save_optimizer=True, create_symlink=False),
        dict(interval=50, hooks=[dict(type="TextLoggerHook")]),
    )
    # Store the just-completed validation decision in the epoch recovery checkpoint.
    runner.register_hook(BestValidationHook(val_loader, interval=1, save_best="map50_95", rule="greater"), priority=49)
    if args.resume:
        runner.resume(str(args.resume), map_location=str(device))
        runner.meta["CLASSES"] = model.CLASSES
        if runner.epoch >= args.epochs:
            raise ValueError("--epochs must exceed the completed checkpoint epoch.")
    runner.run([train_loader], [("train", 1)])


def write_json(output, value):
    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def validate(args, cfg, device):
    from mmdet.apis import single_gpu_test

    ds, data_loader = loader(cfg, args, args.split)
    model = build_model(cfg, args.weights)
    if model.CLASSES and tuple(model.CLASSES) != tuple(ds.CLASSES):
        raise ValueError("Dataset class order differs from the checkpoint.")
    model.CLASSES = tuple(ds.CLASSES)
    results = single_gpu_test(parallel_model(model, device), data_loader)
    report = ds.evaluate_details(results)
    write_json(args.output, report)
    print(f"Evaluation saved to {args.output}")


def predict(args, cfg, device):
    import numpy as np
    from mmdet.apis import inference_detector
    from mmrotate.core import obb2poly_np

    model = build_model(cfg, args.weights)
    model.to(device).eval()
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = [args.source] if args.source.is_file() else sorted(
        path for path in args.source.iterdir() if path.is_file() and path.suffix.lower() in supported)
    if not files:
        raise ValueError("No images found in --source.")
    records = []
    for file in files:
        rows = []
        for index, detections in enumerate(inference_detector(model, str(file))):
            kept = detections[detections[:, -1] >= args.conf]
            if not len(kept):
                continue
            polygons = obb2poly_np(np.asarray(kept), version="le90")
            for box, polygon in zip(kept, polygons):
                rows.append(dict(class_id=index, label=model.CLASSES[index] if model.CLASSES else str(index),
                                 score=float(box[-1]), polygon=polygon[:8].reshape(4, 2).tolist()))
        records.append(dict(image=file.name, detections=rows))
    write_json(args.output, records)
    print(f"Predictions saved to {args.output}")


def run(mode):
    parser = argparse.ArgumentParser(description=f"{CONFIG.parents[1].name} {mode}")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", default="auto", help="auto, cpu or CUDA device index")
    parser.add_argument("--imgsz", type=int, default=640)
    if mode in {"train", "val"}:
        parser.add_argument("--data", type=Path, required=True, help="YOLO OBB dataset YAML (path, train, val, test, names)")
        parser.add_argument("--workers", type=int, default=2)
    if mode == "train":
        init = parser.add_mutually_exclusive_group()
        init.add_argument("--weights", type=Path, help="Initialize the full detector from a local checkpoint")
        init.add_argument("--resume", type=Path, help="Resume model, epoch and optimizer from a checkpoint")
        init.add_argument("--backbone-weights", type=Path, help="Initialize the backbone from a local checkpoint")
        parser.add_argument("--epochs", type=int, default=6)
        parser.add_argument("--batch", type=int, default=2)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--work-dir", type=Path, help="Output directory (defaults to runs/train or the resume directory)")
    else:
        parser.add_argument("--weights", type=Path, required=True)
        parser.add_argument("--output", type=Path, default=Path(f"runs/{mode}.json"))
    if mode == "val":
        parser.add_argument("--split", choices=("val", "test"), default="val")
    if mode == "predict":
        parser.add_argument("--source", type=Path, required=True, help="Image or image directory")
        parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    if mode == "train" and args.work_dir is None:
        args.work_dir = args.resume.resolve().parent if args.resume else Path("runs/train")
    for key in ("config", "data", "weights", "resume", "backbone_weights"):
        path = getattr(args, key, None)
        if path is not None and not path.is_file():
            parser.error(f"File for --{key.replace('_', '-')} does not exist: {path}")
    if args.imgsz <= 0 or args.imgsz % 32:
        parser.error("--imgsz must be a positive multiple of 32")
    if mode == "train" and (args.epochs < 1 or args.batch < 1):
        parser.error("--epochs and --batch must be positive")
    if mode == "predict" and (not args.source.exists() or not 0 <= args.conf <= 1):
        parser.error("--source must exist and --conf must be between 0 and 1")
    {"train": train, "val": validate, "predict": predict}[mode](args, configure(args), resolve_device(args.device))
