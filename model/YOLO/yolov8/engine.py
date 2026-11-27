from argparse import ArgumentParser
from copy import copy
from pathlib import Path

from ultralytics import YOLO
from ultralytics.data import build_yolo_dataset
from ultralytics.models.yolo.obb import OBBTrainer, OBBValidator


NAME = "YOLOv8n-OBB"
CONFIG = Path(__file__).resolve().parent / "cfg" / f"{NAME}.yaml"
LOSSES = ("box_loss", "cls_loss", "dfl_loss")


class SquareValidator(OBBValidator):
    def build_dataset(self, img_path, mode="val", batch=None):
        return build_yolo_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=False,
            stride=int(self.stride),
        )


class Trainer(OBBTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        stride = max(int(self.model.stride.max()), 32)
        return build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=False, stride=stride
        )

    def get_validator(self):
        self.loss_names = LOSSES
        return SquareValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def validate(self):
        metrics = self.validator(self)
        metrics.pop("fitness", None)
        fitness = metrics["metrics/mAP50-95(B)"]
        if self.best_fitness is None or fitness > self.best_fitness:
            self.best_fitness = fitness
        return metrics, fitness


def parser(title, dataset=False):
    p = ArgumentParser(description=title)
    if dataset:
        p.add_argument(
            "--data",
            required=True,
            help="Data YAML. Label rows use class followed by four normalized (x,y) corners.",
        )
        p.add_argument("--batch", type=int, default=32)
        p.add_argument("--workers", type=int, default=4)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", help="cpu or CUDA device index")
    p.add_argument("--project", help="Output directory")
    p.add_argument("--name", help="Run name")
    return p


def train():
    p = parser(f"Train {NAME}", dataset=True)
    p.add_argument("--model", default=str(CONFIG), help="Architecture YAML")
    p.add_argument("--weights", help="Optional pretrained checkpoint")
    p.add_argument(
        "--resume", metavar="LAST_PT", help="Continue an interrupted training run"
    )
    p.add_argument("--epochs", type=int, default=100)
    args = vars(p.parse_args())
    pretrained = args.pop("weights")
    resume = args.pop("resume")
    if resume and pretrained:
        p.error("Choose --resume or --weights")
    opts = {k: v for k, v in args.items() if v is not None}
    opts.update(
        task="obb",
        pretrained=pretrained or False,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        momentum=0.9,
        warmup_bias_lr=0.0,
        weight_decay=0.0005,
        nbs=32,
        seed=42,
        patience=0,
    )
    if resume:
        opts.update(model=resume, resume=resume)
    Trainer(overrides=opts).train()


def val():
    p = parser(f"Validate {NAME}", dataset=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--split", choices=("val", "test"), default="val")
    args = vars(p.parse_args())
    model = YOLO(args.pop("weights"), task="obb")
    model.val(
        validator=SquareValidator,
        rect=False,
        half=False,
        **{k: v for k, v in args.items() if v is not None},
    )


def predict():
    p = parser(f"Predict with {NAME}")
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="Image, video, or directory")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--save-txt", action="store_true")
    args = vars(p.parse_args())
    model = YOLO(args.pop("weights"), task="obb")
    model.predict(save=True, **{k: v for k, v in args.items() if v is not None})
