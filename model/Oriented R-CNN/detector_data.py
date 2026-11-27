"""YOLO OBB annotations for native rotated detectors."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import yaml
from mmrotate.datasets import DOTADataset
from mmrotate.datasets.builder import ROTATED_DATASETS

from metrics import evaluate_predictions


_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _read_yaml(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("Expected a dataset YAML mapping")
    return data


def read_names(data_yaml):
    data = _read_yaml(data_yaml)
    names = data.get("names")
    if isinstance(names, dict):
        indices = {int(i): value for i, value in names.items()}
        if set(indices) != set(range(len(indices))):
            raise ValueError("Class IDs must start at zero and be consecutive")
        names = [indices[i] for i in range(len(indices))]
    if not isinstance(names, list) or len(names) != 6 or not all(isinstance(n, str) and n.strip() for n in names):
        raise ValueError("Expected six class names in the dataset YAML")
    if len(set(names)) != len(names) or int(data.get("nc", len(names))) != len(names):
        raise ValueError("Class names must be unique and agree with nc")
    return tuple(names)


def dataset_config(data_yaml, split, pipeline, test_mode=False):
    return dict(type="OrientedBoxDataset", ann_file=str(Path(data_yaml).resolve()), split=split,
                img_prefix="", classes=read_names(data_yaml), pipeline=pipeline,
                version="le90", filter_empty_gt=False, test_mode=test_mode)


def _images(data_yaml, split):
    document = _read_yaml(data_yaml)
    root = Path(data_yaml).parent / document.get("path", ".")
    entries = document.get(split)
    if isinstance(entries, str):
        entries = [entries]
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Missing split in dataset YAML: {split}")
    paths = []
    for entry in entries:
        path = root / entry
        if path.is_dir():
            paths.extend(sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES))
        elif path.is_file() and path.suffix.lower() == ".txt":
            for value in path.read_text(encoding="utf-8-sig").splitlines():
                value = value.strip()
                if not value:
                    continue
                image = path.parent / value
                if not image.is_file() and not Path(value).is_absolute():
                    image = root / value
                paths.append(image)
        elif path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            paths.append(path)
        else:
            raise FileNotFoundError(f"Image or split source was not found: {path}")
    paths = [p.resolve() for p in paths]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError(f"Split {split} must contain a nonempty list of distinct images")
    return paths


def _label_file(image):
    parts = list(image.parts)
    for pos in range(len(parts) - 2, -1, -1):
        if parts[pos] == "images":
            parts[pos] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _annotation(image, width, height, classes):
    path = _label_file(image)
    boxes, labels, polygons = [], [], []
    if path.is_file():
        for i, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                fields = line.split()
                if len(fields) != 9:
                    raise ValueError("Expected class and four corners")
                cls = int(fields[0])
                xy = np.array(fields[1:], dtype=np.float64).reshape(4, 2)
                if cls < 0 or cls >= len(classes) or not np.isfinite(xy).all() or (xy < 0).any() or (xy > 1).any():
                    raise ValueError("Class or normalized coordinates out of range")
                polygon = (xy * [width, height]).astype(np.float32)
                if abs(cv2.contourArea(polygon)) == 0:
                    raise ValueError("Zero-area polygon")
                (cx, cy), (w, h), angle = cv2.minAreaRect(polygon)
                angle = angle / 180 * np.pi
                if w < h:
                    w, h, angle = h, w, angle + np.pi / 2
                angle = (angle + np.pi / 2) % np.pi - np.pi / 2
            except ValueError as error:
                raise ValueError(f"{path}:{i}: {error}") from error
            boxes.append((cx, cy, w, h, angle))
            labels.append(cls)
            polygons.append(polygon.reshape(-1))
    return dict(bboxes=np.asarray(boxes, np.float32).reshape(-1, 5),
                labels=np.asarray(labels, np.int64), polygons=np.asarray(polygons, np.float32).reshape(-1, 8),
                bboxes_ignore=np.empty((0, 5), np.float32), labels_ignore=np.empty(0, np.int64),
                polygons_ignore=np.empty((0, 8), np.float32))


@ROTATED_DATASETS.register_module()
class OrientedBoxDataset(DOTADataset):
    def __init__(self, split, **kwargs):
        self.split = split
        super().__init__(**kwargs)

    def load_annotations(self, ann_file):
        infos = []
        for image_path in _images(ann_file, self.split):
            with Image.open(image_path) as image:
                width, height = image.size
                if image.getexif().get(274, 1) in (5, 6, 7, 8):
                    width, height = height, width
            infos.append(dict(filename=str(image_path), width=width, height=height, img_id=str(image_path),
                              ann=_annotation(image_path, width, height, self.CLASSES)))
        self.img_ids = [item["img_id"] for item in infos]
        return infos

    def _filter_imgs(self, min_size=0):
        return list(range(len(self.data_infos)))

    def evaluate_details(self, results):
        return evaluate_predictions(results, [info["ann"] for info in self.data_infos], self.CLASSES)

    def evaluate(self, results, metric="map50_95", logger=None, **kwargs):
        if metric == "mAP":
            return super().evaluate(results, metric=metric, logger=logger, **kwargs)
        details = self.evaluate_details(results)
        return {key: details[key] for key in ("precision", "recall", "map50", "map50_95")}
