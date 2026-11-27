from pathlib import Path

import torch
from torch import nn

from ultralytics import YOLO
from ultralytics.nn.tasks import torch_safe_load


def from_yaml(config):
    path = Path(config)
    if not path.is_file():
        path = Path(__file__).resolve().parent / "ultralytics/cfg/models/13" / path.name
    if not path.is_file():
        raise FileNotFoundError(f"Model YAML not found: {config}")
    return YOLO(str(path), task="obb")


def load_weights(detector, path, *, strict):
    # Load only trusted checkpoints. Native Ultralytics checkpoints contain Python objects.
    checkpoint, _ = torch_safe_load(path)
    source = checkpoint.get("ema")
    if source is None:
        source = checkpoint.get("model")
    names = checkpoint.get("names")
    if isinstance(source, nn.Module):
        names = getattr(source, "names", names)
        state = source.float().state_dict()
    else:
        state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict) or not all(
        isinstance(t, torch.Tensor) for t in state.values()
    ):
        raise ValueError("Expected a model checkpoint or tensor state_dict.")
    if strict:
        detector.model.load_state_dict(state, strict=True)
    else:
        detector.model.load_parameters(state)
    if names and len(names) == detector.model.yaml["nc"]:
        detector.model.names = names
    detector.ckpt = {"model": detector.model}
    return detector


def for_inference(weights, config=None):
    if config is None:
        return YOLO(weights, task="obb")
    return load_weights(from_yaml(config), weights, strict=True)
