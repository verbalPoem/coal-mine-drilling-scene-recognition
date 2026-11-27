# Ultralytics AGPL-3.0 License
from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import DetectionModel
from .detect import DetectionPredictor, DetectionTrainer, DetectionValidator

MODEL_CFG = Path(__file__).resolve().parents[2] / "cfg" / "models" / "YOLOv11n-HBB.yaml"


class YOLO(Model):
    """Horizontal bounding-box detector."""

    def __init__(self, model=MODEL_CFG, task=None, verbose=False):
        super().__init__(model=model, task=task or "detect", verbose=verbose)

    @property
    def task_map(self):
        return {"detect": {
            "model": DetectionModel,
            "trainer": DetectionTrainer,
            "validator": DetectionValidator,
            "predictor": DetectionPredictor,
        }}
