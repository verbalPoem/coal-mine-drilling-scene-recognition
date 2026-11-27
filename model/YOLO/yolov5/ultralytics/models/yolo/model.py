# Ultralytics YOLO, AGPL-3.0 license
from ultralytics.engine.model import Model
from ultralytics.models.yolo.obb import OBBPredictor, OBBTrainer, OBBValidator
from ultralytics.nn.tasks import OBBModel


class YOLO(Model):
    """YOLOv8 interface for oriented bounding boxes."""

    def __init__(self, model, task="obb", verbose=False):
        super().__init__(model=model, task=task or "obb", verbose=verbose)

    @property
    def task_map(self):
        return {"obb": {"model": OBBModel, "trainer": OBBTrainer, "validator": OBBValidator, "predictor": OBBPredictor}}
