# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import OBBModel


class YOLO(Model):
    """Oriented detector with the native Ultralytics training and inference API."""

    def __init__(self, model="YOLOv13n-OBB.yaml", task="obb", verbose=False):
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        return {
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            }
        }
