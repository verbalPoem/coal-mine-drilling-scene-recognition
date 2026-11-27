from ultralytics.engine.model import Model
from ultralytics.nn.tasks import YOLOv10DetectionModel
from .val import YOLOv10DetectionValidator
from .predict import YOLOv10DetectionPredictor
from .train import YOLOv10DetectionTrainer

class YOLOv10(Model):
    """YOLOv10 horizontal detector."""

    def __init__(self, model, task="detect", verbose=False):
        super().__init__(model=model, task=task, verbose=verbose)

    def one_to_one(self):
        """Use only the deployed detection branch for evaluation or prediction."""
        self.model.eval()
        head = self.model.model[-1]
        if not getattr(head, "deploy", False):
            del head.cv2, head.cv3
            head.deploy = True
        return self

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": YOLOv10DetectionModel,
                "trainer": YOLOv10DetectionTrainer,
                "validator": YOLOv10DetectionValidator,
                "predictor": YOLOv10DetectionPredictor,
            },
        }
