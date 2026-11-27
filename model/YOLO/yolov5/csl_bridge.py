"""Training targets and oriented-box decoding."""

import numpy as np
import torch
from ultralytics.utils import LOGGER
from ultralytics.utils.ops import xywhr2xyxyxyxy

from models.yolo import Model
from utils.loss import ComputeLoss
from utils.rboxs_utils import pi, poly2rbox


def project_targets_to_native(batch, device):
    boxes = batch["bboxes"].detach().float().cpu().clone()
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError("Expected oriented bounding boxes with five xywhr values")
    if not len(boxes):
        return torch.empty((0, 187), device=device, dtype=torch.float32)
    h, w = batch["img"].shape[-2:]
    boxes[:, :4] *= torch.tensor([w, h, w, h])
    corners = xywhr2xyxyxyxy(boxes).reshape(-1, 8).numpy()
    longside, labels = poly2rbox(
        corners, num_cls_thata=180, radius=2.0, use_pi=True, use_gaussian=True
    )
    indices = batch["batch_idx"].detach().cpu().reshape(-1, 1).numpy()
    classes = batch["cls"].detach().cpu().reshape(-1, 1).numpy()
    targets = np.concatenate((indices, classes, longside, labels), axis=1)
    return torch.as_tensor(targets, device=device, dtype=torch.float32)


def decode_native_predictions(pred):
    class_end = pred.shape[-1] - 180
    scores = pred[..., 4:5] * pred[..., 5:class_end]
    angle = (90 - pred[..., class_end:].argmax(-1, keepdim=True)).to(pred.dtype) * (
        pi / 180
    )
    return torch.cat((pred[..., :4], scores, angle), -1).transpose(1, 2).contiguous()


class CSLNativeLoss:
    def __init__(self, model):
        self.native = ComputeLoss(model)

    def __call__(self, preds, batch):
        raw = preds[1] if isinstance(preds, tuple) else preds
        targets = project_targets_to_native(batch, raw[0].device)
        loss, parts = self.native(raw, targets)
        return loss.squeeze(0), parts


class CSLProjectModel(Model):
    def __init__(self, cfg, ch=3, nc=None):
        super().__init__(cfg, ch, nc)
        self.task = "obb"
        self.criterion = None
        self.hyp = dict(
            box=0.05,
            cls=0.5 * self.nc / 80,
            obj=1.0,
            theta=0.5,
            cls_pw=1.0,
            obj_pw=1.0,
            theta_pw=1.0,
            anchor_t=4.0,
            iou_t=0.2,
            fl_gamma=0.0,
            label_smoothing=0.0,
            cls_theta=180,
            csl_radius=2.0,
        )

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        if augment:
            raise ValueError("This model supports single-scale prediction")
        output = self._predict_once(x, profile, visualize, embed)
        if self.training:
            return output
        pred, raw = output
        return decode_native_predictions(pred), raw

    def init_criterion(self):
        return CSLNativeLoss(self)

    def load(self, weights, verbose=True):
        donor = weights["model"] if isinstance(weights, dict) else weights
        head = donor.model[-1]
        if getattr(head, "no", None) != getattr(head, "nc", 0) + 5:
            return super().load(weights, verbose=verbose)
        source = donor.float().state_dict()
        prefix = f"model.{len(self.model) - 1}."
        state = {
            key: source[key]
            for key, tensor in self.state_dict().items()
            if not key.startswith(prefix)
            and key in source
            and source[key].shape == tensor.shape
        }
        if not state:
            raise ValueError(
                "The checkpoint has no compatible backbone or neck tensors"
            )
        self.load_state_dict(state, strict=False)
        if verbose:
            LOGGER.info(f"Transferred {len(state)} backbone and neck tensors")
