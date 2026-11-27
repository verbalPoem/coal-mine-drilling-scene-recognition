import torch
from torch import nn
from torch.nn import functional as F

from ultralytics.utils.loss import v8OBBLoss


class LevelClassFocal(nn.Module):
    """Redistribute negative classification weight within each level and class."""

    def __init__(self, strength=0.125, gamma=1.0):
        super().__init__()
        self.strength = strength
        self.gamma = gamma
        self.level_sizes = ()

    def forward(self, logits, targets):
        with torch.autocast(device_type=logits.device.type, enabled=False):
            z, q = logits.float(), targets.detach().float()
            bce = F.binary_cross_entropy_with_logits(z, q, reduction="none")
            factors = []
            with torch.no_grad():
                for scores, labels in zip(
                    z.split(self.level_sizes, 1), q.split(self.level_sizes, 1)
                ):
                    p = scores.sigmoid()
                    negative = labels.eq(0)
                    modulation = p.pow(self.gamma)
                    mass = (p * negative).sum((0, 1))
                    mean = (p * modulation * negative).sum((0, 1)) / mass.clamp_min(
                        torch.finfo(p.dtype).tiny
                    )
                    mean = torch.where(mass > 0, mean, torch.zeros_like(mean))
                    w = 1 + self.strength * (modulation - mean)
                    factors.append(torch.where(negative, w, torch.ones_like(w)))
            return bce * torch.cat(factors, dim=1)


class LevelClassOBBLoss(v8OBBLoss):
    def __init__(self, model):
        super().__init__(model)
        self.bce = LevelClassFocal()

    def __call__(self, preds, batch):
        features, _ = preds if isinstance(preds[0], list) else preds[1]
        self.bce.level_sizes = tuple(
            feature.shape[-2] * feature.shape[-1] for feature in features
        )
        return super().__call__(preds, batch)


@torch.no_grad()
def weights(logits, targets, foreground, sizes, strength=0.125, prior=16.0):
    if (
        logits.ndim != 3
        or logits.shape != targets.shape
        or logits.shape[:2] != foreground.shape
    ):
        raise ValueError(
            "Expected batch x anchors x classes and the native TAL foreground mask"
        )
    if sum(sizes) != logits.shape[1] or foreground.dtype != torch.bool or prior < 0:
        raise ValueError("Inconsistent TAL mask, level sizes or support prior")
    outputs = []
    for z, q, fg in zip(
        logits.split(sizes, 1), targets.split(sizes, 1), foreground.split(sizes, 1)
    ):
        prob = z.float().sigmoid()
        w = torch.ones_like(prob)
        for mask in (fg, ~fg):
            eligible = q.eq(0) & mask.unsqueeze(-1)
            mass = (prob * eligible).sum(1, keepdim=True)
            square = (prob.square() * eligible).sum(1, keepdim=True)
            mean = square / torch.where(mass > 0, mass, 1)
            support = mass.square() / torch.where(square > 0, square, 1)
            normalizer = support + prior
            attenuation = support / torch.where(normalizer > 0, normalizer, 1)
            w = torch.where(eligible, 1 + strength * attenuation * (prob - mean), w)
        outputs.append(w)
    return torch.cat(outputs, dim=1)


class SupportBudgetFocal(nn.Module):
    def __init__(self):
        super().__init__()
        self.foreground = None
        self.sizes = None

    def forward(self, logits, targets):
        foreground, self.foreground = self.foreground, None
        if foreground is None or self.sizes is None:
            raise RuntimeError("Native assignment and FPN levels must be supplied")
        with torch.autocast(device_type=logits.device.type, enabled=False):
            z, q = logits.float(), targets.detach().float()
            return F.binary_cross_entropy_with_logits(z, q, reduction="none") * weights(
                z, q, foreground, self.sizes
            )


class CaptureForeground(nn.Module):
    def __init__(self, assigner, focal):
        super().__init__()
        self.native, self.focal = assigner, focal

    def forward(self, *args, **kwargs):
        result = self.native(*args, **kwargs)
        if self.focal.foreground is not None:
            raise RuntimeError("Previous TAL context was not consumed")
        self.focal.foreground = result[3].detach().bool()
        return result


class SupportBudgetOBBLoss(v8OBBLoss):
    def __init__(self, model):
        super().__init__(model)
        if not isinstance(self.bce, nn.BCEWithLogitsLoss):
            raise ValueError("Support Focal requires BCE logits")
        self.bce = SupportBudgetFocal()
        self.assigner = CaptureForeground(self.assigner, self.bce)

    def __call__(self, preds, batch):
        features, _ = preds if isinstance(preds[0], list) else preds[1]
        self.bce.sizes = tuple(x.shape[-2] * x.shape[-1] for x in features)
        return super().__call__(preds, batch)
