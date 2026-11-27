"""Rotated detection metrics using ProbIoU matching and 101-point AP."""
# Matching, precision/recall and AP follow Ultralytics (AGPL-3.0).
import cv2
import numpy as np
import torch


def _covariance(boxes):
    a, b, angle = torch.cat((boxes[:, 2:4].square() / 12, boxes[:, 4:]), dim=-1).split(1, dim=-1)
    c, s = angle.cos(), angle.sin()
    return a * c.square() + b * s.square(), a * s.square() + b * c.square(), (a - b) * c * s


def _probiou(labels, predictions, eps=1e-7):
    x1, y1 = labels[:, :2].split(1, dim=-1)
    x2, y2 = predictions[:, :2].split(1, dim=-1)
    a, b, c = _covariance(labels)
    u, v, w = (t.T for t in _covariance(predictions))
    denom = (a + u) * (b + v) - (c + w).square()
    t1 = ((a + u) * (y1 - y2.T).square() + (b + v) * (x1 - x2.T).square()) / (denom + eps) * 0.25
    t2 = (c + w) * (x2.T - x1) * (y1 - y2.T) / (denom + eps) * 0.5
    t3 = (denom / (4 * ((a * b - c.square()).clamp_min(0) * (u * v - w.square()).clamp_min(0)).sqrt() + eps) + eps).log() * 0.5
    return 1 - (1 - (-(t1 + t2 + t3).clamp(eps, 100)).exp() + eps).sqrt()


def _matches(boxes, classes, polygons, targets):
    hit = np.zeros((len(boxes), 10), dtype=bool)
    valid = (boxes[:, 2:4] > 0).all(axis=1)
    if not len(targets) or not valid.any():
        return hit
    gt = []
    for poly in polygons:
        (x, y), (w, h), deg = cv2.minAreaRect(poly.reshape(4, 2))
        gt.append((x, y, w, h, deg / 180 * np.pi))
    similarity = np.zeros((len(targets), len(boxes)), dtype=np.float32)
    similarity[:, valid] = _probiou(torch.tensor(gt, dtype=torch.float32),
                                     torch.from_numpy(boxes[valid])).numpy()
    similarity *= targets[:, None] == classes
    for i, threshold in enumerate(torch.linspace(.5, .95, 10).tolist()):
        pairs = np.argwhere(similarity >= threshold)
        if len(pairs):
            pairs = pairs[np.argsort(similarity[pairs[:, 0], pairs[:, 1]])[::-1]]
            pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
            pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
            hit[pairs[:, 1], i] = True
    return hit


def evaluate_predictions(predictions, annotations, names):
    """Scores at the mean-F1 operating point and AP over thresholds .50 through .95."""
    if len(predictions) != len(annotations) or not len(annotations):
        raise ValueError("Expected one prediction per image on a nonempty split")
    all_hits, all_scores, all_classes, all_targets = [], [], [], []
    for output, ann in zip(predictions, annotations):
        if len(output) != len(names):
            raise ValueError("Prediction class count differs from the dataset")
        det = np.concatenate([np.column_stack((np.asarray(rows, np.float32).reshape(-1, 6),
                                               np.full(len(rows), i, np.float32)))
                              for i, rows in enumerate(output)])
        if not np.isfinite(det).all() or (det[:, 2:4] < 0).any():
            raise ValueError("Invalid predicted box")
        if np.any((det[:, 5] < 0) | (det[:, 5] > 1)):
            raise ValueError("Invalid confidence")
        det = det[np.argsort(-det[:, 5])]
        targets = ann["labels"]
        all_hits.append(_matches(det[:, :5], det[:, 6], ann["polygons"], targets))
        all_scores.append(det[:, 5])
        all_classes.append(det[:, 6])
        all_targets.append(targets)
    score = np.concatenate(all_scores)
    order = np.argsort(-score)
    score = score[order]
    cls, hits = np.concatenate(all_classes)[order], np.concatenate(all_hits)[order]
    counts = np.bincount(np.concatenate(all_targets), minlength=len(names))
    ids = np.flatnonzero(counts)
    x = np.linspace(0, 1, 1000)
    p = np.zeros((len(names), len(x)))
    r = np.zeros_like(p)
    ap = np.zeros((len(names), 10))
    for c in ids:
        mask = cls == c
        if not mask.any():
            continue
        tp = np.cumsum(hits[mask], axis=0)
        fp = np.cumsum(1 - hits[mask], axis=0)
        recall = tp / (counts[c] + 1e-16)
        precision = tp / (tp + fp)
        r[c] = np.interp(-x, -score[mask], recall[:, 0], left=0)
        p[c] = np.interp(-x, -score[mask], precision[:, 0], left=1)
        for j in range(10):
            rec = np.concatenate(([0.], recall[:, j], [1.]))
            pre = np.concatenate(([1.], precision[:, j], [0.]))
            envelope = np.maximum.accumulate(pre[::-1])[::-1]
            grid = np.linspace(0, 1, 101)
            ap[c, j] = np.trapz(np.interp(grid, rec, envelope), grid)
    best = 0
    if len(ids):
        mean_f1 = (2 * p[ids] * r[ids] / (p[ids] + r[ids] + 1e-16)).mean(axis=0)
        smooth = np.convolve(np.pad(mean_f1, (50, 50), mode="edge"), np.ones(101) / 101, mode="valid")
        best = int(smooth.argmax())
    rows = []
    for i, name in enumerate(names):
        rows.append(dict(class_id=i, class_name=name, instances=int(counts[i]),
                         precision=float(p[i, best]), recall=float(r[i, best]),
                         map50=float(ap[i, 0]), map50_95=float(ap[i].mean())))
    keys = ("precision", "recall", "map50", "map50_95")
    overall = {key: float(np.mean([rows[i][key] for i in ids])) if len(ids) else 0. for key in keys}
    return dict(**overall, per_class=rows, images=len(annotations))
