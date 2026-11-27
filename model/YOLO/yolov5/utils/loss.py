# YOLOv5 by Ultralytics and yolov5_obb contributors, GPL-3.0.
import torch
from torch import nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):

    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ComputeLoss:
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device
        h = model.hyp

        BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h["cls_pw"]], device=device)
        )
        BCEtheta = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h["theta_pw"]], device=device)
        )
        BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h["obj_pw"]], device=device)
        )

        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))

        g = h["fl_gamma"]
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
            BCEtheta = FocalLoss(BCEtheta, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]
        self.stride = det.stride
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])

        self.ssi = list(self.stride).index(16) if autobalance else 0
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = (
            BCEcls,
            BCEobj,
            1.0,
            h,
            autobalance,
        )
        self.BCEtheta = BCEtheta
        for k in "na", "nc", "nl", "anchors":
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):
        """Compute box, objectness, class and angle losses."""
        device = targets.device
        lcls, lbox, lobj = (
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
        )
        ltheta = torch.zeros(1, device=device)

        tcls, tbox, indices, anchors, tgaussian_theta = self.build_targets(p, targets)

        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)

            n = b.shape[0]
            if n:
                ps = pi[b, a, gj, gi]

                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                lbox += (1.0 - iou).mean()

                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = (
                        b[sort_id],
                        a[sort_id],
                        gj[sort_id],
                        gi[sort_id],
                        score_iou[sort_id],
                    )
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou

                class_index = 5 + self.nc
                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:class_index], self.cn, device=device)
                    t[range(n), tcls[i]] = self.cp

                    lcls += self.BCEcls(ps[:, 5:class_index], t)

                t_theta = tgaussian_theta[i].type(ps.dtype)
                ltheta += self.BCEtheta(ps[:, class_index:], t_theta)

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]
            if self.autobalance:
                self.balance[i] = (
                    self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
                )

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        ltheta *= self.hyp["theta"]
        bs = tobj.shape[0]

        return (lbox + lobj + lcls + ltheta) * bs, torch.cat(
            (lbox, lobj, lcls, ltheta)
        ).detach()

    def build_targets(self, p, targets):
        """Match grid anchors to labeled oriented rectangles."""

        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []

        tgaussian_theta = []

        feature_wh = torch.ones(2, device=targets.device)
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)

        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

        g = 0.5
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],
                ],
                device=targets.device,
            ).float()
            * g
        )

        for i in range(self.nl):
            anchors = self.anchors[i]

            feature_wh[0:2] = torch.tensor(p[i].shape)[[3, 2]]

            t = targets.clone()
            t[:, :, 2:6] /= self.stride[i]
            if nt:
                r = t[:, :, 4:6] / anchors[:, None]
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]

                t = t[j]

                gxy = t[:, 2:4]

                gxi = feature_wh[[0, 1]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]

            gaussian_theta_labels = t[:, 7:-1]
            gij = (gxy - offsets).long()
            gi, gj = gij.T

            a = t[:, -1].long()

            indices.append(
                (
                    b,
                    a,
                    gj.clamp_(0, int(feature_wh[1].item()) - 1),
                    gi.clamp_(0, int(feature_wh[0].item()) - 1),
                )
            )
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)

            tgaussian_theta.append(gaussian_theta_labels)

        return tcls, tbox, indices, anch, tgaussian_theta
