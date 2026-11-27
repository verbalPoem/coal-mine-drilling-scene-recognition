# YOLOv5 by Ultralytics and yolov5_obb contributors, GPL-3.0.
from torch import nn


def is_parallel(model):

    return type(model) in (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
