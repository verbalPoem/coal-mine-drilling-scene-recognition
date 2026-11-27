# YOLOv5 by Ultralytics and yolov5_obb contributors, GPL-3.0.
import cv2
import numpy as np

pi = 3.141592


def gaussian_label_cpu(label, num_class, u=0, sig=4.0):
    """Return a circular Gaussian label for an angle bin."""
    x = np.arange(-num_class / 2, num_class / 2)
    y_sig = np.exp(-((x - u) ** 2) / (2 * sig**2))
    index = int(num_class / 2 - label)
    return np.concatenate([y_sig[index:], y_sig[:index]], axis=0)


def regular_theta(theta, mode="180", start=-pi / 2):
    """Wrap an angle to the interval starting at start."""
    assert mode in ["360", "180"]
    cycle = 2 * pi if mode == "360" else pi

    theta = theta - start
    theta = theta % cycle
    return theta + start


def poly2rbox(polys, num_cls_thata=180, radius=6.0, use_pi=False, use_gaussian=False):
    """Fit long-side rectangles to polygons and optionally construct circular labels."""
    assert polys.shape[-1] == 8
    if use_gaussian:
        csl_labels = []
    rboxes = []
    for poly in polys:
        poly = np.float32(poly.reshape(4, 2))
        (x, y), (w, h), angle = cv2.minAreaRect(poly)
        angle = -angle
        theta = angle / 180 * pi

        if w != max(w, h):
            w, h = h, w
            theta += pi / 2
        theta = regular_theta(theta)
        angle = (theta * 180 / pi) + 90

        if not use_pi:
            rboxes.append([x, y, w, h, angle])
        else:
            rboxes.append([x, y, w, h, theta])
        if use_gaussian:
            csl_label = gaussian_label_cpu(
                label=angle, num_class=num_cls_thata, u=0, sig=radius
            )
            csl_labels.append(csl_label)
    if use_gaussian:
        return np.array(rboxes), np.array(csl_labels)
    return np.array(rboxes)
