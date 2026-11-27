# Ultralytics YOLO, AGPL-3.0 license
__version__ = "8.2.103"

from ultralytics.models.yolo import YOLO
from ultralytics.utils import ASSETS, SETTINGS

settings = SETTINGS
__all__ = ("YOLO", "ASSETS", "SETTINGS", "settings", "__version__")
