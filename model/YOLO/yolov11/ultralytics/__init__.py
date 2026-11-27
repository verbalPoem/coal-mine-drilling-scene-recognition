# Ultralytics AGPL-3.0 License

import os

__version__ = "8.3.24"
os.environ.setdefault("OMP_NUM_THREADS", "1")

from ultralytics.models import YOLO
from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

settings = SETTINGS
__all__ = ("YOLO", "ASSETS", "settings", "checks", "download", "__version__")
