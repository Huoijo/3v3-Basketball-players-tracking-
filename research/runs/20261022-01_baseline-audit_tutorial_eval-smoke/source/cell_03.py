# ==============================
# 1. Environment preparation
# ==============================

!pip install ultralytics --quiet
!pip install ndjson --quiet
!pip install easyocr --quiet
!pip install gdown --quiet

import os
import glob
import re
import cv2
import ndjson
import math
import colorsys
import torch
import easyocr

import numpy as np
import pandas as pd

from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from IPython.display import HTML
from base64 import b64encode

# Check GPU availability
assert torch.cuda.is_available(), "Please enable a GPU runtime in Colab."

print("GPU:", torch.cuda.get_device_name(0))