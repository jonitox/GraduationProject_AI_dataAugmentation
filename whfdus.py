import math
import os
import time
from math import ceil

import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

from conv2d import Conv2d
from max_pool_2d import MaxPool2d
import datetime
import io

np.set_printoptions(threshold=np.nan)

