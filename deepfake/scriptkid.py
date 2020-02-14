

import os, sys, time
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import pathlib


isKaggle = pathlib.Path("/kaggle/input").is_dir()


if isKaggle:
    os.chdir('/kaggle/input/pythoncode')

from mp4_frames import get_test_dir



if isKaggle:
    os.chdir('/kaggle/working')


test_dir = get_test_dir()

test_videos = sorted([x for x in os.listdir(test_dir) if x[-4:] == ".mp4"])
frame_h = 5
frame_l = 5
len(test_videos)

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gpu

sys.path.insert(0, "/kaggle/input/blazeface-pytorch")
sys.path.insert(0, "/kaggle/input/deepfakes-inference-demo")

from blazeface import BlazeFace