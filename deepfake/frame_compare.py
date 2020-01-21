

import numpy as np
import pathlib
import cv2
from mtcnn.mtcnn import MTCNN
import random


p = pathlib.Path(r'C:\Users\T149900\Downloads\dfdc_train_part_00\dfdc_train_part_0')

assert p.is_dir()

# Fake
v0 = p / "htorvhbcae.mp4"
assert v0.is_file()

# Real
v1 = p / "wclvkepakb.mp4"
assert v1.is_file()


vidcap0 = cv2.VideoCapture(str(v0))
video_fake = read_image(vidcap0)
vidcap0.release()

vidcap1 = cv2.VideoCapture(str(v1))
video_real, anFeatures = read_image_and_features(vidcap1)
vidcap1.release()





