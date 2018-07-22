

DATA_DIR_PORTABLE = "C:\\tgs_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE


import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split


TRAIN_IMAGE_DIR = DATA_DIR + 'train/images/'
TRAIN_MASK_DIR = DATA_DIR + 'train/masks/'
TEST_IMAGE_DIR = DATA_DIR + 'test/images/'

train_fns = os.listdir(TRAIN_IMAGE_DIR)

X = [np.array(cv2.imread(TRAIN_IMAGE_DIR + p, cv2.IMREAD_GRAYSCALE), dtype=np.uint8) for p in tqdm(train_fns)]