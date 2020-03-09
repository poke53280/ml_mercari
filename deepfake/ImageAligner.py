

import cv2
from mp4_frames import get_ready_data_dir
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from image_grid import GetGrid2DBB


###################################################################################
#
#      align_frames
#

def align_frames(path, border, filename_input, filename_output, num_images):


    im_A = Image.open(path / f"{filename_input}_000.png")

    width = im_A.width
    height = im_A.height

    anA = np.asarray(im_A)

    border = 30
    region = (border, width - border, border, height - border)


    anA_ = anA[region[2] : region[3], region[0] : region[1]]


    plt.imsave(path / f"{filename_output}_000.png", anA_)

    for i in range (1, num_images):
    
        im_B = Image.open(path / f"{filename_input}_{i:03}.png")

        anB_ = align_images(anA_, np.asarray(im_B), border)

        plt.imsave(path / f"{filename_output}_{i:03}.png", anB_)

        anA_ = anB_



###################################################################################
#
#      align_images
#

def align_images(anA_, anB, border):

    width = anB.shape[1]
    height = anB.shape[0]


    def diff_offset(anA_, anB, offsetX, offsetY, region):
        
        anB = anB[region[2] + offsetY : region[3] + offsetY, region[0] + offsetX: region[1] + offsetX]
        rms = np.sqrt(np.sum((anA_-anB)**2))
        return rms


    border = 30

    region = (border, width - border, border, height - border)

    assert anA_.shape == (region[1] - region[0], region[3] - region[2], 3)


    best_rms = 1000000
    best_x = 0
    best_y = 0


    min_offset_x =  - region[0]
    min_offset_y =  - region[2]

    max_offset_x = width - region[1]
    max_offset_y = height - region[3]

    x_min = np.max([-border, min_offset_x])
    y_min = np.max([-border, min_offset_y])

    x_max = np.min([border, max_offset_x])
    y_max = np.min([border, max_offset_y])

    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            rms = diff_offset(anA_, anB, x, y, region)

            if rms < best_rms:
                best_x = x
                best_y = y
                best_rms = rms

    return anB[region[2] + best_y: region[3] + best_y, region[0] + best_x : region[1] + best_x]


###################################################################################
#
#      example
#

def example():
    path = get_ready_data_dir()
    border = 30
    filename_input = "test"
    filename_output = "cut_test"
    num_images = 128

    align_frames(path, border, filename_input, filename_output, num_images)


# Find most busy sub-frame


im_A = Image.open(path / f"cut_test_000.png")
im_B = Image.open(path / f"cut_test_001.png")
im_C = Image.open(path / f"cut_test_002.png")
im_D = Image.open(path / f"cut_test_003.png")


anA = np.asarray(im_A)
anB = np.asarray(im_B)
anC = np.asarray(im_C)
anD = np.asarray(im_D)

width = im_A.width
height = im_A.height


anData = np.stack([anA, anB, anC, anD])

l_BB = GetGrid2DBB(width, height, 50, 1.4)

x = l_BB[0]

anDataChunk = anData[:, x[2] : x[3], x[0]:x[1], :]


anDataChunk.shape

