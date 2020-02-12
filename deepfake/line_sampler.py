


from mp4_frames import get_output_dir
from mp4_frames import get_part_dir
from mp4_frames import get_video_path_from_stem_and_ipart
from mp4_frames import read_video
from image_grid import _get_bb_from_centers_3D
from image_grid import GetSubVolume3D


import numpy as np
import pandas as pd
import cv2
from multiprocessing import Pool


####################################################################################
#
#   get_line
#
#

def get_line(p0, p1):

    dp = p1 - p0
    dp = np.abs(dp)

    num_steps = np.max(dp)

    # t element of [0, 1]

    step_size = 1 / num_steps

    ai = np.arange(start = 0, stop = 1 + step_size, step = step_size)

    ai_t = np.tile(ai, 3).reshape(-1, ai.shape[0])


    p = (p1 - p0).reshape(3, -1) * ai_t

    p = p + p0.reshape(3, -1)

    p = np.round(p)

    return p



####################################################################################
#
#   load_sample_cubes
#

def load_sample_cubes(original, l_fakes, l_ac, nCubeSize, iPart):

    l_bb = _get_bb_from_centers_3D(l_ac, nCubeSize)

    l_video_file = []
    l_video_file.append(original)
    l_video_file.extend(l_fakes)

    d = nCubeSize // 2

    d_cubes = []

    for x in l_video_file:
        print(f"Creating cubes from {x}...")
        video = read_video_from_stem_and_ipart(x, iPart)

        l_cubes = []

        for bb in l_bb:
            cube = GetSubVolume3D(video, bb)
            assert cube.shape == (nCubeSize, nCubeSize, nCubeSize, 3)

            l_cubes.append(cube)
        
        d_cubes.append(l_cubes)


    """c"""
    return d_cubes



####################################################################################
#
#   rasterize_lines
#

def rasterize_lines(p, nLength):
    l_l = []

    for x in p:
        l = get_line(x[::2], x[1::2])
        assert l.shape[1] >= nLength, f"Line is short: {l.shape[1]}"

        l = np.swapaxes(l, 0, 1)
        l = l[:nLength]
        l = l.astype(np.int32)

        l_l.append(l)

    anLines = np.stack(l_l)
    return anLines


####################################################################################
#
#   sample_cube
#

def sample_cube(r, anLines):

    l_sample = []

    for l in anLines:

        l_x = l[:, 0]
        l_y = l[:, 1]
        l_z = l[:, 2]

        r_sample = r[l_z, l_y, l_x]
        l_sample.append(r_sample)

    anSamples = np.stack(l_sample)
    return anSamples
"""c"""






