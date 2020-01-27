

import numpy as np

####################################################################################
#
#   get_grid1D
#

def get_grid1D(width, sample_size, num):
    assert sample_size % 2 == 0
    half_size = sample_size // 2

    anC = np.linspace(start = half_size, stop = width - half_size, num = num)

    anC = np.round(anC).astype(np.int32)

    x_min = anC - half_size
    x_max = anC + half_size

    overlap_length = (anC[0] + half_size) - (anC[1] - half_size)

    print(f"Overlap: {overlap_length}")

    return anC



####################################################################################
#
#   get_grid2D
#

def get_grid2D(width, height, sample_size, num_x, num_y):

    c_x = get_grid1D(width, sample_size, num_x)
    c_y = get_grid1D(height, sample_size, num_y)

    l_c = []

    for x in c_x:
        for y in c_y:
            l_c.append((x, y))

    for c in l_c:
        x_min = c[0] - half_size
        x_max = c[0] + half_size
        y_min = c[1] - half_size
        y_max = c[1] + half_size

        assert x_max - x_min == sample_size
        assert y_max - y_min == sample_size

    return l_c





width = 1920
height = 1080
sample_size = 400
num_x = 10
num_y = 5

get_grid2D(width, height, sample_size, num_x, num_y)

