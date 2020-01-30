

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

    # print(f"Overlap: {overlap_length}")

    return anC


####################################################################################
#
#   get_grid3D_GridSize
#

def get_grid3D_gridsize(width, height, depth, num_x, num_y, num_z, sample_size):

    c_x = get_grid1D(width, sample_size, num_x)
    c_y = get_grid1D(height, sample_size, num_y)
    c_z = get_grid1D(depth, sample_size, num_z)

    l_c = []

    for x in c_x:
        for y in c_y:
                for z in c_z:
                    l_c.append((x, y, z))


    return l_c


####################################################################################
#
#   get_grid3D_overlap
#

def get_grid3D_overlap(width, height, depth, cube_size, rOverlap):
    assert rOverlap > 1.0
    fit_cube = cube_size / rOverlap

    num_x = int (1 + width / fit_cube)
    num_y = int (1 + height / fit_cube)
    num_z = int (1 + depth / fit_cube)

    l_c = get_grid3D_gridsize(width, height, depth, num_x, num_y, num_z, cube_size)

    return l_c


####################################################################################
#
#   get_bb_from_centers_3D
#

def get_bb_from_centers_3D(l_c, sample_size):
    assert sample_size % 2 == 0
    half_size = sample_size // 2

    l_bb = []

    for c in l_c:
        x_min = c[0] - half_size
        x_max = c[0] + half_size
        y_min = c[1] - half_size
        y_max = c[1] + half_size
        z_min = c[2] - half_size
        z_max = c[2] + half_size

        assert x_max - x_min == sample_size
        assert y_max - y_min == sample_size      
        assert z_max - z_min == sample_size 
        
        l_bb.append((x_min, x_max, y_min, y_max, z_min, z_max))
                        
                        
    return l_bb



