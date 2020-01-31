

import numpy as np


####################################################################################
#
#   _get_grid1D
#

def _get_grid1D(width, sample_size, num):
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
#   _get_grid2D
#

def _get_grid2D(width, height, num_x, num_y, sample_size):

    c_x = _get_grid1D(width, sample_size, num_x)
    c_y = _get_grid1D(height, sample_size, num_y)

    l_c = []

    for x in c_x:
        for y in c_y:
            l_c.append((x, y))

    return l_c


####################################################################################
#
#   _get_grid3D
#

def _get_grid3D(width, height, depth, num_x, num_y, num_z, sample_size):

    c_x = _get_grid1D(width, sample_size, num_x)
    c_y = _get_grid1D(height, sample_size, num_y)
    c_z = _get_grid1D(depth, sample_size, num_z)

    l_c = []

    for x in c_x:
        for y in c_y:
                for z in c_z:
                    l_c.append((x, y, z))


    return l_c

####################################################################################
#
#   _get_bb_from_centers_2D
#

def _get_bb_from_centers_2D(l_c, sample_size):
    assert sample_size % 2 == 0
    half_size = sample_size // 2

    l_bb = []

    for c in l_c:
        x_min = c[0] - half_size
        x_max = c[0] + half_size
        y_min = c[1] - half_size
        y_max = c[1] + half_size

        assert x_max - x_min == sample_size
        assert y_max - y_min == sample_size      
        
        l_bb.append((x_min, x_max, y_min, y_max))
                        
    return l_bb


####################################################################################
#
#   _get_bb_from_centers_3D
#

def _get_bb_from_centers_3D(l_c, sample_size):
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


####################################################################################
#
#   GetGrid2DCenters
#

def GetGrid2DCenters(width, height, cube_size, rOverlap):
    
    assert rOverlap > 1.0
    
    fit_cube = cube_size / rOverlap

    num_x = int (1 + width / fit_cube)
    num_y = int (1 + height / fit_cube)

    l_c = _get_grid2D(width, height, num_x, num_y, cube_size)

    return l_c


####################################################################################
#
#   GetGrid2DBB
#

def GetGrid2DBB(width, height, cube_size, rOverlap):
    l_c = GetGrid2DCenters(width, height, cube_size, rOverlap)
    l_bb = _get_bb_from_centers_2D(l_c, cube_size)
    return l_bb



####################################################################################
#
#   GetGrid3DCenters
#

def GetGrid3DCenters(width, height, depth, cube_size, rOverlap):
    
    assert rOverlap > 1.0
    
    fit_cube = cube_size / rOverlap

    num_x = int (1 + width / fit_cube)
    num_y = int (1 + height / fit_cube)
    num_z = int (1 + depth / fit_cube)

    l_c = _get_grid3D(width, height, depth, num_x, num_y, num_z, cube_size)

    return l_c


####################################################################################
#
#   GetGrid3DBB
#

def GetGrid3DBB(width, height, depth, cube_size, rOverlap):
    l_c = GetGrid3DCenters(width, height, depth, cube_size, rOverlap)
    l_bb = _get_bb_from_centers_3D(l_c, cube_size)
    return l_bb


####################################################################################
#
#   GetSubVolume3D
#

def GetSubVolume3D(cube, bb):
    im_min_x = bb[0]
    im_max_x = bb[1]
    im_min_y = bb[2]
    im_max_y = bb[3]
    im_min_z = bb[4]
    im_max_z = bb[5]

    return cube[im_min_x:im_max_x, im_min_y: im_max_y, im_min_z: im_max_z]

