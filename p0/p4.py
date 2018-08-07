

###############################################################################
#
#       Analysis - timings with TimeLineTool.py
#
#

import pandas as pd
import numpy as np
import more_itertools as mit

import sys
import os

import numpy as np

sys.path

cwd = os.getcwd()

os.chdir('C:\\Users\\T149900\\ml_mercari')

from general.TimeLineTool import *


DATA_DIR_PORTABLE = "C:\\p_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE


df = pd.read_pickle(DATA_DIR + "noised_30JUL2018_cleaned.pkl")

df.shape[0]
df = df.drop_duplicates()
df.shape[0]



# Prepare dataframe for TimeLineTool.

df_m = pd.DataFrame()

df_m['IDX'] = df.ID  #Individual ID.
df_m['F1'] =  df.F1  #From day inclusive
df_m['Q'] =   df.T0  #To day inclusive

m = (df_m.Q < df_m.F1)

assert len (df[m]) == 0, "Invalid intervals given"

# t_start, t_end defined a time line interval:

t_start, t_end  = TimeLineText.GetPaddedFullRangeMinMax(df_m)


# Display one line:

is_draw_small = False                                           # Maintain day resolution to not lose small intervals.
is_draw_point_only = False
isVerbose = False
isClipEdge = True
growConst = 2

line_size = t_end - t_start

timelineText = TimeLineText(t_start, t_end, line_size, is_draw_small, is_draw_point_only, isVerbose, isClipEdge)

timelineText.DescribeScale()

idx = 194

l = timelineText.GetTargetInterval(df_m, idx)

target_begin = l[0]
target_end   = l[1]- 1    # -1: Now including last given day

target_length = 1 + target_end - target_begin

print(f"idx = {idx}: [{target_begin}, {target_end}], L = {target_length}")

#
#
# Preliminary preprocessing:
#
#
# For one individual:
#
# Create intervals as above/ p3.py
#
# Rasterize, see overlaps. Take note of warnings.
# For now - discard overlapping.
#
#
# Find air distances.
#
# Take note of small distances. (1,2,3,4,5)
#
# For now discard air gap cases.




def get_target(r_m, t_start, t_end, nGrow):

    r_m_excl = r_m.copy()
    r_m_excl[:, 1] += 1

    line_size = t_end - t_start

    timelineText = TimeLineText(t_start, t_end, line_size, True, False, False, True)

    r_m_processed = timelineText.CombineIntervals(r_m_excl, nGrow)

    # Go to end inclusive mode
    r_m_processed[:, 1] -= 1

    #
    # Go back and find where all sub elements are to be found.
    #
    # All elements are to be found, and fully contained in one and only one combination.
    #

    # Working on inclusive end values - i.e. both begin and end contained in interval.

    r_m_start = r_m[:, 0]
    r_m_end   = r_m[:, 1]

    group_idx = np.zeros(r_m.shape[0])

    nAssignedElements = 0

    for idx, p in enumerate(r_m_processed):
        a = p[0]
        b = p[1]

        # Fully inside range:
        m = (r_m_start >= a) & (r_m_end <= b)

        # Inside range a, b:
        nInside = len(r_m[m])

        group_idx[m] = idx

        # print(f"#intervals in range [{a}, {b}]: {nInside}")

        nAssignedElements = nAssignedElements + nInside

    """c"""

    assert nAssignedElements == r_m.shape[0], f"Not all intervals are grouped"


    isEmpty = len(r_m_processed) == 0

    result = {}
    result['nperiods'] = len(r_m_processed)

    if isEmpty:
        print("No target found")
    else:
        target_idx = len (r_m_processed) - 1

        target_info = r_m_processed[-1]

        target_begin, target_end, target_stitch = target_info[0], target_info[1], target_info[2]

        result['begin'] = target_begin
        result['end'] = target_end
        result['stitch'] = target_stitch

        result['ids'] = []
   
        m = (group_idx == target_idx)

        interval_list = np.where(m)

        for x in interval_list[0]:
            val = r_m[x]
            result['ids'].append(x)

        """c"""

    return result

"""c"""    
    



# Input: Unsorted non null intervals. Both end points are inclusive elements.

# Test

r_m = np.array( [ [4,4], [11, 13], [10, 16], [17, 18], [20, 21], [22, 24], [28, 29], [40, 45] ])

t_start = -3
t_end = 54

nGrow = 2

get_target(r_m, t_start, t_end, nGrow)

# Test end



t_start = 20000
t_end =   35000
idx = 92
nGrow = 2


def get_target_df(df, t_start, t_end, nGrow, idx):

    lf1 = []
    lq = []

    m = (df.ID == idx)

    pt = df[m]

    # print(f"# Input rows: {len(pt)}")
    
    if len(pt) == 0:
        pass
    else:
        lf1 = pt.F1.values
        lq = pt.T0.values


    r_m = np.array((lf1,lq)).T

    res = get_target(r_m, t_start, t_end, nGrow)

    ids = res['ids']

    ids_target = pt.index[ids]
    m = pt.index.isin(ids_target)

    q = pt[m]

    return res, q

"""c"""

aID = np.unique(df.ID)

for x in aID:
    r, q = get_target_df(df, t_start, t_end, 2, x)
    

# BUG:    

== > 26221, 
assert nAssignedElements == r_m.shape[0], f"Not all intervals are grouped"

# See verbose, seems single day or two day period moved ('d'). This can cause original intervals to be outside cluster interval.





























