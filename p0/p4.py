

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



def get_target(r_m, t_start, t_end, nGrow):

    r_m_excl = r_m.copy()
    r_m_excl[:, 1] += 1

    line_size = t_end - t_start

    timelineText = TimeLineText(t_start, t_end, True, False, False, True)

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


t_start = 20000
t_end =   35000
nGrow = 2


# OK
r_m = np.array([[27534, 27534]])
res = get_target(r_m, t_start, t_end, nGrow)

# BAD
r_m = np.array([[27535, 27535]])
res = get_target(r_m, t_start, t_end, nGrow)

# BAD
r_m = np.array([[27536, 27536]])
res = get_target(r_m, t_start, t_end, nGrow)

# BAD
r_m = np.array([[27537, 27537]])
res = get_target(r_m, t_start, t_end, nGrow)

# OK
r_m = np.array([[27538, 27538]])
res = get_target(r_m, t_start, t_end, nGrow)



aID = np.unique(df.ID)

res_list = []

for x in range(26222, 500000):
    print(f"Processing element {x}/ {aID.shape[0]}")

    r, q = get_target_df(df, t_start, t_end, 2, x)
    res_list.append(r)

    


t_start = 20000
t_end =   35000
idx = 92
nGrow = 2


























