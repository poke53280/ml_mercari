

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

aID = np.unique(df.ID)

res_list = []

# Estimate: Two hours processing on local host.

for x in aID:
    if x % 500 == 0 and x > 0:
        print(f"Processing element {x}/ {aID.shape[0]}")

    r, q = get_target_df(df, t_start, t_end, nGrow, x)
    res_list.append(r)




r, q = get_target_df(df, t_start, t_end, nGrow, 9009)


r

start_day = r['begin']

L_full = 1+ r['end'] - start_day

print(f"Length with stich = {L_full}")

L_adjusted = L_full - r['stitch']

print(f"Length adjusted = {L_adjusted}")


a = q.sort_values(by= 'F1')

aMD = a.MD
aD  = a.D

first_MD = aMD.values[0]
first_D  = aD.values[0]

ID = a.ID.values[0]

print(f"ID: {ID}: Start = {start_day}. L = {L_adjusted}. MD = {first_MD}, D = {first_D}")






q
          ID      B    S     F0     F1     T0     MD   D
109262  9009  24992  101  33232  33232  33244  53189  19
109263  9009  24992  101  33232  33245  33250  48262  19




















