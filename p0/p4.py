

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


"""c"""


##################################################################################
#
#     get_target_df
#

def get_target_df(df, t_start, t_end, nGrow, idx):

    lf1 = []
    lq = []

    m = (df.ID == idx)

    pt = df[m]
    
    if len(pt) == 0:
        pass
    else:
        lf1 = pt.F1.values
        lq = pt.T0.values

    r_m = np.array((lf1,lq)).T

    timelineText = TimeLineText(t_start, t_end, True, False, False, True)

    res = timelineText.GetTarget(r_m, nGrow)

    res['ID'] = idx

    if res['nperiods'] == 0:
        pass

    else:

        ids = res['ids']

        ids_target = pt.index[ids]
        m = pt.index.isin(ids_target)

        q = pt[m]

        a = q.sort_values(by= 'F1')

        aMD = a.MD
        aD  = a.D

        assert idx == a.ID.values[0]
        
        first_MD = aMD.values[0]
        first_D  = aD.values[0]

        res['MD'] = first_MD
        res['D'] = first_D

    return res

"""c"""

##################################################################################
#
#     generate_target_data
#

def generate_target_data(df, t_start, t_end, nGrow, nCut):

    aID = np.unique(df.ID)

    if nCut > 0:
        aID = aID[:nCut]

    l_ID = []
    l_start = []
    l_N = []
    l_L = []
    l_Fill = []
    l_MD = []
    l_D = []

    for x in aID:

        if x % 100 == 0 and x > 0:
            print(f"Processing {x}/ {aID.shape[0]}...")

        r = get_target_df(df, t_start, t_end, nGrow, x)

        assert x == r['ID']

        l_ID.append(x)
        l_N.append(r['nperiods'])

        if r['nperiods'] == 0:
            l_start.append(0)
            l_L.append(0)
            l_Fill.append(0)
            l_MD.append(0)
            l_D.append(0)
        else:   
            L_full = 1+ r['end'] - r['begin']
            L_adjusted = L_full - r['stitch']
        
            l_start.append(r['begin'])
            l_L.append(L_adjusted)
            l_Fill.append(r['stitch'])
            l_MD.append(r['MD'])
            l_D.append(r['D'])
        

    sID = pd.Series(l_ID)
    sStart = pd.Series(l_start)
    sN = pd.Series(l_N)
    sL = pd.Series(l_L)
    sFill = pd.Series(l_Fill)
    sMD = pd.Series(l_MD)
    sD = pd.Series(l_D)

    # df_t = pd.DataFrame( {'ID': sID, 'begin_target': sStart, 'num_periods' : sN, 'length_target' :sL, 'fill_target': sFill, 'MD':sMD, 'D':sD})

    df_t = pd.DataFrame( {'ID': sID, 'S': sStart, 'N' : sN, 'MD':sMD, 'D':sD, 'F':sFill, 'Y':sL})
    return df_t

"""c"""

t_start = 20000
t_end =   37000
nGrow = 15

nAllIDS = len (np.unique(df.ID))

nCut = 10000  # nAllIDS for no cut

df_t = generate_target_data(df, t_start, t_end, nGrow, nCut)

# Cut historic data as well

m = (df.ID < nCut)
df = df[m]

# Move individual info from historic dataframe to id dataframe.

q = df.drop_duplicates(['ID'])
q = q.reset_index()

assert len(q) == nCut

df_t['B'] = q.B
df_t['K'] = q.S

df = df.drop(['B', 'S'], axis = 1)

df_t = df_t[['ID', 'B', 'K', 'N', 'MD', 'D', 'S', 'F', 'Y']]

s = df_t.S

df = df.assign(TCUT = df.ID.apply(lambda x: s[x]) )

m_cut = (df.F0 >= df.TCUT) | (df.F1 >= df.TCUT)

df = df[~m_cut]

df = df.reset_index(drop = True)

df = df.drop(['TCUT'], axis = 1)

nGotAdditionalData = len (np.unique(df.ID))
rAdditionalDataFactor = 100.0 * nGotAdditionalData/ nCut

print(f"Additional data elements: {rAdditionalDataFactor:.0f}%")

# Got historic data in df, future cut.
# Got target information in df_t
