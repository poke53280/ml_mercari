

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
from scipy.stats import skew, kurtosis

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

##################################################################################################
#
#       get_prefixed_dict
#

def get_prefixed_dict(d, prefix):
    d_prefixed = {}

    for key, value in d.items():
        d_prefixed[prefix + key] = value

    return d_prefixed

"""c"""

#
# Erik Vattekar
#
# V153753
#
#

def get_stats_on_array(v):

    if len(v) == 0:
        return {'mean': 0, 'std': 0, 'max': 0, 'min':0, 'sum': 0, 'skewness': 0, 'kurtosis': 0, 'median': 0, 'q1': 0, 'q3': 0,'count': 0}


    d = {'mean': v.mean(), 'std': v.std(), 'max': v.max(), 'min':v.min(), 'sum': v.sum(), 'skewness': skew(v), 'kurtosis': kurtosis(v), 'median': np.median(v),
         'q1': np.percentile(v, q=25), 'q3': np.percentile(v, q=75),'count': len(v)}

    return d

##################################################################################
#
#     get_historic_stats
#

def get_historic_stats(start_target_time, l_intervals):

    l_begin = []
    l_l     = []

    for a,b in l_intervals:
        assert a < start_target_time
        assert b < start_target_time
       
        l_l.append(b - a)

        a = start_target_time - a
        
        l_begin.append(a)
       

    """c"""

    anBegin = np.array(l_begin)    
    anLength = np.array(l_l)

    d = {}

    sBegin = get_prefixed_dict(get_stats_on_array(anBegin), 'S_')
    sLength = get_prefixed_dict(get_stats_on_array(anLength), 'L_')

    d.update(sBegin)
    d.update(sLength)

    return d

"""c"""


##################################################################################
#
#     generate_target_data
#

def get_unsigned_series(l):
    return pd.Series(pd.to_numeric(l, errors='raise', downcast = 'unsigned'))

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


    l_S_mean = []
    l_S_std = []
    l_S_max = []
    l_S_min = []
    l_S_sum = []
    l_S_skewness = []
    l_S_kurtosis = []
    l_S_median = []
    l_S_q1 = []
    l_S_q3 = [] 
    l_S_count = []
    l_L_mean = []
    l_L_std = []
    l_L_max = []
    l_L_min = []
    l_L_sum = []
    l_L_skewness = []
    l_L_kurtosis = []
    l_L_median = []
    l_L_q1 = []
    l_L_q3 = []
    l_L_count = []



    for x in aID:

        if x % 100 == 0 and x > 0:
            print(f"Processing {x}/ {aID.shape[0]}...")

        r = get_target_df(df, t_start, t_end, nGrow, x)

        assert x == r['ID']

        l_ID.append(x)
        l_N.append(r['nperiods'])

        d_stats = {}

        if r['nperiods'] == 0:
            l_start.append(0)
            l_L.append(0)
            l_Fill.append(0)
            l_MD.append(0)
            l_D.append(0)

            d_stats = get_historic_stats(start_target_time, [])

        else:   
            L_full = 1+ r['end'] - r['begin']
            L_adjusted = L_full - r['stitch']

            start_target_time = r['begin']
        
            l_start.append(start_target_time)
            l_L.append(L_adjusted)
            l_Fill.append(r['stitch'])
            l_MD.append(r['MD'])
            l_D.append(r['D'])

            d_stats = get_historic_stats(start_target_time, r['historic_intervals'])


        l_S_mean.append(d_stats['S_mean'])
        l_S_std.append(d_stats['S_std'])
        l_S_max.append(d_stats['S_max'])
        l_S_min.append(d_stats['S_min'])
        l_S_sum.append(d_stats['S_sum'])
        l_S_skewness.append(d_stats['S_skewness'])
        l_S_kurtosis.append(d_stats['S_kurtosis'])
        l_S_median.append(d_stats['S_median'])
        l_S_q1.append(d_stats['S_q1'])
        l_S_q3.append(d_stats['S_q3']) 
        l_S_count.append(d_stats['S_count'])

        l_L_mean.append(d_stats['L_mean'])
        l_L_std.append(d_stats['L_std'])
        l_L_max.append(d_stats['L_max'])
        l_L_min.append(d_stats['L_min'])
        l_L_sum.append(d_stats['L_sum'])
        l_L_skewness.append(d_stats['L_skewness'])
        l_L_kurtosis.append(d_stats['L_kurtosis'])
        l_L_median.append(d_stats['L_median'])
        l_L_q1.append(d_stats['L_q1'])
        l_L_q3.append(d_stats['L_q3']) 
        l_L_count.append(d_stats['L_count'])

    df_stat = pd.DataFrame( {'S_mean' : pd.Series(l_S_mean),
                             'S_std' :  pd.Series(l_S_std),
                             'S_max' :  pd.Series(l_S_max),
                             'S_min' :  pd.Series(l_S_min),
                             'S_sum' :  pd.Series(l_S_sum),
                             'S_skewness' : pd.Series(l_S_skewness),
                             'S_kurtosis' : pd.Series(l_S_kurtosis),
                             'S_median' : pd.Series(l_S_median),
                             'S_q1' : pd.Series(l_S_q1),
                             'S_q3' : pd.Series(l_S_q3),
                             'S_count' : pd.Series(l_S_count),
                             'L_mean' : pd.Series(l_L_mean),
                             'L_std' :  pd.Series(l_L_std),
                             'L_max' :  pd.Series(l_L_max),
                             'L_min' :  pd.Series(l_L_min),
                             'L_sum' :  pd.Series(l_L_sum),
                             'L_skewness' : pd.Series(l_L_skewness),
                             'L_kurtosis' : pd.Series(l_L_kurtosis),
                             'L_median' : pd.Series(l_L_median),
                             'L_q1' : pd.Series(l_L_q1),
                             'L_q3' : pd.Series(l_L_q3),
                             'L_count' : pd.Series(l_L_count)})


    sID = get_unsigned_series(l_ID)
    sStart = get_unsigned_series(l_start)
    sN = get_unsigned_series(l_N)
    sL = get_unsigned_series(l_L)
    sFill = get_unsigned_series(l_Fill)
    sMD = get_unsigned_series(l_MD)
    sD = get_unsigned_series(l_D)

    df_t = pd.DataFrame( {'ID': sID, 'S': sStart, 'N' : sN, 'MD':sMD, 'D':sD, 'F':sFill, 'Y':sL})

    df_t = pd.concat([df_t, df_stat], axis = 1)

    return df_t

"""c"""

t_start = 20000
t_end = 37000
nGrow = 15

nAllIDS = len (np.unique(df.ID))

nCut = 200000  #  = nAllIDS for no cut

# First cut historic data 
m = (df.ID < nCut)
df = df[m]

df_t = generate_target_data(df, t_start, t_end, nGrow, nCut)

# Move individual info from historic dataframe to id dataframe.

q = df.drop_duplicates(['ID'])
q = q.reset_index()

assert len(q) == nCut

df_t['B'] = q.B
df_t['K'] = q.S

df = df.drop(['B', 'S'], axis = 1)


s = df_t.S

df = df.assign(TCUT = df.ID.apply(lambda x: s[x]) )

m_cut = (df.F0 >= df.TCUT) | (df.F1 >= df.TCUT)

df = df[~m_cut]

df = df.reset_index(drop = True)

df = df.drop(['TCUT'], axis = 1)

nGotAdditionalData = len (np.unique(df.ID))
rAdditionalDataFactor = 100.0 * nGotAdditionalData/ nCut

print(f"Additional data elements: {rAdditionalDataFactor:.0f}%")

df_t.to_pickle(DATA_DIR + "df_t_10AUG2018.pkl")


