

# Configuration parameters

CONFIG_DATA_START = "2006-01-01"
CONFIG_DATA_INVALID_FUTURE = "2019-01-01"

# Resolving environment

import os

local_dir = os.getenv('LOCAL_PY_DIR')
assert local_dir is not None, "Set environment variable LOCAL_DIR to location of this python file."
print(f"Local python directoy is set to {local_dir}")
os.chdir(local_dir)

config_file = os.getenv('DB_DATA')
assert config_file is not None, "No config file found in environment variable DB_DATA"
print(f"Using database connection file '{config_file}'")


import sys
import pandas as pd
import json
import random
import numpy as np
import bisect
import datetime
import time


from p_general import DataProvider
from p_general import toDaysSinceEpoch
from p_general import apply_FID_COL
from p_general import classifyFID
from p_general import toDaysSinceEpochFromFID
from p_general import get_gender_from_fid

os.environ['NLS_NCHAR_CHARACTERSET']='AL16UTF16'
os.environ['NLS_CHARACTERSET']='WE8ISO8859P15'
os.environ['NLS_LANG']='AMERICAN_AMERICA.WE8ISO8859P15'


f = open(config_file,"r")
config = f.read()
f.close()

dp = DataProvider(config)

l_queries = []

l_queries.append( ("A", "sql_syk3",      "syk") )
#l_queries.append( ("A", "sql_fravar",    "fravar") )
l_queries.append( ("A", "sql_pmap",      "pmap") )
#l_queries.append( ("B", "sql_vedtak",    "vedtak") )
#l_queries.append( ("B", "sql_meldekort", "meldekort") )
#l_queries.append( ("C", "select_large",  "aa") )

d = dp.async_load(l_queries)

p = d['pmap'].copy()

p.columns = ['FID', 'FK', 'A']

m = p.FK == -1

p = p[~m]

# Clean up FID to FK mapping.

p_state = p["FID"].apply(classifyFID)

m = p_state == 'E'

print(f"Removing {len(p[m])} rows with bad fid")

p = p[~m]

p_state = p["FID"].apply(classifyFID)
p_epoch = p["FID"].apply(toDaysSinceEpochFromFID)

p = p.assign(S = p_state)
p = p.assign(E = p_epoch)

# Prepare conversion dictionaries

l_fid = p.FID.tolist()
l_fk  = p.FK.tolist()
l_a   = p.A.tolist()
l_s   = p.S.tolist()
l_e   = p.E.tolist()

d_FK_TO_FID = dict (zip (l_fk, l_fid))
d_FK_TO_E = dict (zip (l_fk, l_e))
d_FK_TO_S = dict (zip (l_fk, l_s))


########################################################################################################



df_syk = d['syk'].copy()

df_syk.columns = ["FK", "DID", "F0", "F1", "T0", "D"]

# Data stream start time as documented in project DPIA. Discard earlier data.

syk_data_begin = toDaysSinceEpoch(CONFIG_DATA_START)

m = (df_syk.F0 >= syk_data_begin)

nLinesAll = df_syk.shape[0]
nLinesCut = df_syk[m].shape[0]

print(f"Discarding data before {CONFIG_DATA_START}. Line count {nLinesAll} => {nLinesCut}")

df_syk = df_syk[m]


syk_data_future = toDaysSinceEpoch(CONFIG_DATA_INVALID_FUTURE)

m = (df_syk.F0 < syk_data_future) & (df_syk.F1 < syk_data_future) & (df_syk.T0 < syk_data_future)

nLinesAll = df_syk.shape[0]
nLinesCut = df_syk[m].shape[0]

print(f"Discarding data after {CONFIG_DATA_INVALID_FUTURE}, assumed bad. Line count {nLinesAll} => {nLinesCut}")

df_syk = df_syk[m]

def isExists(x, d):
    return x in d

m = df_syk.FK.apply(isExists, args = (d_FK_TO_FID,))

nLinesAll = df_syk.shape[0]
nLinesCut = df_syk[m].shape[0]

print(f"Discarding data where no FID <-> FK map found. Line count {nLinesAll} => {nLinesCut}")

df_syk = df_syk[m]

def getValue(x, d):
    return d[x]


fid = df_syk.FK.apply(getValue, args = (d_FK_TO_FID,))
birth = df_syk.FK.apply(getValue, args = (d_FK_TO_E,))
fid_s = df_syk.FK.apply(getValue, args = (d_FK_TO_S,))

df_syk = df_syk.assign(B = birth)
df_syk = df_syk.assign(FID = fid)
df_syk = df_syk.assign(FID_S = fid_s)

g = df_syk["FID"].apply(get_gender_from_fid)

df_syk = df_syk.assign(G = g)

q = df_syk.FID.astype('category').cat.codes

df_syk = df_syk.assign (IDX = q)

q = df_syk.DID.astype('category').cat.codes

df_syk = df_syk.assign (MD = q)

# drop later to verify deduced data. df_syk = df_syk.drop(["FK", "FID", "DID"], axis = 1)

df_syk = df_syk.assign(D = df_syk.D.astype('category'))
df_syk = df_syk.assign(FID_S = df_syk.FID_S.astype('category'))


# For brevity
df = df_syk

# Offset all dates into positive numbers for convenience.

first_day = np.min(  [df.F0.min(), df.F1.min(), df.T0.min(), df.B.min()])

df.F0 -= first_day
df.F1 -= first_day
df.T0 -= first_day
df.B -= first_day


min_day = np.min(  [df.F0.min(), df.F1.min(), df.T0.min(), df.B.min()])

assert min_day == 0

max_day = np.max(  [df.F0.max(), df.F1.max(), df.T0.max(), df.B.max()])

print(f"Max day after offset {first_day} : {max_day}")


def downcast_unsigned(s):
    assert s.min() >= 0
    return pd.to_numeric(s, downcast = 'unsigned')


# Convert data types to compact forms

df = df.assign(MD = downcast_unsigned(df.MD))
df = df.assign(IDX = downcast_unsigned(df.IDX))

df = df.assign(F0 = downcast_unsigned(df.F0))
df = df.assign(F1 = downcast_unsigned(df.F1))
df = df.assign(T0 = downcast_unsigned(df.T0))
df = df.assign(B = downcast_unsigned(df.B))

q = df_syk.FID_S.astype('category').cat.codes

df = df.assign(FID_S = q)
df = df.assign(FID_S = downcast_unsigned(df.FID_S))



df = df.drop(["MD", "FK", "DID", "G", "B"], axis = 1)
df = df.drop(["FID_S"], axis = 1)
df = df.drop(["FID"], axis = 1)

m = df.IDX < 100
df = df[m]

# Save point

DATA_DIR = "C:\\p_data\\"

df = pd.read_pickle(DATA_DIR + "tmp2.pkl")

df = df.sort_values(by = 'IDX')

df = df.reset_index(drop = True)

# Diagnose FE
q = df.D.apply(len)

df = df.assign(D_L = q)

q = df.D.apply(lambda x: x[0])


# First letter
df = df.assign(D_H = q)

q = df.D.apply(lambda x: x[1:])

df = df.assign(D_C = q)


df = df.assign(D_H = df.D_H.astype('category'))
df = df.assign(D_C = df.D_C.astype('category'))

q = df.D_H.cat.codes
df = df.assign(D_H = q)

q = df.D_C.cat.codes
df = df.assign(D_C = q)

q = df.D.cat.codes
df = df.assign(D = q)

df = df.assign(D = downcast_unsigned(df.D))
df = df.assign(G = df.G.astype('uint8'))

df = df.assign(IDX =  downcast_unsigned(df.IDX))

df = df.assign(D_L = downcast_unsigned(df.D_L))

df = df.assign(D_H = downcast_unsigned(df.D_H))

df = df.assign(D_C = downcast_unsigned(df.D_C))

# No value in D now
# DROP LATERdf = df.drop(['D'], axis = 1)

df = df.sort_values(by = ['IDX', 'F0', 'F1', 'T0'])

df = df.reset_index(drop=True)


df = df.drop(['FK', 'DID', 'FID','B', 'G'], axis = 1)

df = df.drop(['D', 'FK', 'DID', 'FID'], axis = 1)
df = df.drop(['FID_S', 'MD'], axis = 1)

# Todo

#
# Logical save point
#
# Main continutes below functions

##################################################################################
#
#     get_target_df
#


from TimeLineTool import TimeLineText

CONF_L_TARGET_MIN = 17
CONF_NGROW = 9
CONF_CUT_INTO_LEAVE = 7


def group_intervals(pt):

#idx = 97   => Early large F1 to T0. In addition to other run.
#Grad?

    lf1 = []
    lq = []

    lf1 = pt.F1.values
    lq = pt.T0.values

    r_m = np.array((lf1,lq)).T

    nEdgeAir = 2 * CONF_NGROW + 5

    # To ensure all intervals are within accepted range
    t_start = lf1.min() - nEdgeAir
    t_end = lq.max() + nEdgeAir


    timelineText = TimeLineText(t_start, t_end, True, False, False, True)


    r_m_excl = r_m.copy()
    r_m_excl[:, 1] += 1

    r_m_processed = timelineText.CombineIntervals(r_m_excl, CONF_NGROW)

    # Back to inclusive mode
    r_m_processed[:, 1] -= 1

    assert len(r_m_processed) > 0, "No resulting groups"


    group_idx = np.zeros(r_m.shape[0])

    group_idx[:] = -1

    r_m_start = r_m[:, 0]
    r_m_end   = r_m[:, 1]

    # Place all intervals into the groups

    for idx, p in enumerate(r_m_processed):
        a = p[0]
        b = p[1]

        # Fully inside range:
        m = (r_m_start >= a) & (r_m_end <= b)

        # Inside range a, b:
        # nInside = len(r_m[m])

        group_idx[m] = idx

        # print(f"#intervals in range [{a}, {b}]: {nInside}")

    assert (group_idx < 0).sum() == 0, "Input interval(s) not assigned to any group"
    assert len(np.unique(group_idx)) == len(r_m_processed), "Found empty groups"


    return r_m_processed, group_idx

"""c"""


# Returns all data in focus as indices into input array r_m_processed. Last element is a valid target interval.
# May return empty index array when no valid target interval is found.

def getValidDataIntervals(r_m_processed):


    # Assert sorted in ascending order
    begin = r_m_processed[:, 0]
    end = r_m_processed[:, 1]

    begin_s = np.sort(begin)
    end_s = np.sort(end)

    assert (begin != begin_s).sum() == 0
    assert (end != end_s).sum() == 0


    # Filter out too early and too late

    nEarly = 32000  # -1 to disable early filter.
    nLate = 33600   # -1 to disable late filter.

    lEarly = []
    lLate = []

    lInterest = []

    for idx, p in enumerate(r_m_processed):
        a = p[0]
        b = p[1]

        if nEarly > 0 and b < nEarly:
            print(f"Early: {idx}")
            lEarly.append(idx)

        elif nLate > 0 and a > nLate:
            print(f"Late: {idx}")
            lLate.append(idx)

        else:
            lInterest.append(idx)


    if len(lInterest) == 0:
        print("No data in interest zone")
    
    idx_target_candidate = -1

    for idx in lInterest:

        p = r_m_processed[idx]

        target_begin, target_end, target_stitch = p[0], p[1], p[2]

        L_full = 1 + target_end - target_begin
        L_adj  = L_full - target_stitch

        
        print(f"Feature space {idx}. L_adj = {L_adj}. L_full = {L_full}")

        if L_adj >= CONF_L_TARGET_MIN:
            print(f"{idx} is a target candidate")
            idx_target_candidate = idx

    """c"""

    m = np.array(lInterest) <= idx_target_candidate  # None smaller for no target found => empty array output

    p_data_idx = np.array(lInterest)[m]

    return p_data_idx

"""c"""


def full_analysis(q):

    assert len (q) > 0
    assert np.unique(q.IDX).shape[0] == 1, "Multiple IDs input"

    r_m_processed, group_idx = group_intervals(q)

    ids_valid_intervals = getValidDataIntervals(r_m_processed)

    s = set (range(len(r_m_processed)))
    s -= set(ids_valid_intervals)

    ids_discarded_intervals = np.array(list(s))  # Out of range or in future to target interval

    assert ids_valid_intervals.shape[0] + ids_discarded_intervals.shape[0] == r_m_processed.shape[0], "ids_valid_intervals.shape[0] + ids_discarded_intervals.shape[0] == r_m_processed.shape[0]"

    isNoData = ids_valid_intervals.shape[0] == 0

    if isNoData:
        print(f"No target found for {q.IDX.values[0]}")
        return (-1, -1, None, None)

    
    ids_feature = ids_valid_intervals[:-1]      # Can be empty: Valid, common condition.

    idx_target = ids_valid_intervals[-1]

    m_discarded = np.isin(group_idx, ids_discarded_intervals)


    # Method end product:
    m_feature = np.isin(group_idx, ids_feature)

    targetData = r_m_processed[idx_target]

    target_begin, target_end, target_stitch = targetData

    L_full = 1 + target_end - target_begin
    L_adj  = L_full - target_stitch

    assert L_adj >= CONF_L_TARGET_MIN, "Interval length < configured min target length"

    target_cut_day = target_begin + CONF_CUT_INTO_LEAVE

    m_valid_train_range = np.array(q.F1 <= target_cut_day)

    m_target = np.isin(group_idx, idx_target)

    # All elements: Discarded, in feature or in target

    anCount = np.zeros(q.shape[0])

    anCount[m_target] += 1
    anCount[m_feature] += 1
    anCount[m_discarded] += 1

    assert np.min(anCount) == 1 and np.max(anCount) == 1

    m_valid_train_target = m_target & m_valid_train_range

    # Data for target leave interval available CONF_CUT_INTO_LEAVE days(s) into leave.

    return (target_begin, L_adj, m_feature, m_valid_train_target)

"""c"""

test_idx = 97

m = (df.IDX == test_idx)
q = df[m]


begin, L, m_f, m_t = full_analysis(q)

print(f"Begin = {begin}. L = {L}")

df = df.assign(S = 'E')


l_Begin = []
l_L     = []

l_IDX = list (range(100))

for x in range(100):
    m = (df.IDX == x)
    q = df[m]

    print(f"Analyzing {x}...")

    begin, L, m_f, m_t = full_analysis(q)

    if L < 0:
        print("No target found")
        l_Begin.append(np.nan)
        l_L.append(np.nan)

    else:
        print(f"IDX = {x}. Begin = {begin}. L = {L}")

        l_Begin.append(begin)
        l_L.append(L)

        q.loc[m_f] = q.loc[m_f].assign(S = 'F')
        q.loc[m_t] = q.loc[m_t].assign(S = 'T')

        df.loc[m] = q


 
# Target df

df_target = pd.DataFrame({'ID': l_IDX, 'Begin': l_Begin,'Y': l_L })


m = df.S != 'E'


df = df[m]   # Remove invalid and target data


#
# Return all data on which to train. Separated in background data and early target data.
#
# Start of leave
#
# Adjusted length of leave (Y).
#


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

def get_stats_on_array(v):

    if len(v) == 0:
        return {'count': 0, 'mean': 0, 'std': 0, 'max': 0, 'min':0, 'sum': 0, 'skewness': 0, 'kurtosis': 0, 'median': 0, 'q1': 0, 'q3': 0}


    d = {'count': len(v), 'mean': v.mean(), 'std': v.std(), 'max': v.max(), 'min':v.min(), 'sum': v.sum(), 'skewness': skew(v), 'kurtosis': kurtosis(v), 'median': np.median(v),
         'q1': np.percentile(v, q=25), 'q3': np.percentile(v, q=75)}

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

    aID = np.unique(df.IDX)

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

n0 = df.shape[0]

df = df.drop_duplicates()

n1 = df.shape[0]

print(f"Dropped duplicates: n: {n0} => {n1}")


t_start = 20000
t_end = 37000
nGrow = 15

nAllIDS = len (np.unique(df.IDX))

nCut = nAllIDS  #  = nAllIDS for no cut

# First cut num unique users 
m = (df.IDX < nCut)

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

df = df.assign(TCUT = df.IDX.apply(lambda x: s[x]) )

m_cut = (df.F0 >= df.TCUT) | (df.F1 >= df.TCUT)

df = df[~m_cut]

df = df.reset_index(drop = True)

df = df.drop(['TCUT'], axis = 1)

nGotAdditionalData = len (np.unique(df.IDX))
rAdditionalDataFactor = 100.0 * nGotAdditionalData/ nCut

print(f"Additional data elements: {rAdditionalDataFactor:.0f}%")

df_t.to_pickle(DATA_DIR + "df_t_14AUG2018.pkl")
df.to_pickle(DATA_DIR + "df_14AUG2018.pkl")