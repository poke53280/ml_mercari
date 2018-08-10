

import pandas as pd
import numpy as np

from general.KDE_study import group_sorted_unique_integers
from scipy.stats import skew, kurtosis

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


import lightgbm as lgb


DATA_DIR_PORTABLE = "C:\\p_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE



##################################################################################
#
#       get_group_list
#
#
                
def get_group_list(g, colname):
    l = list (g[colname])
    assert len (l) > 0

    l = l[0][1]
    l = l.values

    return l


##################################################################################
#
#       get_day_set
#
#

def get_day_set(df, m):

    def add_to_set(x, my_set):
        begin = x['begin']
        end   = x['end']
        my_set.update ( range(begin, end + 1))

    days = set()

    _ = df[m].apply(add_to_set, axis = 1, args = (days, ))

    anDays = np.array(list(days))
    anDays = np.sort(anDays)

    return anDays


##################################################################################
#
#       group_periods
#
#

def group_periods(df, p_id, n_bandwidth):

    m = (df.P == p_id)

    an = get_day_set(df, m)

    l = group_sorted_unique_integers(an, n_bandwidth, False)
    
    return l

"""c"""


##################################################################################################
#
#       extract_targets
#

def extract_targets(df, Target_begin_span, n_bandwidth):


    targets = []

    anRows = df.P.unique()

    nRows = len (anRows)

    assert anRows.max() +1 == nRows

    print(f"Extracting targets in range {Target_begin_span[0]} - {Target_begin_span[1]}...")


    for i in range(nRows):
        l = group_periods(df, i, n_bandwidth)

        if i%500 == 0:
            print(f"Processing ID = {i} out of {nRows}...")

        # Get biggest 'begin' in Target_begin_span range

        begins = [i[0] for i in l]
        anBegins = np.array([begins])

        assert (anBegins == np.sort(anBegins)).all()

        m = ((anBegins >= Target_begin_span[0] ) & (anBegins <= Target_begin_span[1]))

        m_idx = np.where(m)[1]

        if len (m_idx) == 0:
            # No candidate found
            pass
        else:
            idx_max = np.max(m_idx)

            target_interval = l[idx_max]
            begin_interval = target_interval[0]
            end_interval = target_interval[1]

            target = {}
            target['id'] = i
            
            L = 1 + end_interval - begin_interval
            
            target['L'] = L
            target['Begin'] = begin_interval

            targets.append(target)
                

    rPct = 100.0 * len(targets) / nRows
    print(f"{len(targets)} of {nRows} ({rPct:.1f}%) IDs in target zone")


    df_t = pd.DataFrame(targets)

    df_t.columns = ['T0', 'Y', 'id']

    df_t = df_t[['id', 'T0', 'Y']]

    return df_t


##################################################################################################
#
#       get_stats_on_array
#

def get_stats_on_array(v):

    if len(v) == 0:
        return {'mean': 0, 'std': 0, 'max': 0, 'min':0, 'sum': 0, 'skewness': 0, 'kurtosis': 0, 'median': 0, 'q1': 0, 'q3': 0,'count': 0}


    d = {'mean': v.mean(), 'std': v.std(), 'max': v.max(), 'min':v.min(), 'sum': v.sum(), 'skewness': skew(v), 'kurtosis': kurtosis(v), 'median': np.median(v),
         'q1': np.percentile(v, q=25), 'q3': np.percentile(v, q=75),'count': len(v)}

    return d




##################################################################################################
#
#       no_column_info
#

def no_column_info(is_create_stats):

    d_now = { 'md_now': 0, 'd_now': 0 }
    d_uniques = { 'uniq_md' : 0, 'uniq_d' : 0 }

    d_begin = {}
    d_end = {}
    d_period = {}

    if is_create_stats:
        d_begin = get_prefixed_dict (get_stats_on_array(np.array([])), 'begin_')
        d_end   = get_prefixed_dict (get_stats_on_array(np.array([])), 'end_')
        d_period = get_prefixed_dict (get_stats_on_array(np.array([])), 'span_')
    
    d_serialized = {'begin':"", 'end':"", 'md':"", 'd':""}

    d_acc = {}

    d_acc.update(d_now)

    d_acc.update(d_uniques)
    d_acc.update(d_begin)
    d_acc.update(d_end)
    d_acc.update(d_period)

    d_acc.update(d_serialized)

    return pd.Series(d_acc)

##################################################################################################
#
#       extract_column_info
#

def extract_column_info(g, md_0, d_0, is_create_stats):
    # Keep 'no info' above in sync

    l_begin = get_group_list(g, 'begin')
    l_end   = get_group_list(g, 'end')
    l_md    = get_group_list(g, 'MD')
    l_d     = get_group_list(g, 'D')

    assert len (l_begin) > 0

    # Data for onstarting period

    d_now = { 'md_now': md_0, 'd_now': d_0 }

    # Uniques 
    nUniqueMD = len (np.unique(np.array(l_md)))
    nUniqueD = len (np.unique(np.array(l_d)))
    
    d_uniques = { 'uniq_md' : nUniqueMD, 'uniq_d' : nUniqueD }


    d_begin = {}
    d_end = {}
    d_period = {}

    if is_create_stats:
        anPeriod = np.array(np.array(l_end) - np.array(l_begin))
        d_begin = get_prefixed_dict (get_stats_on_array(np.array(l_begin)), 'begin_')
        d_end   = get_prefixed_dict (get_stats_on_array(np.array(l_end)), 'end_')
        d_period = get_prefixed_dict (get_stats_on_array(anPeriod), 'span_')

    # Serialized data

    str_begin = ','.join(str(e) for e in l_begin)
    str_end = ','.join(str(e) for e in l_end)
    str_md = ','.join(str(e) for e in l_md)
    str_d = ','.join(l_d)

    d_serialized = {'begin':str_begin, 'end':str_end, 'md':str_md, 'd':str_d}

    d_acc = {}

    d_acc.update(d_now)

    d_acc.update(d_uniques)
    d_acc.update(d_begin)
    d_acc.update(d_end)
    d_acc.update(d_period)

    d_acc.update(d_serialized)

    return pd.Series(d_acc)


##################################################################################################
#
#       prepare_each_id
#

def prepare_each_id(x, expiry_days, is_create_stats):
    
    ID = x['id']

    begin0 = x['T0']
    y =  x['Y']

    m = (df.P == ID)

    #
    # Find one (or possibly more) intervals staring exactly at begin0
    #
    m_begin = (df.P == ID) & (df.begin == begin0)

    s = m_begin.value_counts()
    
    assert (len(s) > 1) and s[1] > 0

    # Read out data on first hit
    md_0 = df[m_begin][0:1].MD.values[0]
    d_0 = df[m_begin][0:1].D.values[0]

   
    # Now discard everything that is recorded later than or at begin time
    # Check for end time.

    q = df[m].copy()

    s = q.end - begin0

    q = q.assign(end = s)
    
    s = q.begin - begin0
   
    q = q.assign(begin = s)

    m = (q['begin'] < 0) & (q['begin'] > expiry_days)

    q = q[m]

    isEmpty = (len(q) == 0)

    if isEmpty:
        return no_column_info(is_create_stats)

    # Remove any future info. Todo: Investigate: Keep when available at begin_0
    m = q['end'] > 0

    q.loc[m, 'end'] = -1

    g = q.groupby(by = 'P')
    
    return extract_column_info(g, md_0, d_0, is_create_stats)

# main


df = train = pd.read_csv(DATA_DIR + 'noised_intervals.csv')


df.columns = ['drop', 'begin', 'end', 'P']
df = df.drop(['drop'], axis = 1)
df = df[['P', 'begin', 'end']]

md_series = np.random.choice([0,1,2,3,4,5,6], df.shape[0])
df = df.assign(MD=md_series)

d_series = np.random.choice(['D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6'], df.shape[0])
df = df.assign(D = d_series)
df['D'] = df['D'].astype('category')


df = df.sort_values(by = ['P', 'begin'])


Target_begin_span = (16800, 19000)
n_bandwidth = 30

df_t = extract_targets(df, Target_begin_span, n_bandwidth)

w0 = df_t.apply(prepare_each_id, axis = 1, args = (-365 * 12.0, True))


df_t = pd.concat([df_t, w0], axis = 1)

#print(df_t.shape)
#df_t[['id', 'T0', 'Y', 'd_now', 'md_now']]

####################################################################################################
#
# Serialization, load, init area.
#



df_t.to_pickle(DATA_DIR + "dt_t_90000.pkl")

# Verify save/load

df_t = pd.read_pickle(DATA_DIR + "dt_t_90000.pkl")


y = df_t['Y'].values
y = y.astype(np.float32)


df_t = df_t.drop(['uniq_d', 'uniq_md', 'md', 'd', 'id'], axis = 1)
df_t = df_t.drop(['begin', 'end'], axis = 1)
df_t = df_t.drop(['Y'], axis = 1)

df_t = df_t.drop(['d_now', 'md_now'], axis = 1)

#m = df_t.begin_count == 0
#df_t = df_t[~m]


X = np.array(df_t, dtype = np.float32)

##########

from sklearn.datasets import make_regression

X, y = make_regression(n_samples = 5000, n_features = 3000, n_informative = 2110, bias=0.1)


from sklearn.datasets import load_boston

X, y = load_boston(return_X_y=True)








y_p = train(X,y)

# 20JUL2018
#N = 30272, Folds = 7
#RMS  79.5274405725318 +/- 1.7821489966538362
#GINI 0.29705148282955574 +/- 0.008688106701486235 @positive > 60

#N = 705, Folds = 7
#RMS  73.69088975365541 +/- 7.427903157145201
#GINI 0.18106128267935048 +/- 0.17702768996029403 @positive > 60












