

import pandas as pd
import numpy as np

from general.KDE_study import group_sorted_unique_integers

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
    l = l[0][1]
    l = l.values

    return l



##################################################################################
#
#       add_to_set
#
#

def add_to_set(x, my_set):
    begin = x['begin']
    end   = x['end']
    my_set.update ( range(begin, end + 1))


##################################################################################
#
#       get_day_set
#
#

def get_day_set(df, m):

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

        for x in l:
            isTargetCandidate = x[0] >= Target_begin_span[0] and x[0] <= Target_begin_span[1]

            if isTargetCandidate:
                target = {}
                target['id'] = i
            
                L = 1 + x[1] - x[0]
            
                target['L'] = L
                target['Begin'] = x[0]

                targets.append(target)
                break

    rPct = 100.0 * len(targets) / nRows
    print(f"{rPct:.1f}% IDs in target zone")


    df_t = pd.DataFrame(targets)

    df_t.columns = ['T0', 'Y', 'id']

    df_t = df_t[['id', 'T0', 'Y']]

    return df_t


##################################################################################################
#
#       prepare_each_id
#

def prepare_each_id(x, nCut):
    ID = x['id']

    t0 = x['T0']
    y =  x['Y']


    m = (df.P == ID)
    
    record_ahead_time = 7

    # Discard everything that is recorded later than record_ahead_time beyond begin time
    # Check for end time.

    q = df[m].copy()

    s = q.end - t0

    q = q.assign(end = s)
    
    s = q.begin - t0
   
    q = q.assign(begin = s)

    m = q['begin'] <= (record_ahead_time)

    q = q[m]

    m = q['end'] >= (record_ahead_time)

    q.loc[m, 'end'] = record_ahead_time

    g = q.groupby(by = 'P')

    l_begin = get_group_list(g, 'begin')
    l_end   = get_group_list(g, 'end')
    l_md    = get_group_list(g, 'MD')
    l_d     = get_group_list(g, 'D')

    if nCut > 0:
        l_begin = l_begin[:nCut]
        l_end = l_end[:nCut]
        l_md = l_md[:nCut]
        l_d = l_d[:nCut]

    str_begin = ','.join(str(e) for e in l_begin)
    str_end = ','.join(str(e) for e in l_end)
    str_md = ','.join(str(e) for e in l_md)
    str_d = ','.join(l_d)

    return pd.Series({'begin':str_begin, 'end':str_end, 'md':str_md, 'd':str_d})


# main


df = train = pd.read_csv(DATA_DIR + 'noised_intervals.csv')


df.columns = ['drop', 'begin', 'end', 'P']
df = df.drop(['drop'], axis = 1)
df = df[['P', 'begin', 'end']]

df = df.assign(MD=14)
df = df.assign(D = 'P75')


df = df.sort_values(by = ['P', 'begin'])


Target_begin_span = (16800, 17000)
n_bandwidth = 30

df_t = extract_targets(df, Target_begin_span, n_bandwidth)


w0 = df_t.apply(prepare_each_id, axis = 1, args = (0,))


df_t = df_t.assign(begin = w0['begin'])
df_t = df_t.assign(end = w0['end'])
df_t = df_t.assign(md = w0['md'])
df_t = df_t.assign(d = w0['d'])

pd.set_option('max_colwidth', 400)

df_t = df_t.drop(['id'], axis = 1)

df_t[['begin', 'end', 'Y']]