

import pandas as pd
import numpy as np

from general.KDE_study import group_sorted_unique_integers

DATA_DIR_PORTABLE = "C:\\p_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE

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





df = train = pd.read_csv(DATA_DIR + 'noised_intervals.csv')


df.columns = ['drop', 'begin', 'end', 'P']
df = df.drop(['drop'], axis = 1)
df = df[['P', 'begin', 'end']]
df = df.sort_values(by = ['P', 'begin'])




m = (df.P == 1)

an = get_day_set(df, m)


n_bandwidth = 3

l = group_sorted_unique_integers(an, n_bandwidth)


l


df[m]