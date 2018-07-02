


import numpy as np
import pandas as pd

_interval_index = 0
_wait_indexer = 0


def my_func(x):

    l_out = []

    last_recorded = 0

    an = np.array(x)
    m = (an != 0)
    nz = np.where(m)

    for n in nz[0]:

        leading_zero = n - last_recorded
        
        l_out.append(leading_zero)
        l_out.append(an[n])

        last_recorded = n + 1

    return l_out
"""c"""


def wait_find(l):
    return l[::2]


def convert_all(l):

    l_out = []

    it = iter(l)
    for x in it:
        zero_count = x
        value = next(it)

        if zero_count == 0:
            pass
        else:
            wait_binned = _wait_indexer.get_loc(zero_count)
            l_out.append("WAIT_" + str(wait_binned))

        value_binned = _interval_index.get_loc(np.log(value))
        l_out.append("AMT_" + str(value_binned))
    return l_out        
            
"""c"""      

def categorize_transactions(train, count):

    X_train = np.array(train)
    X_train = X_train.flatten()

    m = (X_train != 0)

    X_train = X_train[m]

    X_train = np.log(X_train)

    s = pd.qcut(X_train, count, duplicates='drop')

    s.value_counts()

    interval_index = s.categories

    return interval_index

"""c"""


DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE

train = pd.read_csv(DATA_DIR + 'train.csv')

y_trainFull = np.log(train.target)
train_id = train.ID
train = train.drop(['target', 'ID'], axis = 1)

# Re-arrange so most non-null column to the left

X = np.array(train)
X_nz = np.count_nonzero(X, axis = 0)
idx = (-X_nz).argsort()
X = X[:, idx]
train = pd.DataFrame(X)

_interval_index = categorize_transactions (train, 35000)

q = train.apply(my_func, raw = True, axis = 1)


wf = q.apply(wait_find)

wait_periods = []

for x in q:
    wait_periods.extend(x)
"""c"""

anwait_period = np.array(wait_periods)
s = pd.qcut(anwait_period, 88, duplicates='drop')
_wait_indexer = s.categories

w2 = q.apply(convert_all)

def create_string(l):
    return " ".join(l)
"""c"""

s = w2.apply(create_string)

df = pd.DataFrame({'txt':s, 'target': y_trainFull}, index = s.index)

df.to_csv(DATA_DIR + 'txt_db_huge.csv')




