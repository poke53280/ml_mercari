


import numpy as np
import pandas as pd

from santander_3.txt_regr import txt_reg



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


def convert_all(l, wait_indexer, transaction_indexer):

    l_out = []

    it = iter(l)
    for x in it:
        zero_count = x
        value = next(it)

        if zero_count == 0:
            pass
        else:
            wait_binned = wait_indexer.transform(zero_count)
            l_out.append("WAIT_" + str(wait_binned))

        value_binned = transaction_indexer.transform(value)
        l_out.append("AMT_" + str(value_binned))
    return l_out        
            
"""c"""      



class BinCutter:

    _indexer = 0

    def __init__(self):
        pass


    def fit(self, df, count):

        X_train = np.array(df)
        X_train = X_train.flatten()

        m = (X_train != 0)

        X_train = X_train[m]
        X_train = np.log(X_train)

        s = pd.qcut(X_train, count, duplicates='drop')

        s.value_counts()

        self._indexer = s.categories


    def transform(self, x):
        return self._indexer.get_loc(np.log(x))

"""c"""


def get_txt_set(data, y_trainFull):

    transaction_bin = BinCutter ()
    transaction_bin.fit(data, 50)

    q = data.apply(my_func, raw = True, axis = 1)

    wf = q.apply(wait_find)

    wait_periods = []

    for x in q:
        wait_periods.extend(x)

    anwait_period = np.array(wait_periods)
    
    wait_bin = BinCutter()
    wait_bin.fit(anwait_period, 3)

    w2 = q.apply(convert_all, args = (wait_bin, transaction_bin))

    def create_string(l):
        return " ".join(l)
    """c"""

    s = w2.apply(create_string)

    data_txt = pd.DataFrame({'txt':s, 'target': y_trainFull}, index = s.index)

    return data_txt


def train_txt_set(data, y):
    m, s = txt_reg(data, y)
    return (m,s )

"""c"""


def my_main():

    DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
    DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
    DATA_DIR = DATA_DIR_PORTABLE

    train_CONST = pd.read_csv(DATA_DIR + 'train.csv')

    y_trainFull = np.log(train_CONST.target)
    train_id = train_CONST.ID


    # On original ordering
    train = train_CONST.drop(['target', 'ID'], axis = 1)
    X = np.array(train)
    train = pd.DataFrame(X)

    df = get_txt_set(train, y_trainFull)

    train_txt_set(df, y_trainFull)


    # So most non-null column to the left

    X = np.array(train_CONST.drop(['target', 'ID'], axis = 1))
    X_nz = np.count_nonzero(X, axis = 0)
    idx = (-X_nz).argsort()
    X = X[:, idx]
    train = pd.DataFrame(X)
    run(train, y_trainFull)

    # So most non null to the right

    X = np.array(train_CONST.drop(['target', 'ID'], axis = 1))
    X_nz = np.count_nonzero(X, axis = 0)
    idx = X_nz.argsort()
    X = X[:, idx]
    train = pd.DataFrame(X)
    run(train, y_trainFull)


