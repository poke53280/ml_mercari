

seq_data = []

def func(x):
    st = ""
    count = 0
    for c in x:
        if c != 0:
            st = st + " " + str(c)
            count = count + 1

    seq_data.append(str(count) + ":" + st)

train.apply(func, axis = 1)

def count_on(X):
    anCount = np.zeros(X.shape[0])
    for idx, row in enumerate(X):
        mr = (row != 0)
        anCount[idx] = mr.sum()
    return anCount
"""c"""

def count_users_on_date(Xbool, col):
    Xcol = Xbool[:, col]

    m = (Xcol == True)
    m = np.ravel(m)

    Xtemp = Xbool[m]

    nUsers = Xtemp.shape[0]

    print(f"{nUsers} user(s) have an entry on col = {col}")

    return nUsers
"""c"""


import pandas as pd
import numpy as np

DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE

train = pd.read_csv(DATA_DIR + 'train.csv')


y_target = train.target
y_trainFull = np.log1p(train.target)
train_id = train.ID
train = train.drop(['target', 'ID'], axis = 1)

X = np.matrix (train)

def get_frequent_columns_sorted(X):

    # Mean values for each column
    m = np.mean(X, axis = 0)

    m = m.T
    m = m[:, 0]

    m = np.array(m)
    m = m[:, 0]
    mean_value_for_col = m

    Xbool = X.astype(np.bool)


    r = range (Xbool.shape[1])

    nfreq = []

    for col in r:
        n = count_users_on_date(Xbool, col)
        nfreq.append(n)
    """c"""

    an = np.array(nfreq)


    a_value = np.sort(an)
    a_idx = np.argsort(an)

    a_idx = np.flip(a_idx, axis = 0)


    X_t = X0[:, a_idx]


    Xbool = X_t.astype(np.bool)

    count_users_on_date(Xbool, 2422)





for i, x in enumerate (a_idx):
    m = mean_value_for_col[x]
    print(f"Log mean for sorted col {i} is {np.log(m)}")
   


count_on(X)


c = train.columns.values




#
# 26, 61, 32
#




############################################################
#
#    score_both_on
#   

def score_both_on(X, iTestRow):
    test_row = X[iTestRow]     
    m_test = (test_row != 0)

    lScore = []

    for idx, row in enumerate(X):
        if idx == iTestRow:
            lScore.append(-1)
        else:
            mr = (row != 0)

            nBothOne = (m_test & mr).sum()
            lScore.append(nBothOne)

    return lScore

"""c"""

score_both_on(X, 2)

#
# SIGNAL: These have several in common.
#



