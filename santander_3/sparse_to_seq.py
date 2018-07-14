
import numpy as np
import pandas as pd

from santander_3.txt_regr import txt_reg

########################################################################
#
#    my_func
#
#

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

########################################################################
#
#    wait_find
#
#

def wait_find(l):
    return l[::2]


########################################################################
#
#    convert_all
#
#

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



########################################################################
#
#    BinCutter
#
#

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


########################################################################
#
#    get_txt_set
#
#

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

########################################################################
#
#    train_txt_set
#
#
#

def train_txt_set(data, y):
    m, s = txt_reg(data, y)
    return (m,s )

"""c"""

###########################################################################################################
###########################################################################################################
###########################################################################################################


import numpy as np
import pandas as pd


####################################################################
#
#    Clip or pad to cut elements
#
#

def get_pop_cols(x):

    an = np.array(x)

    my_keys = np.nonzero(an)[0]
    my_values = an[my_keys]

    c = np.empty(2 * len(my_keys), dtype=my_keys.dtype)

    c[:] = np.nan

    c[0::2] = my_keys
    c[1::2] = my_values

    return list(c)

"""c"""


########################################################################
#
#    cut_or_pad
#
#
#

def cut_or_pad(x, cut, nan_value):
    x.extend(cut * [nan_value])
    return x[:cut]

########################################################################
#
#    my_main
#

def my_main():

    DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
    DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
    DATA_DIR = DATA_DIR_PORTABLE

    N_CUT = 5

    train = pd.read_csv(DATA_DIR + 'train.csv')
    train = train.drop(['target', 'ID'], axis = 1)

    test = pd.read_csv(DATA_DIR + 'test.csv')
    test = test.drop(['ID'], axis = 1)

    df = pd.concat([train, test], axis = 0)
  
    X = np.array(df)
   
    X_nz = np.count_nonzero(X, axis = 0)
    idx = (-X_nz).argsort()
    X = X[:, idx]
    df = pd.DataFrame(X)

    q = df.apply(get_pop_cols, raw = True, axis = 1)

    q = q[:9]

    nList = q.apply(lambda x: len(x)/2)

    anLength = np.array(nList)

    print(f"Number of column entries: {anLength.mean()} +/- { anLength.std() }")

    test_threshold = nList.apply(lambda x: x >= N_CUT)

    aTrue = np.array(test_threshold)
    nFilled = (aTrue == True).sum()
    nPadded = len(q) - nFilled

    rPct = 100.0 * nPadded / len(q)

    print(f"Column count {N_CUT}: NA padding at {rPct}% of rows")

    q_idx = q.apply(lambda x: x[0::2])
    q_val = q.apply(lambda x: x[1::2])


    q_idx_cut = q_idx.apply(cut_or_pad, args = (N_CUT, -1))
    q_val_cut = q_val.apply(cut_or_pad, args = (N_CUT, 0))


    pd_IDX = pd.DataFrame.from_items(zip(q_idx_cut.index, q_idx_cut.values)).transpose()

    cols = pd_IDX.columns

    for c in cols:
        pd_IDX[c] += 1

    """c"""

    new_cols = []

    for i in range(N_CUT):
        new_cols.append("col_" + str(i))

    pd_IDX.columns = new_cols







    pd_VAL = pd.DataFrame.from_items(zip(q_val_cut.index, q_val_cut.values)).transpose()

    new_cols = []

    for i in range(N_CUT):
        new_cols.append("val_" + str(i))

    """c"""
    pd_VAL.columns = new_cols

    df = pd.concat([pd_IDX, pd_VAL], axis = 1)

   


    df.shape

    
from sklearn.preprocessing import OneHotEncoder


from sklearn.preprocessing import LabelBinarizer


import numpy as np
import pandas as pd

import category_encoders as ce




#
#
#
# Clip so that few zeros.
#
#
#
# 'col0', 'col1', 'col2', 'col3', 'col4', val0, val1, val2, val3, val4
#
#
# DAE on train + test.
# Categorical + continuous.
#
#





df = pd.DataFrame({

        'name': ['The Dude', 'Walter', 'Donny', 'The Stranger', 'Brandt', 'Bunny'],

        'haircolor': ['brown', 'brown', 'brown', 'silver', 'blonde', 'blonde'],

        'gender': ['male', 'male', 'male', 'male', 'male', 'female'],

        'drink': ['caucasian', 'beer', 'beer', 'sasparilla', 'unknown', 'unknown'],

        'age': [48, 49, 45, 63, 40, 23] 

    },

    columns=['name', 'haircolor', 'gender', 'drink', 'age']

)

encoder = ce.BinaryEncoder(cols=['haircolor'])

df_binary = encoder.fit_transform(df)

df_binary

df
