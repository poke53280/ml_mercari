

import pandas as pd
import numpy as np

from scipy.stats import skew, kurtosis


###############################################################################################
#
#             aggregate_row
#

def aggregate_row(row, prefix):
    
    nRows = row.count()

    # Bottleneck.
    # Todo: Use 'raw' row passing for better performance.

    m = (row != 0)

    nonzero_list = row[m]

    v = np.array(nonzero_list, dtype = np.float32)

    allzero = (len(v) == 0)

    
    aggs = {prefix + 'non_zero_mean':           0 if allzero else v.mean(),
            prefix + 'non_zero_std':            0 if allzero else v.std(),
            prefix + 'non_zero_max':            0 if allzero else v.max(),
            prefix + 'non_zero_min':            0 if allzero else v.min(),
            prefix + 'non_zero_sum':            0 if allzero else v.sum(),
            prefix + 'non_zero_skewness':       0 if allzero else skew(v),
            prefix + 'non_zero_kurtosis':       0 if allzero else kurtosis(v),
            prefix + 'non_zero_median':         0 if allzero else np.median(v),
            prefix + 'non_zero_q1':             0 if allzero else np.percentile(v, q=25),
            prefix + 'non_zero_q3':             0 if allzero else np.percentile(v, q=75),
            prefix + 'non_zero_log_mean':       0 if allzero else np.log1p(v).mean(),
            prefix + 'non_zero_log_std':        0 if allzero else np.log1p(v).std(),
            prefix + 'non_zero_log_max':        0 if allzero else np.log1p(v).max(),
            prefix + 'non_zero_log_min':        0 if allzero else np.log1p(v).min(),
            prefix + 'non_zero_log_sum':        0 if allzero else np.log1p(v).sum(),
            prefix + 'non_zero_log_skewness':   0 if allzero else skew(np.log1p(v)),
            prefix + 'non_zero_log_kurtosis':   0 if allzero else kurtosis(np.log1p(v)),
            prefix + 'non_zero_log_median':     0 if allzero else np.median(np.log1p(v)),
            prefix + 'non_zero_log_q1':         0 if allzero else np.percentile(np.log1p(v), q=25),
            prefix + 'non_zero_log_q3':         0 if allzero else np.percentile(np.log1p(v), q=75),
            prefix + 'non_zero_count':          0 if allzero else len(v),
            prefix + 'non_zero_fraction':       0 if allzero else len(v) / nRows

            }

    s = pd.Series(aggs)

    s.replace([np.inf, -np.inf], np.nan)
    s = s.fillna(0)

    return s

"""c"""

########################################################################################
#
#       create_row_stat_neptune
#

def create_row_stat_neptune(df, p):
    prefix = str(p)

    q = df.apply(aggregate_row, axis = 1, args = (prefix,))

    return q

"""c"""


########################################################################################
#
#       create_row_stat_columns
#

def create_row_stat_columns(df, p):

    prefix = str(p)

    # Replace 0 with NaN to ignore them.
    df_nan = df.replace(0, np.nan)

    data = pd.DataFrame()
    data[prefix + 'mean'] = df_nan.mean(axis=1)
    data[prefix + 'std'] = df_nan.std(axis=1, ddof = 0)
    data[prefix + 'min'] = df_nan.min(axis=1)
    data[prefix + 'max'] = df_nan.max(axis=1)
    data[prefix + 'number_of_different'] = df_nan.nunique(axis=1)               # Number of different values in a row.
    data[prefix + 'non_zero_count'] = df_nan.fillna(0).astype(bool).sum(axis=1) # Number of non zero values (e.g. transaction count)

    del df_nan
    gc.collect()


    m = data[prefix + 'non_zero_count'] == 0

    nNullsZero = data[m].isnull().sum().values.sum()

    if (nNullsZero > 0):
        #Rows with no entries will leave NANs
        # print(f"Found {nNullsZero} NAs in stat dataset, on empty rows. Setting to 0")
        data[m] = data[m].fillna(0)

    nNulls = data.isnull().sum().values.sum()
  
    if (nNulls > 0):
        print(f"Warning: Found {nNulls} NAs in stat dataset, setting to 0")
        data = data.fillna(0)


    return data

"""c"""


############################################################################################
#
#
#       RowStatCollector
#

class RowStatCollector:

    _train_acc = pd.DataFrame()
    _test_acc =  pd.DataFrame()

    _prefix = 0

    def __init__(self):
        pass

    def collect_stats(self, train, test, cols):
        self._train_acc = pd.concat([self._train_acc, create_row_stat_neptune(train[cols], self._prefix)], axis = 1)
        self._test_acc  = pd.concat([self._test_acc,  create_row_stat_neptune(test[cols], self._prefix)], axis = 1)
        self._prefix = self._prefix + 1

"""c"""


DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE

def local_run():


    train = pd.read_csv(DATA_DIR + 'train.csv')
    train = train[:100]

    y_target = train.target
    y_trainFull = np.log1p(train.target)
    train_id = train.ID
    train = train.drop(['target', 'ID'], axis = 1)

    test = pd.read_csv(DATA_DIR + 'test.csv')

    test = test[:100]

    sub_id = test.ID
    test = test.drop(['ID'], axis = 1)

    train_const = train.copy()
    test_const = test.copy()

"""c"""





