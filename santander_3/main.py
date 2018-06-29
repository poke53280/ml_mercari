
#
#
# Time series approach:
# https://www.kaggle.com/mortido/digging-into-the-data-time-series-theory
#
#
#
#
#
#  Geometric mean of each row:
#  https://www.kaggle.com/ianchute/geometric-mean-of-each-row-lb-1-55
#

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.model_selection import KFold
from santander_3.lgbm_basic import LGBMTrainer_BASIC
from santander_3.lgbm_svd import LGBMTrainer_TruncatedSVD

from santander_3.catboost import CatBoost_BASIC

import gc

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FastICA
from sklearn.random_projection import SparseRandomProjection
from sklearn.random_projection import GaussianRandomProjection



class DReduction:

    _N_COMP = 0            ### Number of decomposition components ###

    _pca    = 0
    _tsvd   = 0
    _ica    = 0
    _grp    = 0
    _srp    = 0

    def __init__(self, nComp):
        self._N_COMP = nComp
        self._pca = PCA(n_components=self._N_COMP, random_state=17)
        self._tsvd = TruncatedSVD(n_components=self._N_COMP, random_state=17)
        self._ica = FastICA(n_components=self._N_COMP, random_state=17)
        self._grp = GaussianRandomProjection(n_components=self._N_COMP, eps=0.1, random_state=17)
        self._srp = SparseRandomProjection(n_components=self._N_COMP, dense_output=True, random_state=17)


    def fit(self, X):
        self._pca.fit(X)
        self._tsvd.fit(X)
        self._ica.fit(X)
        self._grp.fit(X)
        self._srp.fit(X)


    def transform(self, X):
        res_pca  = self._pca.transform(X)
        res_tsvd = self._tsvd.transform(X)
        res_ica  = self._ica.transform(X)
        res_grp  = self._grp.transform(X)
        res_srp  = self._srp.transform(X)


        df = pd.DataFrame()

        for i in range(1, self._N_COMP + 1):
            df['pca_' + str(i)] = res_pca[:, i - 1]
            df['tsvd_' + str(i)] = res_tsvd[:, i - 1]
            df['ica_' + str(i)] = res_ica[:, i - 1]
            df['grp_' + str(i)] = res_grp[:, i - 1]
            df['srp_' + str(i)] = res_srp[:, i - 1]

        return df
"""c"""

 


######################################################################################
#
#  get_important_columns
#
#
# Based on: https://www.kaggle.com/the1owl/love-is-the-answer
#
# by    https://www.kaggle.com/the1owl
#
#  dont_consider: List of named columns not to process and consider
#
#  Returns names of the nCut most important columns.
#


class RegImportance:

    _df = 0

    def __init__(self):
        pass         
    
    def rmsle(self, y, pred):
        return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(pred), 2)))

    def fit(self, df, y_true):
        scl = StandardScaler()

        col = [c for c in df.columns]

        x1, x2, y1, y2 = train_test_split(df[col], y_true, test_size=0.20, random_state=5)

        model = RandomForestRegressor(n_jobs = -1, random_state = 7)

        model.fit(scl.fit_transform(x1), y1)

        print(f"RMSLE Random Forest Regressor: {self.rmsle(y2, model.predict(scl.transform(x2)))}")

        self._df = pd.DataFrame({'importance': model.feature_importances_, 'feature': col}).sort_values(by=['importance'], ascending=[False])


    def get_important(self, nCut):
        df = self._df[:nCut]    
        cols = df['feature'].values
        return cols

"""c"""

#############################################################################
#
#         get_cols_low_zero
#
#   Returns cols with zero frequency lower than input perc_threshold
#

def get_cols_low_zero(df, exclude_cols, perc_threshold):

    cols_to_keep = []
    nAll = df.shape[0]

    for c in df.columns:
        if c in exclude_cols:
            continue

        q = df[c]
        a = q.value_counts()

        nZero = a[0] if 0 in a else 0

        isInclude = (nZero < nAll * perc_threshold)

        if isInclude:
            cols_to_keep.append(c)

    return cols_to_keep

"""c"""



#Useful for nan handling
#n = np.array(l)
#
#n = n.squeeze()
#
#n.mean()
#n.var()
#n.min()
#
#a = np.ma.array(n, mask=np.isnan(n)) # Use a mask to mark the NaNs
#
#nxa = nx[~a.mask]

########################################################################################
#
#
# create_row_stat_columns on non NANs for input df.
#
# From: https://www.kaggle.com/mortido/digging-into-the-data-time-series-theory
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


def preprocess(df):
    df = df.applymap(np.float64)
    X = csr_matrix(df).astype(np.float64)
    return X
"""c"""


class RowStatCollector:

    _train_acc = pd.DataFrame()
    _test_acc =  pd.DataFrame()

    _prefix = 0

    def __init__(self):
        pass

    def collect_stats(self, train, test, cols):
        self._train_acc = pd.concat([self._train_acc, create_row_stat_columns(train[cols], self._prefix)], axis = 1)
        self._test_acc  = pd.concat([self._test_acc,  create_row_stat_columns(test[cols], self._prefix)], axis = 1)
        self._prefix = self._prefix + 1


"""c"""

# Data per row investigation

#seq_data = []

def func(x):
    st = ""
    count = 0
    for c in x:
        if c != 0:
            st = st + " " + str(c)
            count = count + 1

    seq_data.append(str(count) + ":" + st)

#train.apply(func, axis = 1)



DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE


train = pd.read_csv(DATA_DIR + 'train.csv')


y_target = train.target
y_trainFull = np.log1p(train.target)
train_id = train.ID
train = train.drop(['target', 'ID'], axis = 1)


test = pd.read_csv(DATA_DIR + 'test.csv')
sub_id = test.ID
test = test.drop(['ID'], axis = 1)

train_const = train.copy()
test_const = test.copy()


# ----------------- data loaded -----------------------------------

train = train_const.copy()
test = test_const.copy()


rc = RowStatCollector()

rc.collect_stats(train, test, train.columns)


zero_threshold = [0.999, 0.995, 0.98, 0.97, 0.95, 0.94, 0.93, 0.92, 0.91 ]

#zero_threshold = [0.999, 0.998, 0.997, 0.996, 0.995, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.8, 0.7, 0.5, 0.3 ]

for zt in zero_threshold:
    rc.collect_stats(train, test, get_cols_low_zero(train, [], zt ))


c_098 =  get_cols_low_zero(train, [], 0.98 )

train = train[c_098]
test = test[c_098]



r = RegImportance()

r.fit(train, y_target)

i_threshold = [3500, 3000, 2500, 2000, 1500, 1000, 400, 300, 200, 100, 50]

for i_t in i_threshold:
    rc.collect_stats(train, test, r.get_important(i_t))

col_1 = r.get_important(500)

train = train[list(col_1)]
test =  test[list(col_1)]

rc.collect_stats(train, test, train.columns)




#d = DReduction(20)
#
#d.fit(train)
#
#train_dim_info_0 = d.transform(train)
#test_dim_info_0 = d.transform(test)


train_and_stats = pd.concat([train, rc._train_acc], axis = 1)
test_and_stats = pd.concat([test, rc._test_acc], axis = 1)


d2 = DReduction(22)

d2.fit(train_and_stats)

train_dim_info_2 = d2.transform(train_and_stats)
test_dim_info_2 = d2.transform(test_and_stats)



train = pd.concat([train, rc._train_acc, train_dim_info_2], axis = 1)
test = pd.concat([test, rc._test_acc, test_dim_info_2], axis = 1)


X_testFull = preprocess(test)

non_zero_rows = X_testFull.getnnz(1) > 0

assert( (non_zero_rows == True).sum() == X_testFull.shape[0])


X_trainFull = preprocess(train)

non_zero_rows = X_trainFull.getnnz(1) > 0
assert( (non_zero_rows == True).sum() == X_trainFull.shape[0])

NUM_FOLDS = 5

# Input to training:
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=22)

lKF = list (enumerate(kf.split(X_trainFull)))

lRMS = []

#
# 5 fold:
#
# RMSLE = 1.3923651817971539 +/- 0.0355577782181301
# RMSLE = 1.3942490903567695 +/- 0.02128855447755487
# RMSLE = 1.3987515329442886 +/- 0.014860682891675773
#
# 20 fold:
#
# 1.3862360884301765 +/- 0.04775749043367667
#
# 50 fold:
#
# RMSLE = 1.3741171143498698 +/- 0.09481640650704826
#


y_oof = np.zeros(len (y_trainFull))
prediction = np.zeros(X_testFull.shape[0])


while len(lKF) > 0:
    iLoop, (train_index, test_index) = lKF.pop(0)

    print(f"--- Fold: {iLoop +1}/ {NUM_FOLDS} ---")

    X_train = X_trainFull[train_index]
    y_train = y_trainFull[train_index]
    
    X_valid = X_trainFull[test_index]
    y_valid = y_trainFull[test_index]

    l = LGBMTrainer_BASIC()

    l.fit(X_train)

    X_train = l.transform(X_train)
    X_valid = l.transform(X_valid)

    l.train_with_validation(X_train, y_train, X_valid, y_valid)

    y_p = l.predict(X_valid)

    y_oof[test_index] = y_p

    rmsle_error = np.sqrt(metrics.mean_squared_error(y_p, y_valid))
    print(f"Rmsle: {rmsle_error}")

    lRMS.append(rmsle_error)

    # Predict on test set
    X_test = l.transform(X_testFull)

    y_pred_this = l.predict(X_test)

    prediction = prediction + (1.0/NUM_FOLDS) * y_pred_this
    
"""c"""

anRMS = np.array(lRMS)

print(f"RMSLE = {anRMS.mean()} +/- {anRMS.std()}")

prediction = np.clip(prediction, 0, 22)
prediction = np.expm1(prediction)

sub_lgb = pd.DataFrame()
sub_lgb["target"] = prediction
sub_lgb = sub_lgb.set_index(sub_id)
sub_lgb.to_csv(DATA_DIR + 'submission.csv', encoding='utf-8-sig')

y_oof = np.clip(y_off, 0, 22)
y_oof = np.expm1(y_oof)

oof_res = pd.DataFrame()
oof_res["target"] = y_oof
oof_res = oof_res.set_index(train_id)
oof_res.to_csv(DATA_DIR + 'submission_oof.csv', encoding='utf-8-sig')

#
# 25.6.18: Split approach: FM_FTRL. LOCAL 2.1 => LB 2.55
# 26.6.18: CV 7. RMSLE = 1.3977451963812833 +/- 0.020619886844509137 => LB 1.44
#

#
# CV 7 - truncated SVD to 1300
# RMSLE = 1.5125528464808173 +/- 0.04438566722957734
#            trunc SVD to 4100
# RMSLE = 1.530016851378886 +/- 0.043423692890769694
#
#
# 27.6.18: CV 7. W. identical column removal.
# ==> Identical score to 26.6.18.
#

# 27.6.18: 1ows preprocess to 500 features, then lgbm basic
# RMSLE = 1.3798356544218948 +/- 0.02245439186675799 => LB 1.41
# 
#
# 
# Owl + kiselev 6 params, then lgbm basic
# RMSLE = 1.3509424818713218 +/- 0.025388774057149885 => LB 1.39
#
# Reproduction:
# RMSLE = 1.3513581053302484 +/- 0.029107351301479786

# 29.6.18
# 4 stats. data = most important + dim red + 4 stats
# RMSLE = 1.3466391208960116 +/- 0.026560647793176247 => LB 1.39 (better)
#
# wo data
# RMSLE = 1.3513601433146296 +/- 0.030562705043870986
#
# Cut to 0.98 on top
#
# RMSLE = 1.3411894612788142 +/- 0.029811893340113513 => LB 1.39 (same pos)
#
#
# Added one more stat
#
# RMSLE = 1.3345900143879534 +/- 0.028837981089004754 => LB 1.39 (same pos)
#
#
#
# Lots of stats. Dim red also on stat
#
#
# RMSLE = 1.3317368027725465 +/- 0.026210934894761846
#

#
# + lots of dim reds
# RMSLE = 1.3339867861359997 +/- 0.027655917990326056
#
# One dim red, 20, on data + stats. d = data + stats + dim red
# 
# RMSLE = 1.3285830630203477 +/- 0.022892193951572624


# Retake.
# zero_threshold = [0.999, 0.995, 0.98, 0.97, 0.95, 0.94, 0.93, 0.92, 0.91 ]
# RMSLE = 1.3276220900730766 +/- 0.021510087721721926
#
# zero_threshold = [0.999, 0.998, 0.997, 0.996, 0.995, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.8, 0.7, 0.5, 0.3 ]
#  
# RMSLE = 1.3276220900730766 +/- 0.021510087721721926
#
# zero_threshold = [0.999, 0.995, 0.98, 0.97, 0.95, 0.94, 0.93, 0.92, 0.91 ]
# DIM 40
# RMSLE = 1.3329107467634214 +/- 0.025289962320957902
#
# zero_threshold = [0.999, 0.995, 0.98, 0.97, 0.95, 0.94, 0.93, 0.92, 0.91 ]
# DIM 30
#
# RMSLE = 1.3329107467634214 +/- 0.025289962320957902
##
# DIM 18
# RMSLE = 1.3314209045892127 +/- 0.02724667939935871
#
#
# DIM 22
#
# RMSLE = 1.3293895815253935 +/- 0.025997633395490664

