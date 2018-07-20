

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

import os
print (os.getcwd())

os.chdir('c:\\Users\\ander\\ds\\ml_mercari')


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.model_selection import KFold
from santander_3.lgbm_basic import LGBMTrainer_BASIC
from santander_3.RowStatCollector import RowStatCollector


import gc

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor




class Conf:
    _zero_threshold = 0
    _i_threshold = 0
    _cat_pct_zero = 0
    _cut_important_factors = 0
    _dim_reduction = 0

    def __init__(self):
        pass

    def configureA(self):
        self._cut_important_factors = 2500

        # Go much lower in column numbers (0.91 => 410)
        self._zero_threshold = [1.0, 0.9995, 0.9990, 0.997, 0.995, 0.992, 0.99, 0.98, 0.975, 0.97, 0.965, 0.96, 0.94, 0.93, 0.915, 0.91, 0.90, 0.87, 0.86, 0.85, 0.82, 0.8, 0.76, 0.7, 0.68, 0.65, 0.64]
        self._i_threshold = [2500, 1500, 1000, 200]
        self._cat_pct_zero = 0.995
        self._dim_reduction = 23
        
    def configureB(self):
        self.configureA()
        self._cut_important_factors = 3000


    def configureC(self):
        self.configureA()
        self._cat_pct_zero = 0.99

    def configureD(self):
        self.configureA()
        self._cat_pct_zero = 0.97

    def configureE(self):
        self.configureA()
        self._cat_pct_zero = 0.999
    

    def info(self):
        print(f"   zt {self._zero_threshold}")
        print(f"   it {self._i_threshold}")
        print(f"   cat_pct_zero {self._cat_pct_zero}")
        print(f"   cut_important_factors {self._cut_important_factors}")
        print(f"   dim_reduction {self._dim_reduction}")

"""c"""



 


#############################################################################
#
#         get_cols_low_zero
#
#   Returns cols with zero frequency lower than input perc_threshold
#

def get_cols_low_zero(df, perc_threshold):

    cols_to_keep = []
    nAll = df.shape[0]

    for c in df.columns:

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

#############################################################
#
#  from the neptune ml open source project on github.
#


def train_process(train, test, conf):

    test = test.applymap(np.float64)
    X_testFull = csr_matrix(test).astype(np.float64)
   
    non_zero_rows = X_testFull.getnnz(1) > 0
    assert( (non_zero_rows == True).sum() == X_testFull.shape[0])

    train = train.applymap(np.float64)
    X_trainFull = csr_matrix(train).astype(np.float64)
    
    non_zero_rows = X_trainFull.getnnz(1) > 0
    assert( (non_zero_rows == True).sum() == X_trainFull.shape[0])

    NUM_FOLDS = 5

    # Input to training:
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=22)

    lKF = list (enumerate(kf.split(X_trainFull)))

    lRMS = []

    y_oof = np.zeros(len (y_trainFull))
    prediction = np.zeros(X_testFull.shape[0])


    while len(lKF) > 0:
        iLoop, (train_index, test_index) = lKF.pop(0)

        print(f"--- Fold: {iLoop +1}/ {NUM_FOLDS} ---")

        X_train = X_trainFull[train_index]
        y_train = y_trainFull[train_index]
    
        X_valid = X_trainFull[test_index]
        y_valid = y_trainFull[test_index]

        d = DReduction(conf._dim_reduction)

        d.fit(X_train.todense())

        type (X_train)

        X2 = csr_matrix(d.transform(X_train.todense()))

        X_train = pd.concat([pd.DataFrame(X_train.todense()), pd.DataFrame(X2.todense())], axis = 1)

        V2 = csr_matrix(d.transform(X_valid.todense()))

        X_valid = pd.concat([pd.DataFrame(X_valid.todense()), pd.DataFrame(V2.todense())], axis = 1)
        
        l = LGBMTrainer_BASIC()

        l.train_with_validation(X_train, y_train, X_valid, y_valid)

        y_p = l.predict(X_valid)

        y_oof[test_index] = y_p

        rmsle_error = np.sqrt(metrics.mean_squared_error(y_p, y_valid))
        print(f"Rmsle: {rmsle_error}")

        lRMS.append(rmsle_error)

        # Predict on test set

        X3 = d.transform(X_testFull.todense())

        X_test = pd.concat([pd.DataFrame(X_testFull.todense()), X3], axis = 1)

        y_pred_this = l.predict(X_test)

        prediction = prediction + (1.0/NUM_FOLDS) * y_pred_this

    return (y_oof, prediction, lRMS)
    
"""c"""

def run9(train, test, conf, y_target):

    rc = RowStatCollector()

    # Collect row stats on removed features by zero.
    for zt in conf._zero_threshold:
        c = get_cols_low_zero(train, zt)
        print(f"Zero pct < {zt}: {len(c)} column(s)")

        rc.collect_stats(train, test, c)


    c_cut =  get_cols_low_zero(train, conf._cat_pct_zero )

    train = train[c_cut]
    test = test[c_cut]

    r = RegImportance()

    r.fit(train, y_target)

    for i_t in conf._i_threshold:
        rc.collect_stats(train, test, r.get_important(i_t))

    col_1 = r.get_important(conf._cut_important_factors)

    train = train[list(col_1)]
    test =  test[list(col_1)]

    rc.collect_stats(train, test, train.columns)

    train_and_stats = pd.concat([train, rc._train_acc], axis = 1)
    test_and_stats = pd.concat([test, rc._test_acc], axis = 1)
   

    y_off, prediction, lRMS = train_process(train_and_stats, test_and_stats, conf)

    anRMS = np.array(lRMS)
    

    prediction = np.clip(prediction, 0, 22)
    prediction = np.expm1(prediction)
    prediction = 1000.0 * prediction

    y_off = np.clip(y_off, 0, 22)
    y_off = np.expm1(y_off)
    y_off = 1000.0 * y_off

    return (y_off, prediction, anRMS)    

"""c"""

DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE

train = pd.read_csv(DATA_DIR + 'train.csv')

# train = train[:50]

#train['one'] = 1


train_id = train.ID

y_trainFull = train.target
y_trainFull = y_trainFull / 1000.0
y_trainFull = np.log1p(y_trainFull)


train = train.drop(['target', 'ID'], axis = 1)
    
for c in train.columns:
    train[c] = 0.001 * train[c]

test = pd.read_csv(DATA_DIR + 'test.csv')

# test = test[:90]

#test['one'] = 1

sub_id = test.ID

test = test.drop(['ID'], axis = 1)

for c in test.columns:
    test[c] = 0.001 * test[c]

r = RowStatCollector()

r.collect_stats(train, train, train.columns)

train_const = train.copy()
test_const = test.copy()


####################################################
#
# From disk

train_const = pd.read_csv(DATA_DIR + 'train_const_lot_of_stats.csv')


# ----------------- data loaded -----------------------------------


lcConf = []

c = Conf()
c.configureA()
lcConf.append(c)


lRMSLEMean = []
lRMSLEStd = []

for c in lcConf:

    print("Begin on:")
    c.info()
    train = train_const.copy()
    test = test_const.copy()


    yoff, prediction, anRMS = run9(train, test, c, y_trainFull)

    c.info()

    RMSLEmean = anRMS.mean()
    RMSLEstd  = anRMS.std()

    lRMSLEMean.append(RMSLEmean)
    lRMSLEStd.append(RMSLEstd)

    print(f"  ==> RMSLE = {RMSLEmean} +/- {RMSLEstd}")
    print(" ------------------------------------------------- ")
"""c"""



sub_lgb = pd.DataFrame()
sub_lgb["target"] = prediction
sub_lgb = sub_lgb.set_index(sub_id)
sub_lgb.to_csv(DATA_DIR + 'submission.csv', encoding='utf-8-sig')


#oof_res = pd.DataFrame()
#oof_res["target"] = y_oof
#oof_res = oof_res.set_index(train_id)
#oof_res.to_csv(DATA_DIR + 'submission_oof.csv', encoding='utf-8-sig')

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

# zt [1.0, 0.999, 0.995, 0.99, 0.975, 0.965, 0.94, 0.915]
#   it [2500, 1500, 1000, 200]
#   cat_pct_zero 0.995
#   cut_important_factors 1500
#   dim_reduction 23
#  ==> RMSLE = 1.3249261485147104 +/- 0.02284215140111256


# zt [1.0, 0.9995, 0.999, 0.997, 0.995, 0.992, 0.99, 0.98, 0.975, 0.97, 0.965, 0.96, 0.94, 0.93, 0.915, 0.91]
#   it [2500, 1500, 1000, 200]
#   cat_pct_zero 0.995
#   cut_important_factors 2500
#  dim_reduction 23
#  ==> RMSLE = 1.3234194129004229 +/- 0.022374313280645982 => LB 1.39 (same position  17)





