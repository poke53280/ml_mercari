
#
#
# Time series approach:
# https://www.kaggle.com/mortido/digging-into-the-data-time-series-theory
#
#
#
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

def create_row_stat_columns(df):

    # Replace 0 with NaN to ignore them.
    df_nan = df.replace(0, np.nan)

    data = pd.DataFrame()
    data['mean'] = df_nan.mean(axis=1)
    data['std'] = df_nan.std(axis=1, ddof = 0)
    data['min'] = df_nan.min(axis=1)
    data['max'] = df_nan.max(axis=1)
    data['number_of_different'] = df_nan.nunique(axis=1)               # Number of different values in a row.
    data['non_zero_count'] = df_nan.fillna(0).astype(bool).sum(axis=1) # Number of non zero values (e.g. transaction count)

    del df_nan
    gc.collect()

    return data


def preprocess(df):
    df = df.applymap(np.float64)
    X = csr_matrix(df).astype(np.float64)
    return X
"""c"""

DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE


train = pd.read_csv(DATA_DIR + 'train.csv')

y_trainFull = np.log1p(train.target)

train_id = train.ID

train = train.drop(['target', 'ID'], axis = 1)



test = pd.read_csv(DATA_DIR + 'test.csv')
sub_id = test.ID

test = test.drop(['ID'], axis = 1)

#
# Train and test loaded and removed ID, target columns.
#

# Add additional columns:

test = pd.concat([test, df_test6], axis =1)
train = pd.concat([train, df_train6], axis =1)



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
# RMSLE = 1.3798356544218948 +/- 0.02245439186675799 => LV 1.41
# 
#
# 
# Owl + kiselev 6 params, then lgbm basic
# RMSLE = 1.3509424818713218 +/- 0.025388774057149885



# Transform:

X = X[:, idx]

