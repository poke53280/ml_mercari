

#
#
#  Todo: Apply: Feature selection using Truncated SVD
#
#  https://www.kaggle.com/ishaan45/lgbm-with-tsvd
#
#


import lightgbm as lgb
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.preprocessing import MaxAbsScaler
from sklearn import metrics

from sklearn.model_selection import KFold
from sklearn.decomposition import TruncatedSVD



lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    "learning_rate": 0.01,
    "num_leaves": 180,
    "feature_fraction": 0.50,
    "bagging_fraction": 0.50,
    'bagging_freq': 4,
    "max_depth": -1,
    "reg_alpha": 0.3,
    "reg_lambda": 0.1,
    #"min_split_gain":0.2,
    "min_child_weight":10,
    'zero_as_missing':True
                }


def preprocess(df):
    df = df.applymap(np.float64)
    X = csr_matrix(df).astype(np.float64)
    return X
"""c"""


class LGBMTrainer:

    _scaler = MaxAbsScaler()    
    _clf = 0
    _svd = 0

    _n_components = 4100

    def fit (self, X):

        self._svd = TruncatedSVD(n_components = self._n_components, n_iter=20, random_state=42)

        print(f"Fitting SVD. {X.shape[1]} => {self._n_components}...")
        result = self._svd.fit(X)
        cumm_perc = np.sum(result.explained_variance_ratio_)

        print("   Cumulative explained variation for 4100 components:"+"{:.2%}".format(cumm_perc))

        self._scaler = MaxAbsScaler()
        self._scaler.fit(X)

    def transform(self, X):
        X = self._scaler.transform(X)
        X = self._svd.transform(X)
                
        return X

    def __init__(self):
        pass

    def train_with_validation(self, X_train, y_train, X_test, y_test):

        lgtrain = lgb.Dataset(X_train, y_train, feature_name = "auto")
        lgvalid = lgb.Dataset(X_test, y_test, feature_name = "auto")

        self._clf = lgb.train(lgbm_params, lgtrain, 100000, early_stopping_rounds=100, valid_sets= [lgvalid], verbose_eval=30)

    def predict(self, X_test):
        return self._clf.predict(X_test)

"""c"""

class LGBMTrainer2:

    _scaler = MaxAbsScaler()    
    _clf = 0
    _nNonZeroColumns = 0

    _n_components = 4100

    def fit (self, X):

        var = GetCSR_X_Variance(X)
        self._nNonZeroColumns = (var > 0)

        self._scaler = MaxAbsScaler()
        self._scaler.fit(X)

    def transform(self, X):
        X = self._scaler.transform(X)

        X = X[:, self._nNonZeroColumns]
                
        return X

    def __init__(self):
        pass

    def train_with_validation(self, X_train, y_train, X_test, y_test):

        lgtrain = lgb.Dataset(X_train, y_train, feature_name = "auto")
        lgvalid = lgb.Dataset(X_test, y_test, feature_name = "auto")

        self._clf = lgb.train(lgbm_params, lgtrain, 100000, early_stopping_rounds=100, valid_sets= [lgvalid], verbose_eval=30)

    def predict(self, X_test):
        return self._clf.predict(X_test)

"""c"""

###########################################################################################
#
#         GetCSR_X_Variance
#
#   Var(X) = Mean[X^2] - (Mean[X])^2

def GetCSR_X_Variance(X):

    squared_X = X.copy()
    squared_X.data **= 2

    m = X.mean(axis = 0)

    m2 = squared_X.mean(axis = 0)

    m = np.squeeze(np.asarray(m))

    m2 = np.squeeze(np.asarray(m2))

    m = (m * m)
    var = m2 - m

    return var

"""c"""


DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE


train = pd.read_csv(DATA_DIR + 'train.csv')

y_trainFull = np.log1p(train.target)

train = train.drop(['target', 'ID'], axis = 1)

test = pd.read_csv(DATA_DIR + 'test.csv')
sub_id = test.ID

test = test.drop(['ID'], axis = 1)

X_testFull = preprocess(test)

non_zero_rows = X_testFull.getnnz(1)>0
assert( (non_zero_rows == True).sum() == X_testFull.shape[0])


X_trainFull = preprocess(train)

non_zero_rows = X_trainFull.getnnz(1)>0
assert( (non_zero_rows == True).sum() == X_trainFull.shape[0])

NUM_FOLDS = 7

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

    l = LGBMTrainer2()

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

#
#
# 25.6.18: Split approach: FM_FTRL. LOCAL 2.1 => LB 2.55
# 26.6.18: CV 7. RMSLE = 1.3977451963812833 +/- 0.020619886844509137 => LB 1.44
#

# CV 7 - trunc SVD to 1300
# RMSLE = 1.5125528464808173 +/- 0.04438566722957734
#            trunc SVD to 4100
#RMSLE = 1.530016851378886 +/- 0.043423692890769694






