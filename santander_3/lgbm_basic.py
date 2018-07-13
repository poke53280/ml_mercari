
import lightgbm as lgb
from sklearn.preprocessing import MaxAbsScaler

import numpy as np

from santander_3.xxutils import GetUniqueColumns
from santander_3.xxutils import GetCSR_X_Variance

lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    "learning_rate": 0.01,
    "num_leaves": 1023,
    "feature_fraction": 0.70,
    "bagging_fraction": 0.10,
    'bagging_freq': 1,
    "max_bin"    : 255,
    
    "reg_lambda": 0.1,
    "lambda_l1": 1,
    "lambda_l2": 1,
    "min_data_in_leaf": 1500
    
    }


class LGBMTrainer_BASIC:

    _clf = 0

    def __init__(self):
        pass

    def train_with_validation(self, X_train, y_train, X_test, y_test):

        lgtrain = lgb.Dataset(X_train, y_train, feature_name = "auto")
        lgvalid = lgb.Dataset(X_test, y_test, feature_name = "auto")

        self._clf = lgb.train(lgbm_params, lgtrain, 100000, valid_sets= [lgvalid], verbose_eval=30)

    def predict(self, X_test):
        return self._clf.predict(X_test)

"""c"""




