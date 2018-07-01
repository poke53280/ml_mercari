
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


class LGBMTrainer_BASIC:

    _clf = 0

    def __init__(self):
        pass

    def train_with_validation(self, X_train, y_train, X_test, y_test):

        lgtrain = lgb.Dataset(X_train, y_train, feature_name = "auto")
        lgvalid = lgb.Dataset(X_test, y_test, feature_name = "auto")

        self._clf = lgb.train(lgbm_params, lgtrain, 100000, early_stopping_rounds=100, valid_sets= [lgvalid], verbose_eval=30)

    def predict(self, X_test):
        return self._clf.predict(X_test)

"""c"""




