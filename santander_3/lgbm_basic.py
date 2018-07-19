
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
    "num_leaves": 31,
    "feature_fraction": 0.70,
    "bagging_fraction": 0.10,
    'bagging_freq': 1,

    }


class LGBMTrainer_BASIC:

    _clf = 0

    def __init__(self):
        pass

    def train_with_validation(self, X_train, y_train, X_test, y_test):

        lgtrain = lgb.Dataset(X_train, y_train, feature_name = "auto")
        lgvalid = lgb.Dataset(X_test, y_test, feature_name = "auto")

        self._clf = lgb.train(lgbm_params, lgtrain, 1000, valid_sets= [lgvalid], verbose_eval=200)

    def predict(self, X_test):
        return self._clf.predict(X_test)

"""c"""




