

from catboost import CatBoostRegressor
from sklearn.preprocessing import MaxAbsScaler
import numpy as np

from santander_3.xxutils import GetUniqueColumns
from santander_3.xxutils import GetCSR_X_Variance

class CatBoost_BASIC:

    _scaler = MaxAbsScaler()    
    _cb_model = 0
    _idxAccepted = 0

    def fit (self, X):

        idx = GetUniqueColumns(X)
        v = GetCSR_X_Variance(X)
        m = v > 0
        an = np.where(m)
        an = an[0]
        # Keep where in both lists
        s = set(idx)
        s = s.intersection(an)
        self._idxAccepted = np.array(list (s))

        self._scaler = MaxAbsScaler()
        self._scaler.fit(X)

    def transform(self, X):
        X = self._scaler.transform(X)
        X = X[:, self._idxAccepted]
                
        return X

    def __init__(self):
        pass

    def train_with_validation(self, X_train, y_train, X_test, y_test):

        c = CatBoostRegressor(iterations=50, learning_rate=0.05, depth=10, eval_metric='RMSE', random_seed = 42, bagging_temperature = 0.2, od_type='Iter', metric_period = 50, od_wait=20)

        c.fit(np.array(X_train.todense()), np.array(y_train), eval_set=(np.array(X_test.todense()), np.array(y_test)), use_best_model=True, verbose=True) 
        self._cb_model = c

    def predict(self, X_test):
        return self._cb_model.predict(np.array(X_test.todense()))

"""c"""

