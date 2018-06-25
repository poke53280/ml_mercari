
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from scipy.sparse import csr_matrix
from sklearn.preprocessing import MaxAbsScaler
from sklearn import metrics

from sklearn.model_selection import KFold



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

    _non_zero_columns = {}
    _scaler = MaxAbsScaler()    

    _clf = 0


    def fit (self, X):
        self._non_zero_columns = X_train.getnnz(0) > 0
        self._scaler = MaxAbsScaler()
        self._scaler.fit(X_train)

    def transform(self, X):
        X = self._scaler.transform(X)
        X = X[:,self._non_zero_columns]
        return X

    def __init__(self):
        pass

    def train_with_validation(self, X_train, y_train, X_test, y_test):

        lgtrain = lgb.Dataset(X_train, y_train, feature_name = "auto")
        lgvalid = lgb.Dataset(X_test, y_test, feature_name = "auto")

        self._clf = lgb.train(lgbm_params, lgtrain, 100000, early_stopping_rounds=100, valid_sets= [lgvalid], verbose_eval=30)


    def trainXXX(self, X_train, y_train):
        lgtrain = lgb.Dataset(X_train, y_train, feature_name = "auto")
        self._clf = lgb.train(lgbm_params, lgtrain, num_boost_round = 20)


    def predict(self, X_test):
        return self._clf.predict(X_test)

"""c"""


DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE


train = pd.read_csv(DATA_DIR + 'train.csv')

y_trainFull = np.log1p(train.target)

train = train.drop(['target', 'ID'], axis = 1)

X_trainFull = preprocess(train)

non_zero_rows = X_trainFull.getnnz(1)>0
assert( (non_zero_rows == True).sum() == X_trainFull.shape[0])

# Input to training:
kf = KFold(n_splits=5, shuffle=True, random_state=114514)


for iLoop, (train_index, test_index) in enumerate(kf.split(X_trainFull)):

    print(f"--- Fold: {iLoop} ---")

    #train_index = [1,2,4, 40, 41, 44, 49]
    #test_index = [9,10, 11, 12, 100]    


    X_train = X_trainFull[train_index]
    y_train = y_trainFull[train_index]
    
    X_valid = X_trainFull[test_index]
    y_valid = y_trainFull[test_index]

    l = LGBMTrainer()

    l.fit(X_train)

    X_train = l.transform(X_train)
    X_valid = l.transform(X_valid)

    l.trainXXX(X_train, y_train)

    l.train_with_validation(X_train, y_train, X_valid, y_valid)

    y_p = l.predict(X_valid)

    rmsle_error = np.sqrt(metrics.mean_squared_error(y_p, y_valid))
    print(f"Rmsle: {rmsle_error2}")
    
"""c"""


# Train
model = FTRL(verbose=1)
   
model.fit(dev_X, dev_y)

# Evaluate
y_val_predicted = model.predict(val_X)
y_val_predicted = np.clip(y_val_predicted, 0, 22)
print('RMSLE:', np.sqrt(metrics.mean_squared_error(y_val_predicted, val_y)))


test = pd.read_csv(DATA_DIR + 'test.csv')

sub_id = test.ID

test = test.drop(['ID'], axis = 1)

X_test = preprocess(test)

non_zero_rows = X_test.getnnz(1)>0
assert( (non_zero_rows == True).sum() == X_test.shape[0])


X_test = transform(X_test, _scaler, _non_zero_columns)


prediction = model.predict(X_test)
prediction = np.clip(prediction, 0, 22)

prediction = np.expm1(prediction)

sub_lgb = pd.DataFrame()
sub_lgb["target"] = prediction
sub_lgb = sub_lgb.set_index(sub_id)


sub_lgb.to_csv(DATA_DIR + 'submission.csv', encoding='utf-8-sig')


# Split: 2.1 => LB 2.55



pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, X_test)


u = run_fm(dev_X, dev_y, val_X, val_y, X_test)


X = [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 5.0]]

X = csr_matrix(X)

X = X.astype(np.float64)

y = [1.0,2.0,3.0,4.0]

m = FM_FTRL()

m.fit(X, y)

y_p = m.predict([4.0,4.0,4.0,5.0])

print(f"y_p = {y_p}")





