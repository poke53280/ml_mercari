
from wordbatch.models import FM_FTRL
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from scipy.sparse import csr_matrix
import lightgbm as lgb
from sklearn.preprocessing import MaxAbsScaler
from sklearn import metrics

def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "boosting_type":'gbdt',
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 40,
        "learning_rate" : 0.005,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "random_seed": 42
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000, 
                      valid_sets=[lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=50, 
                      evals_result=evals_result)
    
    pred_test_y = np.expm1(model.predict(test_X, num_iteration=model.best_iteration))
    return pred_test_y, model, evals_result
"""c"""


def preprocess(df):
    df = df.applymap(np.float64)
    X = csr_matrix(df).astype(np.float64)
    return X
"""c"""


def transform(X, scaler, nz_columns):
    X = _scaler.transform(X)
    X = X[:,_non_zero_columns]
    return X

"""c"""


DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE


train = pd.read_csv(DATA_DIR + 'train.csv')

y_train = np.log1p(train.target)

train = train.drop(['target', 'ID'], axis = 1)

X_train = preprocess(train)

non_zero_rows = X_train.getnnz(1)>0
assert( (non_zero_rows == True).sum() == X_train.shape[0])


dev_X, val_X, dev_y, val_y = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)


# Fit
_non_zero_columns = dev_X.getnnz(0) > 0
_scaler = MaxAbsScaler()
_scaler.fit(dev_X)


# Transform
dev_X = transform(dev_X, _scaler, _non_zero_columns)
val_X = transform(val_X, _scaler, _non_zero_columns)

# Train
model = FM_FTRL(alpha=0.1, beta=0.09, L1=2.6, L2=2.4, D=val_X.shape[1], alpha_fm=0.05, L2_fm=0.01, init_fm=0.01, D_fm=64, weight_fm=1.0, e_noise=0.0, iters=4, verbose=1)
   
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





