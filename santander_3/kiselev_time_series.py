
# From: https://www.kaggle.com/mortido/digging-into-the-data-time-series-theory
#
# by: https://www.kaggle.com/mortido
#
#

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_predict

import lightgbm as lgb




DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE

train_df = pd.read_csv(DATA_DIR + 'train.csv')
test_df = pd.read_csv(DATA_DIR + 'test.csv')  

(train_df.iloc[:, 2:].nunique() == test_df.iloc[:, 1:].nunique()).any()


X_train_orig = train_df.drop(["ID", "target"], axis=1)
X_test_orig = test_df.drop(["ID"], axis=1)

# Apply log transform to target variable
y_train = np.log1p(train_df["target"].values)

FOLDS = 10
SEED = 2707
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

model = lgb.LGBMRegressor(objective='regression',
                        num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=500)

# For the sake of speed just print the result
predict = cross_val_predict(model, X_train_orig, y_train, cv=kf)

print(np.sqrt(np.mean((predict-y_train) ** 2)))
print(1.4794830145766735)

# Ok, same number

def prepare(data_orig):
    data = pd.DataFrame()
    data['mean'] = data_orig.mean(axis=1)
    data['std'] = data_orig.std(axis=1)
    data['min'] = data_orig.min(axis=1)
    data['max'] = data_orig.max(axis=1)
    data['number_of_different'] = data_orig.nunique(axis=1)               # Number of different values in a row.
    data['non_zero_count'] = data_orig.fillna(0).astype(bool).sum(axis=1) # Number of non zero values (e.g. transaction count)
    return data


# Replace 0 with NaN to ignore them.
X_test = prepare(X_test_orig.replace(0, np.nan))
X_train = prepare(X_train_orig.replace(0, np.nan))

predict = cross_val_predict(model, X_train, y_train, cv=kf)
print(np.sqrt(np.mean((predict-y_train) ** 2)))

# => 1.3850759583660452

fillna on std. why are there nulls?


X = X_test_orig.replace(0, np.nan)

X_train_orig = X_train_orig[36:37]

l = list (X_train_orig[0:1].values)

n = np.array(l)

n = n.squeeze()

n.mean()
n.var()
n.min()

a = np.ma.array(n, mask=np.isnan(n)) # Use a mask to mark the NaNs

n[~a.mask].sum()
n[~a.mask].mean()
n[~a.mask].min()
n[~a.mask].max()
n[~a.mask].std()
n[~a.mask].var()

X_train_orig.sum(axis = 1)
X_train_orig.mean(axis = 1)
X_train_orig.min(axis = 1)
X_train_orig.max(axis = 1)
X_train_orig.std(axis = 1)
X_train_orig.var(axis = 1)


X = X_test_orig.replace(0, np.nan)

X

X.sum(axis = 1).isnull().sum()
X.mean(axis = 1).isnull().sum()
X.min(axis = 1).isnull().sum()
X.max(axis = 1).isnull().sum()

X.std(axis = 1).isnull().sum()
X.var(axis = 1).isnull().sum()


n = np.matrix(X)


nx = n[118, :]

a = np.ma.array(nx, mask=np.isnan(nx))

nxa = nx[~a.mask]

len (nxa)
nxa.var()
nxa.mean()

nxa.std()


105

# ===> X[115:120].std(axis = 1, ddof = 0)




