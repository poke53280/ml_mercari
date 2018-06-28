
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


X_test = prepare(X_test_orig.replace(0, np.nan))
X_train = prepare(X_train_orig.replace(0, np.nan))

predict = cross_val_predict(model, X_train, y_train, cv=kf)
print(np.sqrt(np.mean((predict-y_train) ** 2)))

# => 1.3850759583660452


n = np.array(l)

n = n.squeeze()

n.mean()
n.var()
n.min()

a = np.ma.array(n, mask=np.isnan(n)) # Use a mask to mark the NaNs

nxa = nx[~a.mask]





