

# FROM: https://www.kaggle.com/the1owl/love-is-the-answer
# by https://www.kaggle.com/the1owl

import xgboost as xgb
import lightgbm as lgb
from sklearn import *
import pandas as pd
import numpy as np

def rmsle(y, pred):
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(pred), 2)))


DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE

train = pd.read_csv(DATA_DIR + 'train.csv')
test = pd.read_csv(DATA_DIR + 'test.csv')  

y_trainFull = np.log1p(train.target)


print(train.shape)
 
def get_important_columns(train):

    scl = preprocessing.StandardScaler()

    col = [c for c in train.columns if c not in ['ID', 'target']]

    x1, x2, y1, y2 = model_selection.train_test_split(train[col], train.target.values, test_size=0.20, random_state=5)

    model = ensemble.RandomForestRegressor(n_jobs = -1, random_state = 7)

    model.fit(scl.fit_transform(x1), y1)

    print(f"RMSLE Random Forest Regressor: {rmsle(y2, model.predict(scl.transform(x2)))}")

    df = pd.DataFrame({'importance': model.feature_importances_, 'feature': col}).sort_values(by=['importance'], ascending=[False])

    df = df[:500]

    cols = df['feature'].values

    return cols

"""c"""


col = get_important_columns(train)



print("Train shape: {}\nTest shape: {}".format(train.shape, test.shape))


#Added Columns from feature_selection
train = train[['ID', 'target']+list(col)]
test = test[['ID']+list(col)]

train.shape
test.shape


### Percentage of zeros in each feature ###
PERC_TRESHOLD = 0.98   


target = np.log1p(train['target']).values

cols_to_drop = [col for col in train.columns[2:] if [i[1] for i in list(train[col].value_counts().items()) if i[0] == 0][0] >= train.shape[0] * PERC_TRESHOLD]

exclude_other = ['ID', 'target']
train_features = []

for c in train.columns:
    if c not in cols_to_drop and c not in exclude_other:
        train_features.append(c)
print("Number of featuress for training: %s" % len(train_features))

train, test = train[train_features], test[train_features]
print("\nTrain shape: {}\nTest shape: {}".format(train.shape, test.shape))

N_COMP = 20            ### Number of decomposition components ###



# Continue on notebook




