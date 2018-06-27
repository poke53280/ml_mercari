

# FROM: https://www.kaggle.com/the1owl/love-is-the-answer
# by https://www.kaggle.com/the1owl

import xgboost as xgb
import lightgbm as lgb
from sklearn import *
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FastICA
from sklearn.random_projection import SparseRandomProjection
from sklearn.random_projection import GaussianRandomProjection



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


print("\nStart decomposition process...")



N_COMP = 20            ### Number of decomposition components ###


print("PCA")

pca = PCA(n_components=N_COMP, random_state=17)

pca_results_train = pca.fit_transform(train)
pca_results_test = pca.transform(test)


print("tSVD")
tsvd = TruncatedSVD(n_components=N_COMP, random_state=17)
tsvd_results_train = tsvd.fit_transform(train)
tsvd_results_test = tsvd.transform(test)

print("ICA")
ica = FastICA(n_components=N_COMP, random_state=17)
ica_results_train = ica.fit_transform(train)
ica_results_test = ica.transform(test)

print("GRP")
grp = GaussianRandomProjection(n_components=N_COMP, eps=0.1, random_state=17)
grp_results_train = grp.fit_transform(train)
grp_results_test = grp.transform(test)

print("SRP")
srp = SparseRandomProjection(n_components=N_COMP, dense_output=True, random_state=17)
srp_results_train = srp.fit_transform(train)
srp_results_test = srp.transform(test)


print("Append decomposition components to datasets...")

for i in range(1, N_COMP + 1):
    train['pca_' + str(i)] = pca_results_train[:, i - 1]
    test['pca_' + str(i)] = pca_results_test[:, i - 1]

    train['ica_' + str(i)] = ica_results_train[:, i - 1]
    test['ica_' + str(i)] = ica_results_test[:, i - 1]

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]


print('\nTrain shape: {}\nTest shape: {}'.format(train.shape, test.shape))

type (train)
type (test)

train = pd.concat([train, df_train6], axis = 1)
test = pd.concat([test, df_test6], axis = 1)

