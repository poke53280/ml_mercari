

import pandas as pd
import numpy as np

import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix



from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FastICA
from sklearn.random_projection import SparseRandomProjection
from sklearn.random_projection import GaussianRandomProjection



class DReduction:

    _N_COMP = 0            ### Number of decomposition components ###

    _pca    = 0
    _tsvd   = 0
    _ica    = 0
    _grp    = 0
    _srp    = 0

    def __init__(self, nComp):
        self._N_COMP = nComp
        self._pca = PCA(n_components=self._N_COMP, random_state=17)
        self._tsvd = TruncatedSVD(n_components=self._N_COMP, random_state=17)
        self._ica = FastICA(n_components=self._N_COMP, random_state=17)
        self._grp = GaussianRandomProjection(n_components=self._N_COMP, eps=0.1, random_state=17)
        self._srp = SparseRandomProjection(n_components=self._N_COMP, dense_output=True, random_state=17)


    def fit(self, X):
        self._pca.fit(X)
        self._tsvd.fit(X)
        self._ica.fit(X)
        self._grp.fit(X)
        self._srp.fit(X)


    def transform(self, X):
        res_pca  = self._pca.transform(X)
        res_tsvd = self._tsvd.transform(X)
        res_ica  = self._ica.transform(X)
        res_grp  = self._grp.transform(X)
        res_srp  = self._srp.transform(X)


        df = pd.DataFrame()

        for i in range(1, self._N_COMP + 1):
            df['pca_' + str(i)] = res_pca[:, i - 1]
            df['tsvd_' + str(i)] = res_tsvd[:, i - 1]
            df['ica_' + str(i)] = res_ica[:, i - 1]
            df['grp_' + str(i)] = res_grp[:, i - 1]
            df['srp_' + str(i)] = res_srp[:, i - 1]

        return df
"""c"""





DATA_DIR_PORTABLE = "C:\\p_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE



df_t = pd.read_pickle(DATA_DIR + "df_t_10AUG2018.pkl")


y = df_t['Y'].values

# Remember 'F' - fill - is a future feature. Remove from train set.
df_t = df_t.drop(['F', 'Y'], axis = 1)
df_t = df_t.drop(['ID'], axis = 1)

df_t.dtypes

df_t = df_t.assign(K = df_t.K.astype('category'))
df_t = df_t.assign(MD = df_t.MD.astype('category'))
df_t = df_t.assign(D = df_t.D.astype('category'))

# Todo category to stats for MD, D




############################################################################
#
#       train_classification()
#


def train_classification(df_t, y, params):
    
    
    THRESHOLD = 17
    
    y_b = y > THRESHOLD

    NUM_FOLDS = 7

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=22)

    lKF = list (enumerate(kf.split(df_t)))

    l_gini = []

    a_conf_acc = np.zeros((2,2), dtype = np.int32)

    y_oof = np.zeros(len (y))

    while len(lKF) > 0:
        iLoop, (train_index, test_index) = lKF.pop(0)

        print(f"--- Fold: {iLoop +1}/ {NUM_FOLDS} ---")
        
        X_train = df_t.iloc[train_index]
        y_train = y_b[train_index]
    
        X_valid = df_t.iloc[test_index]
        y_valid = y_b[test_index]

        isOneHot = False

        if isOneHot:
            X_train = pd.get_dummies(X_train)
            X_valid = pd.get_dummies(X_valid)

        else:
            pass


        dr = DReduction(20)

        dr.fit(X_train)

        X_train = pd.concat([X_train.reset_index(), dr.transform(X_train).reset_index()], axis = 1, ignore_index = True)
        X_valid = pd.concat([X_valid.reset_index(), dr.transform(X_valid).reset_index()], axis = 1, ignore_index = True)

        lgtrain = lgb.Dataset(data=X_train, label=y_train)
        lgvalid = lgb.Dataset(data=X_valid, label=y_valid)

        clf = lgb.train(params, lgtrain, num_boost_round=15000, early_stopping_rounds=300, valid_sets= [lgtrain, lgvalid], verbose_eval=50)

        y_p = clf.predict(X_valid)

        y_oof[test_index] = y_p

        auc = roc_auc_score(y_valid, y_p)

        gini = 2 * auc - 1

        print(f"AUC: {auc}. Gini: {gini}")

        l_gini.append(gini)


    anGINI = np.array(l_gini)

    print(f"N = {df_t.shape[0]}, Folds = {NUM_FOLDS}")
    print(f"GINI {anGINI.mean()} +/- {anGINI.std()} @positive > {THRESHOLD}." )

    return anGINI

"""c"""


def getParams():
    params = {}
    params['task'] = 'train'
    params['boosting_type'] = 'gbdt'
    params['objective'] = 'binary'
    params['metric'] = 'auc'
    params['max_bin'] = 255
    params['feature_fraction'] = 0.8
    
    params['learning_rate'] = 0.005
    
    params['num_leaves'] = np.random.choice([ 63])
    params['bagging_freq'] = 3
    params['boosting_type'] = np.random.choice(['gbdt'])
    
    params['bagging_fraction'] = 0.75

    return params


gini_mean = []
param_list = []

for x in range(4):

    params = getParams()
    print (params)
    anGINI = train_classification(df_t, y, params)
    gini_mean.append(anGINI.mean())
    param_list.append(params)



# Feature engineering D -> Length stats. MD -> Length stats. (D, MD) -> Length stats. Then remove D, MD.

df_t.D.value_counts()

m = df_t.D == 5924

q = df_t[m]

y_q = y[m]

y_q.min()