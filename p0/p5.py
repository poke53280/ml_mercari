

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



df_t = pd.read_pickle(DATA_DIR + "df_t_12AUG2018.pkl")


y = df_t['Y'].values

# Remember 'F' - fill - is a future feature. Remove from train set.
df_t = df_t.drop(['F', 'Y'], axis = 1)
df_t = df_t.drop(['ID'], axis = 1)

df_t.dtypes


df_t = df_t.drop(['D', 'MD'], axis = 1)

df_t = df_t.assign(K = df_t.K.astype('int'))

X = np.array(df_t, dtype = np.float32)


############################################################################
#
#       train_classification()
#


# from 1st seguro: gbdt,  max bin 255, learn 0.01, min data in leaf 1500, feature frac 0.7
# bagging freq 1 , bagging fraq 0.7, lambda l1 = 1, lambda l2  = 1



def train_classification(X, y):
    
    
    THRESHOLD = 17
    
    y_b = y > THRESHOLD

    NUM_FOLDS = 7

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=22)

    lKF = list (enumerate(kf.split(X)))

    l_gini = []

    a_conf_acc = np.zeros((2,2), dtype = np.int32)

    y_oof = np.zeros(len (y))

    while len(lKF) > 0:
        iLoop, (train_index, test_index) = lKF.pop(0)

        print(f"--- Fold: {iLoop +1}/ {NUM_FOLDS} ---")
        
        X_train = X[train_index]
        y_train = y_b[train_index]
    
        X_valid = X[test_index]
        y_valid = y_b[test_index]

        dr = DReduction(5)

        dr.fit(X_train)

        X_train_dr = dr.transform(X_train)
        X_train = np.hstack([X_train, X_train_dr])


        X_valid_dr = dr.transform(X_valid)
        X_valid = np.hstack([X_valid, X_valid_dr])

        clf = lgb.LGBMClassifier(n_estimators  = 5000, objective='binary', metric = 'auc', max_bin = 255, num_leaves=127, learning_rate = 0.005, silent = False, feature_fraction = 0.8, bagging_fraction = 0.75, bagging_freq = 5)

        clf.fit(X_train, y_train, verbose = True, eval_metric = 'auc', eval_set = [(X_train, y_train), (X_valid, y_valid)], early_stopping_rounds  = 150)

        y_p = clf.predict_proba(X_valid)[:,1]

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

gini_mean = []

anGINI = train_classification(X, y)
gini_mean.append(anGINI.mean())


