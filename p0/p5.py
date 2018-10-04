

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


from scipy.stats import skew, kurtosis

def get_prefixed_dict(d, prefix):
    d_prefixed = {}

    for key, value in d.items():
        d_prefixed[prefix + key] = value

    return d_prefixed

"""c"""

def get_stats_on_array(v):

    if len(v) == 0:
        return {'count': 0, 'mean': 0, 'std': 0, 'max': 0, 'min':0, 'sum': 0, 'skewness': 0, 'kurtosis': 0, 'median': 0, 'q1': 0, 'q3': 0}


    d = {'count': len(v), 'mean': v.mean(), 'std': v.std(), 'max': v.max(), 'min':v.min(), 'sum': v.sum(), 'skewness': skew(v), 'kurtosis': kurtosis(v), 'median': np.median(v),
         'q1': np.percentile(v, q=25), 'q3': np.percentile(v, q=75)}

    return d

DATA_DIR_PORTABLE = "X:\\p_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE


df_t = pd.read_pickle(DATA_DIR + "df_t_14AUG2018.pkl")
df = pd.read_pickle(DATA_DIR + "df_14AUG2018.pkl")


#############################################
#
#  Investigate D, MD and Length
#
#
# L - all raw lengths 
#

# Todo : Use T1

df = df.assign(L = (1 + df.T0 - df.F1))

df = df.drop(['F0', 'T0'], axis = 1)
df = df.drop(['ID'], axis = 1)

l = list(df['L'].groupby(by = df['D']))

d_info = {}

for a in l:
    d_id = a[0]
    data = a[1]
    an = np.array(data)
    d = get_stats_on_array(an)
    d_info[d_id] = d


def d_frame(x):
    d = {}

    if x in d_info:
        d = d_info[x]
    else:
        d = get_stats_on_array([])


    d = get_prefixed_dict(d, 'dcurr_')
    return pd.Series(d)

q = df_t.D.apply(d_frame)

df_t = pd.concat([df_t, q], axis = 1)


l = list(df['L'].groupby(by = df['MD']))

d_info = {}

for a in l:
    d_id = a[0]
    data = a[1]
    an = np.array(data)
    d = get_stats_on_array(an)
    d_info[d_id] = d


def d_frame2(x):
    d = {}

    if x in d_info:
        d = d_info[x]
    else:
        d = get_stats_on_array([])


    d = get_prefixed_dict(d, 'mdcurr_')
    return pd.Series(d)


q = df_t.MD.apply(d_frame)

df_t = pd.concat([df_t, q], axis = 1)


df_t.to_pickle(DATA_DIR + "df_t_p5_17AUG2018.pkl")


######################################################################
#
#
#       get_cut_off_threshold
#
#

def get_cut_off_threshold(false_factor, y_p):
    
    assert false_factor > 0 and false_factor < 1
 
    y_p_sorted = np.sort(np.array(y_p))

    y_false_number = false_factor * len (y_p_sorted)

    iCutIndex = int (y_false_number)

    threshold_prob = y_p_sorted[iCutIndex]

    return threshold_prob





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


import pandas as pd

DATA_DIR_PORTABLE = "C:\\p_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE

df_t = pd.read_pickle(DATA_DIR + "df_t_p5_17AUG2018.pkl")


y = df_t['Y'].values

# Remember 'F' - fill - is a future feature. Remove from train set.
df_t = df_t.drop(['F', 'Y'], axis = 1)
df_t = df_t.drop(['ID'], axis = 1)

df_t.dtypes

# Basic feature importance

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators= 100)


model.fit(df_t, (y > 17))

importances = model.feature_importances_

sorted_feature_importance = sorted(zip(importances, list(df_t)), reverse=True)
print (sorted_feature_importance)

#df_t = df_t.drop(['D', 'MD'], axis = 1)
#df_t = df_t.assign(K = df_t.K.astype('int'))



############################################################################
#
#       train_classification()
#

def train_classification(df_t, y):
    
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

        dr = DReduction(5)

        dr.fit(X_train)

        X_train_dr = dr.transform(X_train)

        #cols = X_train_dr.columns
        #matching = [s for s in cols if "pca" in s]
        #q = X_train_dr[matching]

        X_train_dr = np.array(X_train_dr, dtype = np.float32)
        X_train = np.array(X_train, dtype = np.float32)


        X_train = np.hstack([X_train, X_train_dr])


        X_valid_dr = dr.transform(X_valid)

        #q = X_valid_dr[matching]

        X_valid_dr = np.array(X_valid_dr, dtype = np.float32)
        X_valid = np.array(X_valid, dtype = np.float32)

        X_valid = np.hstack([X_valid, X_valid_dr])

        rPosWeight = 1.0 / (y_b.sum() / (len (y_b) - y_b.sum()))

        clf = lgb.LGBMClassifier(scale_pos_weight = rPosWeight, n_estimators  = 9000, objective='binary', metric = 'auc', max_bin = 255, num_leaves=63, learning_rate = 0.01, silent = False, feature_fraction = 0.8, bagging_fraction = 0.75, bagging_freq = 3)

        clf.fit(X_train, y_train, verbose = 50, eval_metric = 'auc', eval_set = [(X_train, y_train), (X_valid, y_valid)], early_stopping_rounds  = 300)

        y_p = clf.predict_proba(X_valid)[:,1]

        y_oof[test_index] = y_p

        auc = roc_auc_score(y_valid, y_p)

        gini = 2 * auc - 1

        print(f"AUC: {auc}. Gini: {gini}")

        l_gini.append(gini)

        # False frequency.
        fFalseRatio = (len (y_b) - y_b.sum()) / len (y_b)

        cut_off_value = get_cut_off_threshold(fFalseRatio, y_p)

        y_pred = (y_p >= cut_off_value)

        conf_this = confusion_matrix(y_valid, y_pred)

        a_conf_acc += conf_this


    anGINI = np.array(l_gini)

    print(f"N = {df_t.shape[0]}, Folds = {NUM_FOLDS}")
    print(f"True lo: {len (y_b) - y_b.sum()}, true hi: { y_b.sum()} ")
    print(f"GINI {anGINI.mean()} +/- {anGINI.std()} @positive > {THRESHOLD}." )

    a_conf_acc = a_conf_acc / len (y)
    a_conf_acc *= 100.0

    df = pd.DataFrame(a_conf_acc,columns=['pred_lo', 'pred_hi'])
    df.index = pd.Series(['true_lo', 'true_hi'])

    print(df)

    return anGINI

"""c"""

gini_mean = []

anGINI = train_classification(df_t, y)
gini_mean.append(anGINI.mean())



