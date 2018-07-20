

import pandas as pd
import numpy as np

from general.KDE_study import group_sorted_unique_integers
from scipy.stats import skew, kurtosis

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from santander_3.lgbm_basic import LGBMTrainer_BASIC

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FastICA
from sklearn.random_projection import SparseRandomProjection
from sklearn.random_projection import GaussianRandomProjection



DATA_DIR_PORTABLE = "C:\\p_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE

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




######################################################################################
#
#  get_important_columns
#
#
# Based on: https://www.kaggle.com/the1owl/love-is-the-answer
#
# by    https://www.kaggle.com/the1owl
#
#  dont_consider: List of named columns not to process and consider
#
#  Returns names of the nCut most important columns.
#


class RegImportance:

    _df = 0

    def __init__(self):
        pass         
    
    def rmsle(self, y, pred):
        return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(pred), 2)))

    def fit(self, df, y_true):
        scl = StandardScaler()

        col = [c for c in df.columns]

        x1, x2, y1, y2 = train_test_split(df[col], y_true, test_size=0.20, random_state=5)

        model = RandomForestRegressor(n_jobs = -1, random_state = 7)

        model.fit(scl.fit_transform(x1), y1)

        print(f"RMSLE Random Forest Regressor: {self.rmsle(y2, model.predict(scl.transform(x2)))}")

        self._df = pd.DataFrame({'importance': model.feature_importances_, 'feature': col}).sort_values(by=['importance'], ascending=[False])


    def get_important(self, nCut):
        df = self._df[:nCut]    
        cols = df['feature'].values
        return cols

"""c"""


##################################################################################
#
#       get_group_list
#
#
                
def get_group_list(g, colname):
    l = list (g[colname])
    assert len (l) > 0

    l = l[0][1]
    l = l.values

    return l


##################################################################################
#
#       get_day_set
#
#

def get_day_set(df, m):

    def add_to_set(x, my_set):
        begin = x['begin']
        end   = x['end']
        my_set.update ( range(begin, end + 1))

    days = set()

    _ = df[m].apply(add_to_set, axis = 1, args = (days, ))

    anDays = np.array(list(days))
    anDays = np.sort(anDays)

    return anDays


##################################################################################
#
#       group_periods
#
#

def group_periods(df, p_id, n_bandwidth):

    m = (df.P == p_id)

    an = get_day_set(df, m)

    l = group_sorted_unique_integers(an, n_bandwidth, False)
    
    return l


"""c"""

##################################################################################################
#
#       extract_targets
#

def extract_targets(df, Target_begin_span, n_bandwidth):


    targets = []

    anRows = df.P.unique()

    nRows = len (anRows)

    assert anRows.max() +1 == nRows

    print(f"Extracting targets in range {Target_begin_span[0]} - {Target_begin_span[1]}...")


    for i in range(nRows):
        l = group_periods(df, i, n_bandwidth)

        if i%500 == 0:
            print(f"Processing ID = {i} out of {nRows}...")

        for x in l:
            isTargetCandidate = x[0] >= Target_begin_span[0] and x[0] <= Target_begin_span[1]

            if isTargetCandidate:
                target = {}
                target['id'] = i
            
                L = 1 + x[1] - x[0]
            
                target['L'] = L
                target['Begin'] = x[0]

                targets.append(target)
                break

    rPct = 100.0 * len(targets) / nRows
    print(f"{len(targets)} of {nRows} ({rPct:.1f}%) IDs in target zone")


    df_t = pd.DataFrame(targets)

    df_t.columns = ['T0', 'Y', 'id']

    df_t = df_t[['id', 'T0', 'Y']]

    return df_t


##################################################################################################
#
#       get_stats_on_array
#

def get_stats_on_array(v):

    if len(v) == 0:
        return {'mean': 0, 'std': 0, 'max': 0, 'min':0, 'sum': 0, 'skewness': 0, 'kurtosis': 0, 'median': 0, 'q1': 0, 'q3': 0,'count': 0}


    d = {'mean': v.mean(), 'std': v.std(), 'max': v.max(), 'min':v.min(), 'sum': v.sum(), 'skewness': skew(v), 'kurtosis': kurtosis(v), 'median': np.median(v),
         'q1': np.percentile(v, q=25), 'q3': np.percentile(v, q=75),'count': len(v)}

    return d

##################################################################################################
#
#       get_prefixed_dict
#

def get_prefixed_dict(d, prefix):
    d_prefixed = {}

    for key, value in d.items():
        d_prefixed[prefix + key] = value

    return d_prefixed

"""c"""


##################################################################################################
#
#       extract_column_info
#

def extract_column_info(g):

    l_begin = get_group_list(g, 'begin')
    l_end   = get_group_list(g, 'end')
    l_md    = get_group_list(g, 'MD')
    l_d     = get_group_list(g, 'D')

    assert len (l_begin) > 0

    # Some stats

    # Uniques 
    nUniqueMD = len (np.unique(np.array(l_md)))
    nUniqueD = len (np.unique(np.array(l_d)))
    
    d_uniques = { 'uniq_md' : nUniqueMD, 'uniq_d' : nUniqueD }


    anPeriod = np.array(np.array(l_end) - np.array(l_begin))

    d_begin = get_prefixed_dict (get_stats_on_array(np.array(l_begin)), 'begin_')
    d_end   = get_prefixed_dict (get_stats_on_array(np.array(l_end)), 'end_')
    d_period = get_prefixed_dict (get_stats_on_array(anPeriod), 'span_')

    # Serialized data

    str_begin = ','.join(str(e) for e in l_begin)
    str_end = ','.join(str(e) for e in l_end)
    str_md = ','.join(str(e) for e in l_md)
    str_d = ','.join(l_d)


    d_serialized = {'begin':str_begin, 'end':str_end, 'md':str_md, 'd':str_d}

    d_acc = {}

    d_acc.update(d_uniques)
    d_acc.update(d_begin)
    d_acc.update(d_end)
    d_acc.update(d_period)

    d_acc.update(d_serialized)

    return pd.Series(d_acc)



##################################################################################################
#
#       prepare_each_id
#

def prepare_each_id(x, expiry_days):
    ID = x['id']

    t0 = x['T0']
    y =  x['Y']


    m = (df.P == ID)
    
    record_ahead_time = 7

    # Discard everything that is recorded later than record_ahead_time beyond begin time
    # Check for end time.

    q = df[m].copy()

    s = q.end - t0

    q = q.assign(end = s)
    
    s = q.begin - t0
   
    q = q.assign(begin = s)

    m = (q['begin'] <= record_ahead_time) & (q['begin'] > expiry_days)

    q = q[m]

    isEmpty = (len(q) == 0)

    if isEmpty:
        return no_column_info()

    m = q['end'] >= (record_ahead_time)

    q.loc[m, 'end'] = record_ahead_time

    g = q.groupby(by = 'P')
    
    return extract_column_info(g)

# main


df = train = pd.read_csv(DATA_DIR + 'noised_intervals.csv')


df.columns = ['drop', 'begin', 'end', 'P']
df = df.drop(['drop'], axis = 1)
df = df[['P', 'begin', 'end']]

md_series = np.random.choice([0,1,2,3,4,5,6], df.shape[0])
df = df.assign(MD=md_series)

d_series = np.random.choice(['D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6'], df.shape[0])
df = df.assign(D = d_series)
df['D'] = df['D'].astype('category')


df = df.sort_values(by = ['P', 'begin'])


Target_begin_span = (16800, 19000)
n_bandwidth = 30

df_t = extract_targets(df, Target_begin_span, n_bandwidth)

w0 = df_t.apply(prepare_each_id, axis = 1, args = (-365 * 12.0,))

df_t = pd.concat([df_t, w0], axis = 1)

print(df_t.shape)


####################################################################################################


y = df_t['Y'].values
y = y.astype(np.float32)


df_t = df_t.drop(['uniq_d', 'uniq_md', 'md', 'd', 'id'], axis = 1)
df_t = df_t.drop(['begin', 'end'], axis = 1)
df_t = df_t.drop(['Y'], axis = 1)


X = np.array(df_t, dtype = np.float32)


#### CHECKPOINT - X and y.

from sklearn.datasets import make_regression

X, y = make_regression(n_samples = 5000, n_features = 3000, n_informative = 2110, bias=0.1)


from sklearn.datasets import load_boston

X, y = load_boston(return_X_y=True)


#################################################################################
#
#       train(X, y)
#
#
    
def train(X, y):

    THRESHOLD = 28

    NUM_FOLDS = 7

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=22)

    lKF = list (enumerate(kf.split(X)))

    lRMS = []
    l_auc = []

    a_conf_acc = np.zeros((2,2), dtype = np.int32)

    y_oof = np.zeros(len (y))
    prediction = np.zeros(X.shape[0])

    while len(lKF) > 0:
        iLoop, (train_index, test_index) = lKF.pop(0)

        print(f"--- Fold: {iLoop +1}/ {NUM_FOLDS} ---")
        
        X_train = X[train_index]
        y_train = y[train_index]
    
        X_valid = X[test_index]
        y_valid = y[test_index]

        d = DReduction(79)

        d.fit(X_train)

        X_train_d = np.array(d.transform(X_train))
        X_valid_d = np.array(d.transform(X_valid))


        isIncludeRaw = True

        if isIncludeRaw:
            X_train = np.hstack([X_train, X_train_d])
            X_valid = np.hstack([X_valid, X_valid_d])

        else:
            X_train = X_train_d
            X_valid = X_valid_d

                
        l = LGBMTrainer_BASIC()
        l.train_with_validation(X_train, y_train, X_valid, y_valid)

        y_p = l.predict(X_valid)

        y_oof[test_index] = y_p

        rmse_error = np.sqrt(mean_squared_error(y_p, y_valid))
        print(f"Rmsle: {rmse_error}")

        lRMS.append(rmse_error)

        # 0: Short, 1: Long
        y_true_classifier = (y_valid > THRESHOLD)
        y_pred_classifier = (y_p > THRESHOLD)

        auc_score = roc_auc_score(y_true_classifier, y_pred_classifier)
        a_confusion = confusion_matrix(y_true_classifier, y_pred_classifier)

        a_conf_acc = a_conf_acc + a_confusion

        l_auc.append(auc_score)


    anRMS = np.array(lRMS)
    anAUC = np.array(l_auc)

    print(f"N = {X.shape[0]}, Folds = {NUM_FOLDS}")
    print(f"RMS {anRMS.mean()} +/- {anRMS.std()}")
    print(f"AUC {anAUC.mean()} +/- {anAUC.std()} @threshold = {THRESHOLD}" )
    print(f"{a_conf_acc}")


"""c"""

train(X,y)
