
import numpy as np
from datetime import datetime
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import gc

from numba import jit


def getNumTargets(anTarget, itarget, num_items):

    m = (anTarget == itarget)
    
    assert m.sum() >= num_items, f"Not enough target = {itarget}: {m.sum()} < {num_items}"
    
    cumsum = np.cumsum(m)
    idx = np.searchsorted(cumsum, num_items)

    m[idx +1:] = False

    return m

"""c"""


# Equal amounts of two targets, maximum number

def getBalancedSplit(anTarget, itarget0, itarget1):

    ntarget0 = (anTarget == itarget0).sum()
    ntarget1 = (anTarget == itarget1).sum()

    nTargetsForClass = np.min([ntarget0, ntarget1])

    assert nTargetsForClass > 0

    print(f"Extracting {nTargetsForClass} item(s) of either target type")

    m0 = getNumTargets(anTarget, itarget0, nTargetsForClass)
    m1 = getNumTargets(anTarget, itarget1, nTargetsForClass)

    m_both = m0 | m1

    return m_both

"""c"""



#######################################################################
#
#     get_run_length_stops
#
#       Input: sorted array
#
#      Returns one beyond index for all value firsts

def get_run_length_stops(id_s):

    m = np.diff(id_s) != 0
    a = np.where(m)[0]
    a = a + 1
    a = np.append(a, id_s.shape[0])

    a = a.astype(dtype = np.int32)

    return a

#@jit(nopython=True)
def generate_fast(data_sample, a_all, y_all, p_all, begin_offset, lengths, num_objects):

    for i in range(num_objects):
        min = begin_offset[i]
        max = min + lengths[i]

        a = a_all[min:max]
        y = y_all[min:max]
        p = p_all[min:max]

        m_p0 = (p == 0)
        m_p1 = (p == 1)
        m_p2 = (p == 2)
        m_p3 = (p == 3)
        m_p4 = (p == 4)
        m_p5 = (p == 5)

        b = 300

        N = 200

        res = np.empty((6, N), dtype = np.float32)
        res[:, :] = np.nan

        a_min = np.min(a)
        a_max = np.max(a)

        L = a_max - a_min

        assert b < L

        begin = a_min + np.random.uniform(0, L - b)
        end = begin + b

        ai = a

        ai = ai - begin
        ai = ai / b
        ai = ai * N
        ai = ai + .5
        ai = ai.astype(dtype = np.int32)
        m = (ai >= 0) & (ai < N)

        res[0, ai[m & m_p0]] = y[m & m_p0]
        res[1, ai[m & m_p1]] = y[m & m_p1]
        res[2, ai[m & m_p2]] = y[m & m_p2]
        res[3, ai[m & m_p3]] = y[m & m_p3]
        res[4, ai[m & m_p4]] = y[m & m_p4]
        res[5, ai[m & m_p5]] = y[m & m_p5]

        res = res.flatten('F')

        data_sample[i, :] = res



def generate_sets(df_meta, df, num_sets):


    

    idx = get_run_length_stops(df.object_id.values)

    idx = np.insert(idx, 0, 0)

    begin_offset = idx[:-1]
    lengths = np.diff(idx)

    num_objects = begin_offset.shape[0]

    del idx

    a_all = df['mjd'].values.astype(dtype = np.float32)
    p_all = df['passband'].values.astype(dtype = np.int32)

    y_flux = df['flux'].values.astype(dtype = np.float32)
    y_flux_err = df['flux_err'].values.astype(dtype = np.float32)

    y_all = np.random.normal(y_flux, y_flux_err).astype(dtype = np.float32)

    data = np.empty((num_sets *  num_objects, 1200), dtype = np.float32)

    generate_fast(data[0:num_objects, :], a_all, y_all, p_all, begin_offset, lengths, num_objects)


    for i in range(num_sets):
        if i % 10 == 0:
            print(f"Generating {i+1}/ {num_sets}")

        y_all = np.random.normal(y_flux, y_flux_err)

        start_idx = i * num_objects
        end_idx = start_idx + num_objects

        generate_fast(data[start_idx:end_idx, :], a_all, y_all, p_all, begin_offset, lengths, num_objects)

    
    y = np.tile(np.array (df_meta.target, dtype = np.int32), num_sets)

    id = np.tile(np.array (df_meta.object_id, dtype = np.int64), num_sets)

    return data, y, id
"""c"""



DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE

df_meta = pd.read_csv(DATA_DIR + "training_set_metadata.csv")
df = pd.read_csv(DATA_DIR + "training_set.csv")


num_sets = 60

data, y, ids = generate_sets(df_meta, df, num_sets)


num_all = y.shape[0]


ntarget0 = (y == 90).sum()
ntarget1 = (y == 52).sum()

nTargetsForClass = np.min([ntarget0, ntarget1])

print(f"# Elements per class: {nTargetsForClass}")

m0 = getNumTargets (y, 90, nTargetsForClass)
m1 = getNumTargets (y, 52, nTargetsForClass)

y[m0] = 1
y[m1] = 0

m = m0 | m1

y = y[m]
data = data[m]
ids = ids[m]

print(f"Element reduction {num_all} => 2 x {nTargetsForClass}")

gc.collect()







num_splits = 10

uniqueIDs = np.unique(ids)


folds = KFold(n_splits=num_splits, shuffle=True, random_state=11)

oof_preds = np.zeros(data.shape[0])

for n_fold, (trn_idx_unique, val_idx_unique) in enumerate(folds.split(uniqueIDs)):

    trn_ids = uniqueIDs[trn_idx_unique]
    val_ids = uniqueIDs[val_idx_unique]

    m_trn = np.in1d(ids, trn_ids) 
    m_val = np.in1d(ids, val_ids)
        
    trn_x, trn_y = data[m_trn], y[m_trn]
    val_x, val_y = data[m_val], y[m_val]

    print (f"Fold {n_fold + 1} / {num_splits}")

    clf = LGBMClassifier(n_estimators=20000, learning_rate=0.01, num_leaves = 255, silent=-1, verbose=-1)

    clf.fit(trn_x, trn_y,  eval_set= [(trn_x, trn_y), (val_x, val_y)], eval_metric='auc', verbose=10, early_stopping_rounds=400)  

    oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]

    print('Fold %2d AUC : %.3f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
    del clf, trn_x, trn_y, val_x, val_y
    gc.collect()


print('Full AUC score %.3f' % roc_auc_score(y, oof_preds)) 



