
import numpy as np
import pandas as pd
import gc
from datetime import datetime
import sys

from scipy.stats import skew, kurtosis

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier




DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE

filename = DATA_DIR + 'data.h5'


####################################################################################
#
#    process_single_get_data_out_size()
#

def process_single_get_data_out_size():
    return 52 * 6


####################################################################################
#
#    process_single_get_data_out_type()
#

def process_single_get_data_out_type():
    return np.float32

####################################################################################
#
#    get_stats()
#

def get_stats(afValues, afData, b):
    stats = np.array([afValues.shape[0], afValues.min(), afValues.max(), afValues.mean(), np.median(afValues), afValues.std(), afValues.sum(), skew(afValues), kurtosis(afValues),np.percentile(afValues, q = 5), np.percentile(afValues, q = 25), np.percentile(afValues, q = 75),  np.percentile(afValues, q = 95)])

    afData[b: b + stats.shape[0]] = stats

    return b + stats.shape[0]

####################################################################################
#
#    sample_and_get_stats()
#

def sample_and_get_stats(x, y, num_samples, afData, iWrite):

    x_min = np.min(x)
    x_max = np.max(x)
    
    slot_x = np.linspace(x_min, x_max, num_samples, endpoint=True)

    y_out = np.interp(slot_x, x, y, left=None, right=None, period=None)

    iWrite = get_stats(y_out, afData, iWrite)

    return iWrite


####################################################################################
#
#    process_single_item_inner0()
#

def process_single_item_inner0(df, idx_begin, idx_end, data_out):
    

    all_bands = np.array([0, 1, 2, 3, 4, 5])

    start_indinces = np.searchsorted(df.iloc[idx_begin:idx_end].passband.values, all_bands, side = 'left')

    stop_indices = start_indinces[1:]
    stop_indices = np.append(stop_indices, df.iloc[idx_begin:idx_end].passband.shape[0])

    iWrite = 0

    for b, e in zip (start_indinces, stop_indices):

        v = df.iloc[idx_begin:idx_end][b:e]

        assert e > b, "e > b"

        iWrite = get_stats(v.flux.values, data_out, iWrite)
        iWrite = get_stats(v.flux_err.values, data_out, iWrite)

        iWrite = get_stats(v.mjd.values - np.min(v.mjd.values), data_out, iWrite)

        num_samples = 20

        iWrite = sample_and_get_stats(v.mjd.values, v.flux.values, num_samples, data_out, iWrite)

"""c"""


####################################################################################
#
#    processRowBatch()
#

def processRowBatch(store, idx_chunk_begin, idx_chunk_length):
    num_objects = idx_chunk_begin.shape[0]

    row_min = idx_chunk_begin[0]
    row_max = idx_chunk_begin[-1] + idx_chunk_length[-1]

    df = store.select('data', start = row_min , stop = row_max)
    df = df.reset_index(drop = True)


    datasize_per_object = process_single_get_data_out_size()
    data_out_type = process_single_get_data_out_type()

    data_out = np.empty(shape = (idx_chunk_begin.shape[0], datasize_per_object), dtype = data_out_type)
    
    assert data_out.shape[0] == num_objects

    for loc_obj_id in range (0, num_objects):

        if loc_obj_id % 50 == 0:
            print(f"Processing {loc_obj_id}/ {num_objects}...")

        idx_begin = idx_chunk_begin[loc_obj_id] - row_min
        idx_end = idx_begin + idx_chunk_length[loc_obj_id]

        process_single_item_inner0(df, idx_begin, idx_end, data_out[loc_obj_id, :])

    return data_out

 

#######################################################
#
#    process_train_set
#

def process_train_set(store, begin_offset, lengths):

# 7848 train items
    train_begin = begin_offset[-7848:]
    train_length = lengths[-7848:]
    
    data_out = processRowBatch(store, train_begin, train_length)

    return data_out




#######################################################
#
#    process_chunk_set
#

def process_chunk_set(store, l_chunk_begin, l_chunk_length, processChunks):
    
    l_chunk_data_out = []

    for iChunk in processChunks:

        idx_chunk_begin = l_chunk_begin[iChunk]
        idx_chunk_length = l_chunk_length[iChunk]

        data_out = processRowBatch(store, idx_chunk_begin, idx_chunk_length)

        l_chunk_data_out.append(data_out)

        print(f"Chunk {iChunk +1}/{processChunks.shape[0]} completed")
        
    return l_chunk_data_out
        

"""c"""



def train():
    store = pd.HDFStore(filename)

    df_idx = pd.read_hdf(store, 'idx')

    idx = np.array(df_idx[0], dtype = np.int32)

    idx = np.insert(idx, 0, 0)

    begin_offset = idx[:-1]
    lengths = np.diff(idx)

    del df_idx
    del idx

    gc.collect()


    data = process_train_set(store, begin_offset, lengths)

    meta = pd.read_csv(DATA_DIR + "training_set_metadata.csv")   

    y = np.array (meta.target, dtype = np.int32)

    m = y == 90

    y[m] = 1
    y[~m] = 0

    num_splits = 8

    folds = KFold(n_splits=num_splits, shuffle=True, random_state=11)

    oof_preds = np.zeros(data.shape[0])

    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data)):
        
        trn_x, trn_y = data[trn_idx], y[trn_idx]
        val_x, val_y = data[val_idx], y[val_idx]

        print (n_fold)

        clf = LGBMClassifier(n_estimators=20000, learning_rate=0.01, num_leaves = 255, silent=-1, verbose=-1)


        clf.fit(trn_x, trn_y,  eval_set= [(trn_x, trn_y), (val_x, val_y)], eval_metric='auc', verbose=25, early_stopping_rounds=400)  

        oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]

        print('Fold %2d AUC : %.3f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()


    print('Full AUC score %.3f' % roc_auc_score(y, oof_preds)) 


    clf = LGBMClassifier(n_estimators=200, learning_rate=0.01, max_depth=5, num_leaves = 31, silent=-1, verbose=-1)
    clf.fit(data, y,  eval_set= [(data, y)], eval_metric='auc', verbose=25, early_stopping_rounds=400) 

    return clf


def predict(l, clf):
    
    num_rows = 0
    
    for data in l:
        num_rows = num_rows + data.shape[0]

    y_pred = np.empty(shape=num_rows)

    offset = 0

    for data in l:
        y_pred[offset: offset + data.shape[0]] = clf.predict_proba(data, num_iteration=clf.best_iteration_)[:, 0]

    return pd.Series(y_pred)           











####################################################################################
#
#   run()
#
#

def run(i_split):

    n_split = 16

    print(f"Running split {i_split} / {n_split}")

    clf = train()  # Fix

    store = pd.HDFStore(filename)

    df_idx = pd.read_hdf(store, 'idx')

    idx = np.array(df_idx[0], dtype = np.int32)

    idx = np.insert(idx, 0, 0)

    begin_offset = idx[:-1]
    lengths = np.diff(idx)

    del df_idx
    del idx

    gc.collect()


    print(f"Row count in dataset: {begin_offset[-1] + lengths[-1]}. Number of objects: {begin_offset.shape[0]}")

   

    num_chunks = 4000

    an = np.array(range(num_chunks))

    all_splits = np.array_split(an, n_split)

    l_chunk_begin = np.array_split(begin_offset, num_chunks)
    l_chunk_length = np.array_split(lengths, num_chunks)

    assert len(l_chunk_begin) == num_chunks
    assert len(l_chunk_length) == num_chunks

    print(f"Number of chunks: {num_chunks}")

    assert i_split >= 0 and i_split < n_split
    processChunks = all_splits[i_split]

    start = datetime.now()

    l = process_chunk_set(store, l_chunk_begin, l_chunk_length, processChunks)

    s_pred = predict(l, clf)

    s_pred.to_pickle(DATA_DIR + "y_pred_split_" + str (i_split) + ".pkl")
    
    end = datetime.now()

    dT = end - start
    dSeconds = dT.total_seconds()

    print(f"Seconds {dSeconds} for 1 split out of {n_split}")



if __name__ == "__main__":
    l = []
    for arg in sys.argv[1:]:
        l.append(arg)

    i_split = int(l[0])

   

    run (i_split)
    


