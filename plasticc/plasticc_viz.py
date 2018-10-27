
import numpy as np
import pandas as pd
import gc
import h5py

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)

pd.set_option('display.max_colwidth', 500)

DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE


#######################################################################
#
#     read_data
#

def read_data(filename):

    return pd.read_csv(filename,
                usecols=["object_id", "mjd", "passband", "flux", "flux_err", "detected"],
                dtype = {'object_id': np.int32, 'mjd':np.float32, 'passband':np.uint8, 'flux':np.float32, 'flux_err': np.float32, 'detected': np.bool})

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
"""c"""

data_test = read_data(DATA_DIR + 'test_set.csv')
data_train = read_data(DATA_DIR + 'training_set.csv')

df = pd.concat([data_test, data_train], axis = 0 )

del data_test
del data_train
gc.collect()

idx = get_run_length_stops(df.object_id)

print (df.shape)

df_idx = pd.DataFrame(idx)

filename = DATA_DIR + 'data.h5'

store = pd.HDFStore(filename)

store.put('data', df)
store.put('idx', df_idx)

store.close()

del df
del idx

gc.collect()



def get_data(idx_chunk, idx_start_object, idx_end_object, idx_start_of_first):

    assert idx_end_object > 0
    assert idx_start_object < idx_end_object

    assert idx_end_object <= idx_chunk.shape[0]

    end_idx = idx_chunk[idx_end_object -1]

    if idx_start_object == 0:
        begin_idx = idx_start_of_first
    else:
        begin_idx = idx_chunk[idx_start_object -1]

    assert begin_idx < end_idx

    q = store.select('data', start = begin_idx , stop = end_idx)

    return q

"""c"""  

####################################################################################
#
#    process_single_item()
#


def process_single_item(df, idx_begin, idx_end):
    an_id = df.object_id.values[idx_begin:idx_end]

    assert an_id.min() == an_id.max()

    return an_id.shape[0]
"""c"""


####################################################################################
#
#    processChunk()
#

def processChunk(df, idx_chunk, idx_start_of_first):

    print(f"Objects in chunk: {idx_chunk.shape[0]}")
    
    nRowCount = 0

    for loc_obj_id in range (0, idx_chunk.shape[0]):

        if loc_obj_id == 0:
            from_idx = 0
        else:
            from_idx = idx_chunk[loc_obj_id - 1]

        to_idx = idx_chunk[loc_obj_id]

        assert to_idx > from_idx

        idx_begin = from_idx - idx_start_of_first
        idx_end   = to_idx - idx_start_of_first

        nRowCount += process_single_item(df, idx_begin, idx_end)
    
    return nRowCount

####################################################################################
#
# Reset. load
#

import numpy as np
import pandas as pd
import gc
from datetime import datetime

DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE

filename = DATA_DIR + 'data.h5'

store = pd.HDFStore(filename)

df_idx = pd.read_hdf(store, 'idx')

idx = np.array(df_idx[0], dtype = np.int32)

del df_idx

gc.collect()

print(f"Row count in dataset: {idx[-1]}. Number of objects: {idx.shape[0]}")

n_split = 1

num_chunks = 400

an = np.array(range(num_chunks))

all_splits = np.array_split(an, n_split)

l_chunk_idx = np.array_split(idx, num_chunks)

assert len(l_chunk_idx) == num_chunks

print(f"Number of chunks: {num_chunks}")


i_split = 0
assert i_split >= 0 and i_split < n_split
processChunks = all_splits[i_split]


c = process_chunk_set(l_chunk_idx, processChunks)



def process_chunk_set(l_chunk_idx, processChunks):
    
    nElementCount = 0

    for iChunk in processChunks:

        idx_chunk = l_chunk_idx[iChunk]

        # Start of first object
        if iChunk == 0:
            idx_start_of_first = 0
        else:
            idx_start_of_first = l_chunk_idx[iChunk -1][-1]

        # All objects in chunk
        idx_start_object = 0
        idx_end_object = idx_chunk.shape[0]

        df = get_data(idx_chunk, idx_start_object, idx_end_object, idx_start_of_first)

        rowcount = processChunk(df, idx_chunk, idx_start_of_first)

        assert rowcount == df.shape[0]

        #print(f"Chunk {iChunk +1}/{num_chunks} with {idx_chunk.shape[0]} object(s) and {df.shape[0]} element(s). Time = {dT:.1f}s")

        nElementCount += df.shape[0]

    return nElementCount
        

"""c"""

print(f"Read elements: {nElementCount}")

assert nElementCount == idx[-1]

total_end = datetime.now()

dT_total = total_end - total_start

dT_total = dT_total.total_seconds()

print(f"Total time = {dT_total:.1f}s")











    res_G = np.empty(shape = num_rows, dtype = np.uint16)
    res_mjd = np.empty(shape = num_rows, dtype = np.uint16)

    for idx in range(num_items):

        if idx % 100 == 0:
            print(f"Processing item {idx}/{num_items}...")

        start_idx = idx_array[idx]

        if idx == num_items -1:
            end_idx = num_rows
        else:
            end_idx = idx_array[idx +1]

        armjd = mjd_s[start_idx: end_idx]

        armjd -= np.min(armjd)

        m = np.diff(armjd) > 2

        cluster = np.concatenate([[0], np.cumsum(m)]) 

        res_G[start_idx: end_idx] = cluster

        nClusters = cluster.max()

        for c in range (nClusters):
            m = (cluster == c)

            time_adjusted = np.min(armjd[m])
            time_adjusted = int (0.5 + time_adjusted)

            res_mjd[start_idx:end_idx][m] = time_adjusted
    """c""" 

    print(f"{res_mjd.shape}")

# data = data.assign(G = res_G)
# data = data.assign(T_ADJ = res_mjd)

if __name__ == "__main__":
    main()

