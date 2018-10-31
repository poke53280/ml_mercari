
import numpy as np
import pandas as pd
import gc
from datetime import datetime

from scipy.stats import skew, kurtosis

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
    stats = np.array([afValues.shape[0], afValues.min(), afValues.max(), afValues.mean(), np.median(afValues), afValues.std(), afValues.sum(),skew(afValues), kurtosis(afValues), np.percentile(afValues, q = 5),  np.percentile(afValues, q = 25), np.percentile(afValues, q = 75),  np.percentile(afValues, q = 95)])

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
#    get_data()
#

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
#    processChunk()
#

def processChunk(df, idx_chunk, idx_start_of_first, data_out):

    num_objects = idx_chunk.shape[0]

    print(f"Objects in chunk: {num_objects}")

    assert data_out.shape[0] == num_objects

    for loc_obj_id in range (0, num_objects):

        if loc_obj_id % 50 == 0:
            print(f"Processing {loc_obj_id}/ {num_objects}...")


        if loc_obj_id == 0:
            from_idx = 0
        else:
            from_idx = idx_chunk[loc_obj_id - 1]

        to_idx = idx_chunk[loc_obj_id]

        assert to_idx > from_idx

        idx_begin = from_idx - idx_start_of_first
        idx_end   = to_idx - idx_start_of_first

        process_single_item_inner0(df, idx_begin, idx_end, data_out[loc_obj_id, :])
    
 

#######################################################
#
#    process_chunk_set
#

def process_chunk_set(l_chunk_idx, processChunks):
    
    datasize_per_object = process_single_get_data_out_size()
    data_out_type = process_single_get_data_out_type()

    nElementCount = 0

    l_chunk_data_out = []

    for iChunk in processChunks:

        idx_chunk = l_chunk_idx[iChunk]

        # Start of first object
        if iChunk == 0:
            idx_start_of_first = 0
        else:
            idx_start_of_first = l_chunk_idx[iChunk -1][-1]

        # Process objects in chunk
        idx_start_object = 0
        idx_end_object = idx_chunk.shape[0]

        # Read in all chunk data in dataframe

        df = get_data(idx_chunk, idx_start_object, idx_end_object, idx_start_of_first)

        # Allocate out memory
       
        data_out = np.empty(shape = (idx_chunk.shape[0], datasize_per_object), dtype = data_out_type)

        processChunk(df, idx_chunk, idx_start_of_first, data_out)

        l_chunk_data_out.append(data_out)

        #print(f"Chunk {iChunk +1}/{num_chunks} with {idx_chunk.shape[0]} object(s) and {df.shape[0]} element(s). Time = {dT:.1f}s")

        nElementCount += df.shape[0]

    return nElementCount, l_chunk_data_out
        

"""c"""


####################################################################################
#
# Reset. load
#


store = pd.HDFStore(filename)

df_idx = pd.read_hdf(store, 'idx')

idx = np.array(df_idx[0], dtype = np.int32)

del df_idx

gc.collect()

print(f"Row count in dataset: {idx[-1]}. Number of objects: {idx.shape[0]}")

n_split = 3000

num_chunks = 40000

an = np.array(range(num_chunks))

all_splits = np.array_split(an, n_split)

l_chunk_idx = np.array_split(idx, num_chunks)

assert len(l_chunk_idx) == num_chunks

print(f"Number of chunks: {num_chunks}")

i_split = 0
assert i_split >= 0 and i_split < n_split
processChunks = all_splits[i_split]


start = datetime.now()

c, l = process_chunk_set(l_chunk_idx, processChunks)

end = datetime.now()

dT = end - start
dSeconds = dT.total_seconds()

print(f"Seconds {dSeconds} for 1 split out of {n_split}")





if __name__ == "__main__":
    main()

