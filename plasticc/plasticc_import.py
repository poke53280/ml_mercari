
import numpy as np
import pandas as pd
import gc
import h5py
import os
from importlib import reload

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)

pd.set_option('display.max_colwidth', 500)

DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE

local_dir = os.getenv('LOCAL_PY_DIR')
assert local_dir is not None, "Set environment variable LOCAL_PY_DIR to parent folder of ai_lab_datapipe folder. Instructions in code."

print(f"Local python top directoy is set to {local_dir}")
os.chdir(local_dir)

from ml_mercari.plasticc.plasticc_inner import process_single_item_inner0

from ml_mercari.general.TimeSlotAllocator import *


#######################################################################
#
#     read_data
#

def read_data(filename):

    df = pd.read_csv(filename,
                usecols=["object_id", "mjd", "passband", "flux", "flux_err", "detected"],
                dtype = {'object_id': np.int32, 'mjd':np.float32, 'passband':np.uint8, 'flux':np.float32, 'flux_err': np.float32, 'detected': np.bool})


    print("Sorting...")
    df.sort_values(by = ['object_id', 'passband', 'mjd'], inplace = True)
    df = df.reset_index(drop=True)

    return df

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

gc.collect()

print("Concatenating...")

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




