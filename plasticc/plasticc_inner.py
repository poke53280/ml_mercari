
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis



DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE

# jump start from scratch

df_meta = pd.read_csv(DATA_DIR + "training_set_metadata.csv")
df = pd.read_csv(DATA_DIR + "training_set.csv")

# Lots of data:
m = df.object_id == 130659834
df[m]


q = df[m].sort_values(by = ['mjd'])

df = q

idx_begin = 0
idx_end = df.shape[0]

datasize_per_object = process_single_get_data_out_size()
data_out_type = process_single_get_data_out_type()


data_out = np.zeros(shape = (3, datasize_per_object), dtype = data_out_type)


process_single_item_inner0(df, idx_begin, idx_end, data_out[2, :])


