


import numpy as np
import pandas as pd
import gc
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Flatten

from keras.layers import Subtract
from keras.layers import LSTM
from keras.layers import TimeDistributed


from keras.models import Model

from keras.constraints import unitnorm

import tensorflow as tf
from sklearn.model_selection import KFold

def generate_objects(meta, df, target_id, num_readouts):

    m = meta.target == target_id

    l = list (meta[m].object_id)

    m = df.object_id.isin(l)

    df_t = df[m]

    cat = df_t.object_id.astype('category')

    o_cat = cat.cat.codes

    df_t = df_t.assign(id = o_cat)

    id = df_t.id.values.astype(np.int32)
    mjd = df_t.mjd.values
    p = df_t.passband.values
    flux = df_t.flux.values
    err = df_t.flux_err

    num_rows = id.shape[0]

    id_max = np.max(id)

    id_a = np.tile(id, num_readouts)

    for i in range(1, num_readouts):
        id_a[i * num_rows:] += (id_max + 1)
    """c"""

    mjd_a = np.tile(mjd, num_readouts)
    p_a = np.tile(p, num_readouts)
    flux_a = np.tile(flux, num_readouts)
    err_a = np.tile(err, num_readouts)

    y_a = np.random.normal(flux_a, err_a).astype(dtype = np.float32)

    df_new = pd.DataFrame({'object_id': id_a, 'passband': p_a, 'mjd': mjd_a, 'flux' : flux_a})

    return df_new

"""c"""




np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

np.set_printoptions(edgeitems=10)
np.set_printoptions(linewidth=200)
np.core.arrayprint._line_width = 480


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 500)

DATA_DIR_PORTABLE = "C:\\plasticc_data\\"
DATA_DIR_BASEMENT = "D:\\XXX\\"
DATA_DIR = DATA_DIR_PORTABLE


df, begin_offset, anLength = get_first_items(1, DATA_DIR + 'data.h5')


df.shape

x = df.mjd.values

x = x - x.min()

p = df.passband.values

y = df.flux.values

y_read_out = np.empty(y.shape, dtype = np.uint16)

num_bins_y = 100

for b in range(6):
    m = (p == b)
    y_read_out[m] = get_binned_sequence(y[m], num_bins_y)

"""c"""


slot_size = 5

sentenceLength= 200

num_bins_x = int (.5 + x.max()/slot_size)

bins = np.linspace(0,  x.max(), num_bins_x, endpoint = False)

# Fixed bin size

digitized = np.digitize(x, bins)
digitized = digitized - 1


out = np.empty((6, num_bins_x), dtype = np.float32)

out[:, :] = np.nan

for b in range (6):

    m = (p == b)

    y_p = y_read_out[np.where(m)]
    d_p = digitized[np.where(m)]

    out[b, d_p] = y_p

    
out[0, :] *= 1
out[1, :] += num_bins_y
out[2, :] += (2 * num_bins_y)
out[3, :] += (num_bins_y * 3)
out[4, :] += (num_bins_y * 4)
out[5, :] += (num_bins_y * 5)

out += 1

out = np.nan_to_num(out)

out = out.flatten(order = 'F')

res = np.zeros(6 * sentenceLength, dtype = np.int16)

res[:out.shape[0]] = out

res.shape

# Look up in embedding

# => Flat 1200 floats
#
#
# Group to six dimensional neurons.
# 6D into convolution 1D.
# Dense 
# Dense
# 
# Compare with trained embedding.

encoder_inputs = Input(shape=(sentenceLength,), name="Encoder_input")
target_inputs = Input(shape=(sentenceLength,), name="target_input")

vocab_size = np.max(res) + 1

emb_obj = Embedding(output_dim=6, input_dim=vocab_size, name="Embedding", embeddings_constraint=unitnorm(axis=1), trainable=True)

x = emb_obj (encoder_inputs)








