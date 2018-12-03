


import numpy as np
import pandas as pd
import gc
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Subtract
from keras.layers import LSTM
from keras.layers import TimeDistributed


from keras.models import Model

from keras.constraints import unitnorm

import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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

slot_size = 5
num_sequence_length = 200
num_bins_y = 100

# Load all:

df, begin_offset, anLength = get_first_items(0, DATA_DIR + 'data.h5')

num_objects = np.unique(df.object_id).shape[0]

# Check num objects with meta:

df_meta = pd.read_csv(DATA_DIR + "training_set_metadata.csv")
df_meta_t = pd.read_csv(DATA_DIR + "test_set_metadata.csv")

assert df_meta.shape[0] + df_meta_t.shape[0] == num_objects


x_all = df.mjd.values

x_all = x_all - x_all.min()

p_all = df.passband.values

y_all = df.flux.values

y_read_out = np.empty(y_all.shape, dtype = np.uint16)
y_std_scaled = np.empty(y_all.shape, dtype = np.float32)

s = StandardScaler(with_std=False)


for b in range(6):
    m = (p_all == b)
    y_read_out[m] = get_binned_sequence(y_all[m], num_bins_y)
    y_std_scaled[m] = s.fit_transform(y_all[m].reshape(-1, 1)).flatten()

"""c"""

anData = np.zeros((num_objects,6 * num_sequence_length), dtype = np.uint16)   
anData_d = np.zeros((num_objects, 6 * num_sequence_length), dtype = np.float32)

res_size = np.zeros(num_objects, dtype = np.uint16)

for i in range(num_objects):

    if i % 10000 == 0:
        print(f"Row {i} / {num_objects}")

    offset = begin_offset[i]
    length = anLength[i]

    x = x_all[offset: offset + length].copy()

    x = x - x.min()

    p = p_all[offset: offset + length]

    y_this_read_out = y_read_out[offset: offset + length]

    y_this_scaled = y_std_scaled[offset: offset + length]

    num_bins_x = int (.5 + x.max()/slot_size)

    bins = np.linspace(0,  x.max(), num_bins_x, endpoint = False)

    digitized = np.digitize(x, bins)
    digitized = digitized - 1

    out = np.empty((6, num_bins_x), dtype = np.float32)
    out_d = np.empty((6, num_bins_x), dtype = np.float32)

    out[:, :] = np.nan
    out_d[:, :] = np.nan

    for b in range (6):

        m = (p == b)

        y_p = y_read_out[np.where(m)]
        y_d = y_std_scaled[np.where(m)]
        d_p = digitized[np.where(m)]

        out[b, d_p] = y_p
        out_d[b, d_p] = y_d

    """c"""

    out[1, :] += num_bins_y
    out[2, :] += (2 * num_bins_y)
    out[3, :] += (num_bins_y * 3)
    out[4, :] += (num_bins_y * 4)
    out[5, :] += (num_bins_y * 5)





    out += 1

    out = np.nan_to_num(out)
    out_d = np.nan_to_num(out_d)

    out = out.flatten(order = 'F')
    out_d = out_d.flatten(order = 'F')

    res_size[i] = out.shape[0]


    res = np.zeros(6 * num_sequence_length, dtype = np.int16)
    res_d = np.zeros(6 * num_sequence_length, dtype = np.float32)

    iCopyElements = np.min([res.shape[0], out.shape[0]])

    res[:iCopyElements] = out[:iCopyElements]
    res[iCopyElements:] = 0

    res_d[:iCopyElements] = out_d[:iCopyElements]
    res_d[iCopyElements:] = 0

    anData[i, 0:res.shape[0]] = res
    anData_d[i, 0:res.shape[0]] = res_d

"""c"""

np.save(DATA_DIR + "anData_all_2", anData)
np.save(DATA_DIR + "anData_d_all_2", anData_d)




### LOAD


import numpy as np
import pandas as pd
import gc
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Flatten
from keras import optimizers
from keras.layers import Subtract
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Dropout

from keras.models import Model

from keras.constraints import unitnorm

import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import time


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

slot_size = 5
num_sequence_length = 200

anData = np.load(DATA_DIR + "anData_all_2.npy")
anData_d = np.load(DATA_DIR + "anData_d_all_2.npy")


num_objects = anData.shape[0]

encoder_inputs = Input(shape=(num_sequence_length * 6,), name="Encoder_input")

vocab_size = np.max(anData) + 1

emb_dim = 6

emb_obj = Embedding(output_dim=emb_dim, input_dim=vocab_size, name="Embedding", trainable=True)

x = emb_obj (encoder_inputs)

x = Flatten() (x)

x = Dense(num_sequence_length * 6) (x)

x = Dense(num_sequence_length * 6) (x)

model = Model(encoder_inputs, x)

optimizer = optimizers.Adam()

model.compile(loss='mse', optimizer=optimizer)


max_commit = 10000


num_folds = 9
lRunFolds = list (range(num_folds))

kf = KFold(n_splits=num_folds, shuffle=True, random_state=22)

lKF = list (enumerate(kf.split(anData)))

lMSE = []

iFold = 0

iLoop, (train_index, test_index) = lKF[iFold]

anDataTrain = anData[train_index]
anDataValid = anData[test_index]

arYTrain = anData_d[train_index]
arYValid = anData_d[test_index]

n_commit_split = 1 + int (num_objects / max_commit)

idxTrain = np.arange(0, anDataTrain.shape[0])
idxValid = np.arange(0, anDataValid.shape[0])

idxTrainSplit = np.array_split(idxTrain, n_commit_split, axis = 0)
idxValidSplit = np.array_split(idxValid, n_commit_split, axis = 0)

for ix in range(4, n_commit_split):
    print(f"ix = {ix} of {n_commit_split}")

    num_train = anDataTrain[idxTrainSplit[ix]].shape[0]
    num_valid = anDataValid[idxValidSplit[ix]].shape[0]

    h = model.fit(batch_size=16,
                  x = anDataTrain[idxTrainSplit[ix]],
                  y = arYTrain[idxTrainSplit[ix]],
                  validation_data = (anDataValid[idxValidSplit[ix]], arYValid[idxValidSplit[ix]]),
                  epochs = 3,
                  verbose = 1)
    """c"""           
    
    z_p = model.predict(anDataValid[idxValidSplit[ix]])

    y_true = arYValid[idxValidSplit[ix]]
    
    mse = mean_squared_error(z_p, y_true)
    print(f"Valid set MSE = {mse:.4f}")

    time.sleep(10)
"""c"""



for ix in range(n_commit_split):
    print(f"ix = {ix} of {n_commit_split}")

    num_train = anDataTrain[idxTrainSplit[ix]].shape[0]
    num_valid = anDataValid[idxValidSplit[ix]].shape[0]

    z_p = model.predict(anDataValid[idxValidSplit[ix]])

    y_true = arYValid[idxValidSplit[ix]]
    
    mse = mean_squared_error(z_p, y_true)
    print(f"Valid set MSE = {mse:.4f}")

"""c"""


# Get output from inner layers

# Retrieve end store weights, 
# Setup a prediction engine for use offline.

# Predict one/few and predict the full dataset incl training.

# Run nn and light gbm on this dataset.


# Last reported MSE = 0.01111

# last are 0.0111 0.0092 0.0106 0.0118 0.0125 0.0109


for layer in model.layers: print(layer.get_config(), layer.get_weights())


for idx, layer in enumerate(model.layers): print(idx, layer.name)


type (model.layers)

emb = model.layers[1]

d64 = model.layers[2]


# The embedding matrix

len (emb.get_weights())

emb_w = emb.get_weights()[0]

emb_w.shape

# D64

d64_w0 = d64.get_weights()[0]
d64_w1 = d64.get_weights()[1]


d64_w0.shape

d64_w1.shape


model.predict()
