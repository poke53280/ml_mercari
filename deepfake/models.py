

import numpy as np
import pandas as pd
import pathlib

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
from sklearn.metrics import mean_squared_error



####################################################################################
#
#   reconstruction_error
#

def reconstruction_error(model, data):
    data_p = model.predict(data)
    rms = mean_squared_error(data_p.reshape(-1), data.reshape(-1))
    return rms



####################################################################################
#
#   preprocess_input
#

def preprocess_input(data):
    data = (data.astype(np.float32) - 256/2.0) / (256/2.0)
    return data



p = pathlib.Path(f"C:\\Users\\T149900\\vid_out")
assert p.is_dir()

l_datafiles = []

for x in p.iterdir():
    l_datafiles.append(x)


l_data = []

for x in l_datafiles:
    data = np.load(x)
    l_data.append(data)


anData = np.vstack(l_data)


anPart = anData[:, 0]
anViLo = anData[:, 1]
anVidHi = anData[:, 2]

anReal = anData[:, 3:3 + 16 * 3]
anFake = anData[:, 3 + 16 * 3 : ]

anReal = anReal.reshape(-1, 16, 3)
anFake = anFake.reshape(-1, 16, 3)

v_id = anViLo + 256 * anVidHi.astype(np.int32)
id = v_id * 50 + anPart

unique_id = np.unique(id)

seq_id = np.searchsorted(unique_id, id)

# Train and test

m = seq_id % 10 < 1

m_desc(m)

sequence = anReal[~m]
test_sequence_real = anReal[m]
test_sequence_fake = anFake[m]

sequence = preprocess_input(sequence)
test_sequence_real = preprocess_input(test_sequence_real)
test_sequence_fake = preprocess_input(test_sequence_fake)

np.random.shuffle(sequence)

num_timesteps = sequence.shape[1]


# define model
model = Sequential()
# model.add(LSTM(2048, activation='relu', return_sequences=True, input_shape=(num_timesteps, 3)))
model.add(LSTM(512, activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(LSTM(8, activation='relu'))

model.add(RepeatVector(num_timesteps))

model.add(LSTM(8, activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(LSTM(512, activation='relu', return_sequences=True))
# model.add(LSTM(2048, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(3)))
model.compile(optimizer='adam', loss='mse')



# fit model
model.fit(sequence, sequence, epochs=2, verbose=1)

# create random sequence as baseline
y_random = np.random.uniform(size = test_sequence_real.shape)
y_random = y_random.reshape((-1, 16, 3))

reconstruction_error(model, sequence)
reconstruction_error(model, y_random)
reconstruction_error(model, test_sequence_real)
reconstruction_error(model, test_sequence_fake)






