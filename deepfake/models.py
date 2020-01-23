

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
    data = (data - 256/2) / (256/2)
    return data.astype(np.uint8)


def load_dataset(p, zLabel, num):
    assert p.is_dir()

    l_real = []
    l_fake = []

    for x in range (num):
        anDataReal = np.load(p / f"{zLabel}_{x:03}.npy")
        anDataFake = np.load(p / f"{zLabel}_{x:03}.npy")
        l_real.append(anDataReal)
        l_fake.append(anDataFake)

    anDataReal = np.vstack(l_real)
    anDataFake = np.vstack(l_fake)

    return (anDataReal, anDataFake)


p = pathlib.Path(r'C:\Users\T149900\Downloads\dfdc_train_part_00\dfdc_train_part_0')
assert p.is_dir()

anDataReal, anDataFake = load_dataset("") # 




sequence_fake = preprocess_input(anDataFake)





sequence = (anDataReal - 256/2) / (256/2)

sequence = sequence.astype(np.float32)

np.random.shuffle(sequence)

# Remove split here. Use split above, else videos are seen leakage.
num_train = int (0.7 * sequence.shape[0])
num_test = sequence.shape[0] - num_train


test_sequence = sequence[num_train:num_train + num_test]
test_sequence = test_sequence.reshape((test_sequence.shape[0], test_sequence.shape[1], 3))


sequence = sequence[:num_train]

num_samples = sequence.shape[0]
num_timesteps = sequence.shape[1]


# reshape input into [samples, timesteps, features]
sequence = sequence.reshape((-1, num_timesteps, 3))



# define model
model = Sequential()
#model.add(LSTM(2048, activation='relu', return_sequences=True, input_shape=(num_timesteps, 3)))
#model.add(LSTM(512, activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(LSTM(8, activation='relu'))

model.add(RepeatVector(num_timesteps))

model.add(LSTM(8, activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=True))
#model.add(LSTM(512, activation='relu', return_sequences=True))
#model.add(LSTM(2048, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(3)))
model.compile(optimizer='adam', loss='mse')



# fit model
model.fit(sequence, sequence, epochs=2, verbose=1)

# create random sequence as baseline
y_random = np.random.uniform(size = test_sequence.shape)
y_random = y_random.reshape((-1, 16, 3))

rand_mse = reconstruction_error(model, y_random)
data_mse = reconstruction_error(model, test_sequence)
fake_mse = reconstruction_error(model, sequence_fake)



