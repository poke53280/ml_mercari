

import numpy as np
import pandas as pd
import pathlib

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model


p = pathlib.Path(r'C:\Users\T149900\Downloads\dfdc_train_part_21\dfdc_train_part_21')


anDataReal = np.load(p / "real_data_000.npy")
anDataFake = np.load(p / "fake_data.npy")

print (f"anDataReal {anDataReal.shape}")
print (f"anDataFake {anDataFake.shape}")

sequence = anDataReal


np.random.shuffle(sequence)

num_train = int (0.7 * sequence.shape[0])
num_test = sequence.shape[0] - num_train


test_sequence = sequence[num_train:num_train + num_test]
test_sequence = test_sequence.reshape((test_sequence.shape[0], test_sequence.shape[1], 3))


sequence = sequence[:num_train]

num_samples = sequence.shape[0]
num_timesteps = sequence.shape[1]


# reshape input into [samples, timesteps, features]
sequence = sequence.reshape((num_samples, num_timesteps, 3))



# define model
model = Sequential()
model.add(LSTM(2048, activation='relu', return_sequences=True, input_shape=(num_timesteps, 3)))
model.add(LSTM(512, activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(LSTM(8, activation='relu'))

model.add(RepeatVector(num_timesteps))

model.add(LSTM(8, activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(LSTM(512, activation='relu', return_sequences=True))
model.add(LSTM(2048, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(3)))
model.compile(optimizer='adam', loss='mse')



# fit model
model.fit(sequence, sequence, epochs=2, verbose=1)




y_test = model.predict(test_sequence)

from sklearn.metrics import mean_squared_error

y_test = y_test.reshape(-1)
test_sequence = test_sequence.reshape(-1)

data_mse = mean_squared_error(y_test, test_sequence)

y_random = np.random.uniform(size = test_sequence.shape)

y_random = y_random.reshape((num_test, 16, 3))

y_random_predict = model.predict(y_random)

y_random_predict = y_random_predict.reshape(-1)
y_random = y_random.reshape(-1)

ran_mse = mean_squared_error(y_random, y_random_predict)

data_mse = data_mse * 1000
ran_mse =ran_mse * 1000

data_mse
ran_mse



