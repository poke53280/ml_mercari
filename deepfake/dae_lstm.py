
#
#
#   https://machinelearningmastery.com/lstm-autoencoders/
#
#


# lstm autoencoder recreate sequence
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model

import numpy as np

num_train = 1000000 - 300000
num_test = 300000


10 mill loss 0.0021  32 - 32    two hours(?)

500,000   loss: 0.0064   1024 - 1024   5 hours (?)

sequenceR = np.load("C:\\Users\\T149900\\source\\repos\\sm_data\\sm_data\\data_16.npy")

sequenceR = sequenceR[:num_train + num_test]

sequenceG = sequenceR.copy()
sequenceB = sequenceR.copy()


sequence = np.stack([sequenceR, sequenceG, sequenceB], axis = -1)

del sequenceR
del sequenceG
del sequenceB

np.random.shuffle(sequence)


test_sequence = sequence[num_train:num_train + num_test]
test_sequence = test_sequence.reshape((test_sequence.shape[0], test_sequence.shape[1], 3))


sequence = sequence[:num_train]

num_samples = sequence.shape[0]
num_timesteps = sequence.shape[1]


# reshape input into [samples, timesteps, features]
sequence = sequence.reshape((num_samples, num_timesteps, 3))



# define model
model = Sequential()


model.add(LSTM(1024, activation='relu', input_shape=(num_timesteps, 3)))
model.add(RepeatVector(num_timesteps))
model.add(LSTM(1024, activation='relu', return_sequences=True))

#model.add(Dense(512))
#model.add(Dense(128))

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

