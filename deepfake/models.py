

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model


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
