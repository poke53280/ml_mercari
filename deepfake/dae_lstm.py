

from numpy import array
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import mean_squared_error

from mp4_frames import get_output_dir
import datetime


####################################################################################
#
#   preprocess_input
#

def preprocess_input(data):
    data = (data.astype(np.float32) - 256/2.0) / (256/2.0)
    return data


####################################################################################
#
#   reconstruction_error
#

def reconstruction_error(model, data):
    data_p = model.predict(data)
    rms = mean_squared_error(data_p.reshape(-1), data.reshape(-1))
    return rms



def train():

    input_dir = get_output_dir()

    anTrain = np.load(input_dir / "train_c_nose.npy")

    np.random.shuffle(anTrain)

    anTrain = preprocess_input(anTrain)
  

    # Real part only

    sequence_real = anTrain[:, :16, :]
    sequence_fake = anTrain[:, 16:, :]
    


    num_samples = sequence_real.shape[0]
    num_timesteps = sequence_real.shape[1]


    model = Sequential()
    model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(num_timesteps, 3)))
    model.add(LSTM(32, activation='relu', return_sequences=True))
    model.add(LSTM(4, activation='relu'))

    model.add(RepeatVector(num_timesteps))

    model.add(LSTM(4, activation='relu', return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=True))

    model.add(TimeDistributed(Dense(3)))
    model.compile(optimizer='adam', loss='mse')

    model.summary()


    p = get_output_dir()

    # fit models
    # 0.0165
    model.fit(sequence_real, sequence_real, epochs=1, batch_size=256, verbose=1)
    model.save(p / 'my_model_rr.h5')

    model.fit(sequence_real[:2000000], sequence_fake[:2000000], epochs=1, batch_size=256, verbose=1)
    model.save(p / 'my_model_rf.h5')


    model = Sequential()
    model.add(LSTM(256, activation='relu', return_sequences=True, input_shape=(num_timesteps, 3)))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(LSTM(4, activation='relu'))

    model.add(RepeatVector(num_timesteps))

    model.add(LSTM(4, activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(LSTM(256, activation='relu', return_sequences=True))

    model.add(TimeDistributed(Dense(3)))
    model.compile(optimizer='adam', loss='mse')

    model.summary()


    model.fit(sequence_fake[:2000000], sequence_real[:2000000], epochs=1, batch_size=256, verbose=1)
    model.save(p / 'my_model_fr.h5')

    # 0.0124
    model.fit(sequence_fake[:2000000], sequence_fake[:2000000], epochs=1, batch_size=256, verbose=1)
    model.save(p / 'my_model_ff.h5')













