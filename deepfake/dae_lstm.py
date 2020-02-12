

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

from mp4_frames import get_ready_data_dir
from mp4_frames import get_model_dir
import datetime

import argparse

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





####################################################################################
#
#   get_model
#

def get_model(num_timesteps):


    model = Sequential()
    model.add(LSTM(256, activation='relu', return_sequences=True, input_shape=(num_timesteps, 3)))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(LSTM(12, activation='relu'))

    model.add(RepeatVector(num_timesteps))

    model.add(LSTM(12, activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(LSTM(256, activation='relu', return_sequences=True))

    model.add(TimeDistributed(Dense(3)))
    model.compile(optimizer='adam', loss='mse')

    model.summary()
    return model


####################################################################################
#
#   train
#

def train(zFeature, nLimit):

    input_dir = get_ready_data_dir()

    m_dir = get_model_dir()


    anTrain = np.load(input_dir / f"train_{zFeature}.npy")

    np.random.shuffle(anTrain)

    if nLimit > 0:
        anTrain = anTrain[:nLimit]

    anTrain = preprocess_input(anTrain)

    sequence_real = anTrain[:, :32, :]
    sequence_fake = anTrain[:, 32:, :]
  

    num_samples = anTrain.shape[0]

    num_train = int (0.9 * num_samples)
    num_test = num_samples - num_train

    num_timesteps = sequence_real.shape[1]

    model = get_model(num_timesteps)
    
    for iEpoch in range(5):
        model.fit(sequence_real[:num_train], sequence_real[:num_train], epochs=1, batch_size=256, verbose=1)
        rms = reconstruction_error(model, sequence_real[num_train:])
        print(f"Reconstuction error rms = {rms}")

    
    model.save(m_dir / f"model_{zFeature}_rr.h5")
   

    model = get_model(num_timesteps)
    model.fit(sequence_fake, sequence_real, epochs=5, batch_size=256, verbose=1)
    model.save(m_dir/ f"model_{zFeature}_fr.h5")


    model = get_model(num_timesteps)
    model.fit(sequence_fake[:num_train], sequence_fake[:num_train], epochs=5, batch_size=256, verbose=1)
    model.save(m_dir / f"model_{zFeature}_ff.h5")

  


#################################################################################
#
#   main_get_art_arg
#

def main_get_art_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", "-f", help="facial feature dataset", required = True)
    parser.add_argument("--limit", "-l", help="data cap. 0: no limit", required = True)

    args = parser.parse_args()

    zFeature = args.feature

    nLimit = int(args.limit)

    print(zFeature)

    return zFeature, nLimit


#################################################################################
#
#   __main__
#


if __name__ == '__main__':
    zFeature, nLimit = main_get_art_arg()
    train(zFeature, nLimit)




