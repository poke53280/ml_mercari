

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

def train(trainpath, testpath, nLimit):
    
    m_dir = get_model_dir()

    anTrain = np.load(trainpath)
    anTest = np.load(testpath)

    np.random.shuffle(anTrain)

    if nLimit > 0:
        anTrain = anTrain[:nLimit]
        anTest = anTest[:nLimit]

    anTrain = preprocess_input(anTrain)
    anTest = preprocess_input(anTest)


    train_real = anTrain[:, :32, :]
    train_fake = anTrain[:, 32:, :]

    test_real = anTest[:, :32, :]
    test_fake = anTest[:, 32:, :]


    num_train = anTrain.shape[0]
    num_test = anTest.shape[0]

    num_timesteps = train_real.shape[1]

    model = get_model(num_timesteps)
    
    for iEpoch in range(6):
        # Todo reshuffle anTrain for each epoch.
        # Todo provide new real /fake<n> set for each epoch.

        model.fit(train_real, train_real, epochs=1, batch_size=256, verbose=1)

        data_p = model.predict(test_real)
        rms = mean_squared_error(data_p.reshape(-1), test_real.reshape(-1))
        print(f"Reconstuction error rms = {rms}")
   
    model.save(m_dir / f"model_{trainpath.stem}_{testpath.stem}_rr.h5")
   

    model = get_model(num_timesteps)

    for iEpoch in range(6):

        model.fit(train_fake, train_real, epochs=1, batch_size=256, verbose=1)
        data_p = model.predict(test_fake)
        rms = mean_squared_error(data_p.reshape(-1), test_real.reshape(-1))
        print(f"Reconstuction error rms = {rms}")
    
    model.save(m_dir/ f"model_{trainpath.stem}_{testpath.stem}_fr.h5")

    model = get_model(num_timesteps)

    for iEpoch in range(6):
        model.fit(train_fake, train_fake, epochs=1, batch_size=256, verbose=1)
        data_p = model.predict(test_fake)
        rms = mean_squared_error(data_p.reshape(-1), test_fake.reshape(-1))
        print(f"Reconstuction error rms = {rms}")
    
    model.save(m_dir / f"model_{trainpath.stem}_{testpath.stem}_ff.h5")

  


#################################################################################
#
#   main_get_art_arg
#

def main_get_art_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-tr", help="train pair feature dataset", required = True)
    parser.add_argument("--test", "-te",  help="test pair feature dataset", required = True)

    parser.add_argument("--limit", "-l", help="data cap. 0: no limit", required = True)

    args = parser.parse_args()

    zTrainfile = args.train
    zTestfile = args.test

    input_dir = get_ready_data_dir()

    trainpath = input_dir / (zTrainfile + ".npy")
    assert trainpath.is_file(), f"{str(trainpath)} is not a file"
   
    testpath = input_dir / (zTestfile + ".npy")
    assert testpath.is_file(), f"{str(testpath)} is not a file"

    nLimit = int(args.limit)

    return trainpath, testpath, nLimit


#################################################################################
#
#   __main__
#


if __name__ == '__main__':
    trainpath, testpath, nLimit = main_get_art_arg()
    train(trainpath, testpath, nLimit)




