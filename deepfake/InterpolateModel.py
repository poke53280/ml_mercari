
from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd


import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

from keras.optimizers import SGD

from keras.initializers import Constant

from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
#from DataLoader import load_clives
#from DataLoader import get_clive_files

from mp4_frames import get_log_dir

from mp4_frames import get_ready_data_dir
from mp4_frames import get_model_dir

import matplotlib.pyplot as plt
import operator
import random

import numpy as np

import datetime

from sklearn.preprocessing import StandardScaler

from multiprocessing import Pool
from pathlib import Path



class SwapNoise:

    _var = 0

    def __init__(self):
        pass

    def add_swap_noise(self, X_batch, X_clean, p, verbose):

        nNumRowsBatch = X_batch.shape[0]
        nNumRowsSource = X_clean.shape[0]

        if verbose == 1:
            print(f"Adding {p * 100.0}% noise to {nNumRowsBatch} row(s) from noise pool of {nNumRowsSource} row(s).")
            print(f"   Creating noise source indices")

        aiNoiseIndex = np.random.randint(nNumRowsSource, size=nNumRowsBatch)
        aiNoiseIndex = np.sort(aiNoiseIndex)

        if verbose == 1:
            print(f"   Allocating noise source")
        
        X_noise = X_clean[aiNoiseIndex]

        if verbose == 1:
            print(f"   Allocating noise mask")

        X_mask = np.random.rand(X_batch.shape[0], X_batch.shape[1])

        if verbose == 1:
            print(f"   Applying noise")
        
        m = X_mask < p

        X_batch[m] = X_noise[m]

        return X_batch
"""c"""

####################################################################################
#
#   get_model_dense
#

def get_model_dense_small(num_n):

    num_timesteps = 100
    model = Sequential()
    model.add(Dense(num_n, activation='linear', kernel_initializer = Constant(value=0.5), bias_initializer='zeros', input_shape=(3 * num_timesteps, )))
   
    model.compile(optimizer='adam', loss='mse')

    # model.summary()
    return model



def get_model_dense_large(num_n):

    num_timesteps = 100
    model = Sequential()
    model.add(Dense(3 * num_timesteps, activation='linear', kernel_initializer = Constant(value=0.0), bias_initializer='zeros', input_shape=(3 * num_timesteps, )))
    model.add(Dense(3 * num_timesteps, activation='linear', kernel_initializer = Constant(value=0.0), bias_initializer='zeros'))
    model.add(Dense(3 * num_timesteps, activation='linear', kernel_initializer = Constant(value=0.0), bias_initializer='zeros'))

    model.add(Dense(num_n, activation='linear', kernel_initializer = Constant(value=0.0), bias_initializer='zeros'))
    model.add(Dense(3 * num_timesteps, activation='linear', kernel_initializer = Constant(value=0.0), bias_initializer='zeros'))

    model.compile(optimizer='adam', loss='mse')

    model.summary()
    return model




def get_model_lstm_small(num_timesteps):
    model = Sequential()
    model.add(LSTM(100, activation='relu', kernel_initializer='zeros', bias_initializer='zeros', input_shape=(num_timesteps, 3)))

    model.add(RepeatVector(num_timesteps))

    model.add(LSTM(100, activation='relu', return_sequences=True, kernel_initializer='zeros', bias_initializer='zeros'))

    model.add(TimeDistributed(Dense(3, kernel_initializer='zeros', bias_initializer='zeros')))
    model.compile(optimizer='adam', loss='mse')

    model.summary()
    return model

def get_model_lstm(num_timesteps, bottleneck, complexity):


    model = Sequential()
    model.add(LSTM(2 *128 * complexity, activation='relu', return_sequences=True, kernel_initializer='zeros', bias_initializer='zeros', input_shape=(num_timesteps, 3)))
    model.add(LSTM(128 * complexity, activation='relu', return_sequences=True, kernel_initializer='zeros', bias_initializer='zeros'))
    model.add(LSTM(bottleneck, activation='relu', kernel_initializer='zeros', bias_initializer='zeros'))

    model.add(RepeatVector(num_timesteps))

    model.add(LSTM(bottleneck, activation='relu', return_sequences=True, kernel_initializer='zeros', bias_initializer='zeros'))
    model.add(LSTM(128 * complexity, activation='relu', return_sequences=True, kernel_initializer='zeros', bias_initializer='zeros'))
    model.add(LSTM(2 *128 * complexity, activation='relu', return_sequences=True, kernel_initializer='zeros', bias_initializer='zeros'))

    model.add(TimeDistributed(Dense(3, kernel_initializer='zeros', bias_initializer='zeros')))
    model.compile(optimizer='adam', loss='mse')

    model.summary()
    return model



def reconstruction_error(model, data):

    data_p = model.predict(data)
    mse = mean_squared_error(data_p.reshape(-1), data.reshape(-1))
    return mse





def train_model(data):

    num_epochs = 10
    
    data = data.reshape(-1, 300)

    rNoise = 0.15
    num_test = int(0.3 * data.shape[0])

    model = get_model_dense_small(300)

    l_weights = model.get_weights()

    weights0 = l_weights[0]

    weights0 = np.identity(300)

    l_weights = [weights0, l_weights[1]]

    model.set_weights(l_weights)


    for i in range(num_epochs):

        X = data[num_test:].copy()
        Y = data[num_test:]

        s = SwapNoise()

        X = s.add_swap_noise(X, X, rNoise, False)

        model.fit(X, Y, epochs=2, batch_size=128, verbose=0)


    data_p = model.predict(data[:num_test])
    mse = mean_squared_error(data[:num_test].reshape(-1), data_p.reshape(-1))

    return mse, model



def process_videoset(iCluster, original):
    input_dir = get_ready_data_dir()
    output_dir = get_model_dir()
    
    
    input_df = input_dir / f"c_{iCluster}_{original}.pkl"
    input_npy = input_dir / f"c_{iCluster}_{original}.npy"


    isInputExisting = input_df.is_file() and input_npy.is_file()

    if not isInputExisting:
        # print (f"Missing input for {iCluster}_{original}")
        return

    output_model_real = output_dir / f"c_{iCluster}_{original}_real.h5"
    output_model_fake = output_dir / f"c_{iCluster}_{original}_fake.h5"

    isOutputExisting = output_model_real.is_file() and output_model_fake.is_file()

    if isOutputExisting:
        print (f"{iCluster}_{original} already created")
        return

    print(f"Processing c_{iCluster}_{original}...")

    df = pd.read_pickle(input_df)

    data = np.load(input_npy)

    m_fake = (df.fake == True)
    m_real = (df.fake == False)


    mse_fake, model_fake = train_model(data[m_fake])
    mse_real, model_real = train_model(data[m_real])

    print(f"c_{iCluster}_{original}: mse_fake {mse_fake} mse_real {mse_real}")

    model_fake.save(output_model_fake)
    model_real.save(output_model_real)



####################################################################################
#
#   __main__
#

