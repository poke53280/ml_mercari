
from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from DataLoader import load_clives
from DataLoader import get_clive_files

import random

import numpy as np

from Logger import Logger



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
#   get_model
#


####################################################################################
#
#   get_model
#

def get_model(num_timesteps, bottleneck):

    encoder_input = keras.Input(shape=(num_timesteps, 3))

    x = layers.Bidirectional(layers.LSTM(256, activation='relu', kernel_initializer='zeros',bias_initializer='zeros', return_sequences=True))(encoder_input)

    x = layers.Bidirectional(layers.LSTM(128, activation='relu', kernel_initializer='zeros',bias_initializer='zeros', return_sequences=True)) (x)

    encoder_output = layers.Bidirectional(layers.LSTM(bottleneck, activation='relu', kernel_initializer='zeros',bias_initializer='zeros',)) (x)

    encoder = keras.Model(inputs=encoder_input, outputs=encoder_output, name='encoder')

    encoder.summary()

    x = layers.RepeatVector(num_timesteps)(encoder_output)

    x = layers.Bidirectional(layers.LSTM(bottleneck, activation='relu', kernel_initializer='zeros',bias_initializer='zeros', return_sequences=True)) (x)

    x = layers.Bidirectional(layers.LSTM(128, activation='relu', kernel_initializer='zeros',bias_initializer='zeros', return_sequences=True)) (x)

    x = layers.Bidirectional(layers.LSTM(256, activation='relu', kernel_initializer='zeros',bias_initializer='zeros', return_sequences=True)) (x)

    decoder_output = layers.TimeDistributed(layers.Dense(3, kernel_initializer='zeros',bias_initializer='zeros')) (x)

    autoencoder  = keras.Model(inputs=encoder_input, outputs=decoder_output, name='autoencoder')

    autoencoder.summary()

    return autoencoder



def preprocess_input(data):
    data = (data.astype(np.float32) - 256/2.0) / (256/2.0)
    return data


def reconstruction_error(model, data):
        data_p = model.predict(data)
        data_p = data_p.reshape(data_p.shape[0], -1)

        data = data.reshape(data_p.shape[0], -1)

        mse = np.sum((data_p-data)**2,axis=1)

        return mse



num_timesteps = 50

l = Logger("bottlenect_test")

for bottleneck in [5, 9, 12, 15, 22]:

    l.write(f"bottleneck {bottleneck}")


    model = get_model(num_timesteps, bottleneck)


    l_files = get_clive_files()

    df_valid = load_clives(l_files[:1])

    anValidTest = np.stack(df_valid.test.values)
    anValidTest = preprocess_input(anValidTest)

    anValidReal = np.stack(df_valid.real.values)
    anValidReal = preprocess_input(anValidReal)


    l_load_files = l_files[8:]



    s = SwapNoise()

    for x in range (3):

        l.write(f"x = {x}")

        l_train_files = random.sample(l_load_files, 27)

        # print(f"Training on {l_train_files}")

        df = load_clives(l_train_files)

        anData = np.stack(df.test.values)

        anData = anData[:300]

        anData = preprocess_input(anData)

        np.random.shuffle(anData)


        y_train = anData.copy()
        X_train = anData.copy()

        X_train = s.add_swap_noise(X_train, y_train, 0.1, True)

        model.fit(X_train, y_train, epochs=1, batch_size=64, validation_data = (anValidTest, anValidTest), verbose=1)


        p_test = reconstruction_error(model, anValidTest)
        p_real = reconstruction_error(model, anValidReal)

        l.write(f"p_test mean {p_test.mean()}")
        l.write(f"p_real mean {p_real.mean()}")

        df_valid = df_valid.assign(err_test = p_test, err_real = p_real)

        g = df_valid.groupby('file')


        s_test = g.err_test.mean()

        s_test_name = list(s_test.index)
        s_test_name = [x + "_test" for x in s_test_name]
        s_test.index = s_test_name


        s_real = g.err_real.mean()
    
        s_real_name = list(s_real.index)
        s_real_name = [x + "_real" for x in s_real_name]
        s_real.index = s_real_name
    
        df_res = pd.concat([s_test, s_real])

        df_res = df_res.sort_values()

        m_test = ["_test" in x for x in list(df_res.index)]

        df_res = pd.DataFrame({'err': df_res, 'm_test': m_test})

        l_predict = [True] * (df_res.shape[0] // 2)
        l_predict.extend ([False] * (df_res.shape[0] // 2))

        df_res = df_res.assign(m_predict = l_predict)
    

        roc_auc = roc_auc_score(df_res.m_test, df_res.m_predict)

        gini = 2 * roc_auc - 1

        print(f"GINI {gini:.3}")
        l.write(f"GINI {gini:.3}")
