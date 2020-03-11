



import pandas as pd

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from DataLoader import load_clives
from DataLoader import get_clive_files

import random

import numpy as np



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

def get_model(num_timesteps):


    model = Sequential()
    model.add(LSTM(256, activation='relu', kernel_initializer='zeros',bias_initializer='zeros', return_sequences=True, input_shape=(num_timesteps, 3)))
    model.add(LSTM(128, activation='relu', kernel_initializer='zeros',bias_initializer='zeros',return_sequences=True))
    model.add(LSTM(12, activation='relu', kernel_initializer='zeros',bias_initializer='zeros'))

    model.add(RepeatVector(num_timesteps))

    model.add(LSTM(12, activation='relu', kernel_initializer='zeros',bias_initializer='zeros', return_sequences=True))
    model.add(LSTM(128, activation='relu', kernel_initializer='zeros',bias_initializer='zeros', return_sequences=True))
    model.add(LSTM(256, activation='relu', kernel_initializer='zeros',bias_initializer='zeros', return_sequences=True))

    model.add(TimeDistributed(Dense(3)))
    model.compile(optimizer='adam', loss='mse')

    model.summary()
    return model

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

model = get_model(num_timesteps)


l_files = get_clive_files()

df_valid = load_clives(l_files[:4])

anValidTest = np.stack(df_valid.test.values)
anValidTest = preprocess_input(anValidTest)

anValidReal = np.stack(df_valid.real.values)
anValidReal = preprocess_input(anValidReal)


l_load_files = l_files[4:]



s = SwapNoise()

for x in range (40):

    l_train_files = random.sample(l_load_files, 17)

    print(f"Training on {l_train_files}")

    df = load_clives(l_train_files)

    anData = np.stack(df.test.values)

    anData = preprocess_input(anData)

    np.random.shuffle(anData)


    y_train = anData.copy()
    X_train = anData.copy()

    X_train = s.add_swap_noise(X_train, y_train, 0.1, True)

    model.fit(X_train, y_train, epochs=1, batch_size=64, verbose=1)


    p_test = reconstruction_error(model, anValidTest)
    p_real = reconstruction_error(model, anValidReal)


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
