

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

def train():

    input_dir = get_ready_data_dir()

    # Todo: Test and predict as far as possible on unseen faces and scenes.
    # Here, most frames are seen in train and memorization can produce artifically good results.
    # Testing on fully different data will likely show eventual overfitting on known features.

    anTrain = np.load(input_dir / "train_l_mouth.npy")

    np.random.shuffle(anTrain)

    anTrain = preprocess_input(anTrain)

    sequence_real = anTrain[:, :16, :]
    sequence_fake = anTrain[:, 16:, :]

    num_samples = anTrain.shape[0]

    num_train = int (0.9 * num_samples)
    num_test = num_samples - num_train

    num_timesteps = sequence_real.shape[1]

    model = get_model(num_timesteps)
         
    # 4 or 5 on batch 256 0.0042 threshold with less daa
    model.fit(sequence_real[:num_train], sequence_real[:num_train], epochs=4, batch_size=256, verbose=1)

    m_dir = get_model_dir()
    model.save(m_dir / 'm1_256-128-12_l_mouth_rr.h5')


    # 0.0061 after 2 epochs         b 56, 128 -  32 - 4
    # 0.0062 after 2-3  epochs      b256, 256 - 128 - 4   (~ 2 hrs)
    # 0.0056 after 5 epochs                                  6 hrs)

    #  256 - 12 - 10
    #

    

    num_test = 2000
    assert num_test <= num_train
    assert num_test <= num_test


    # On train chunk
    reconstruction_error(model, sequence_real[:num_test])

    # on test chunk
    reconstruction_error(model, sequence_real[num_train:num_train + num_test])

    # shuffle test chunk for new test chunk sample
    np.random.shuffle(sequence_real[num_train:])
    reconstruction_error(model, sequence_real[num_train:num_train + num_test])


    m_dir = get_model_dir()
    model.save(m_dir / 'my_model_l_mouth_rr.h5')

    #model.fit(sequence_real, sequence_fake, epochs=1, batch_size=256, verbose=1)
    #model.save(p / 'my_model_rf.h5')


    #model.fit(sequence_fake, sequence_real, epochs=1, batch_size=256, verbose=1)
    #model.save(p / 'my_model_fr.h5')


    model = get_model(num_timesteps)
    model.fit(sequence_fake[:num_train], sequence_fake[:num_train], epochs=5, batch_size=256, verbose=1)
    model.save(p / 'my_model_ff.h5')

    # 0.0031 after 5 epochs                                  6 hrs)