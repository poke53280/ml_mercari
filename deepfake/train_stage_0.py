
#
# Launch tensorboard:
#
# sudo python3.7 -m tensorboard.main --logdir=. --port 80 --bind_all
#
#

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pathlib
import numpy as np
import pandas as pd
import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



####################################################################################
#
#   get_model
#


def get_model(num_timesteps):

    encoder_input = keras.Input(shape=(num_timesteps, 3))

    x = layers.LSTM(512, activation='relu', kernel_initializer='zeros',bias_initializer='zeros', return_sequences=True)(encoder_input)

    x = layers.LSTM(256, activation='relu', kernel_initializer='zeros',bias_initializer='zeros', return_sequences=True) (x)

    encoder_output = layers.LSTM(12, activation='relu', kernel_initializer='zeros',bias_initializer='zeros',) (x)

    encoder = keras.Model(inputs=encoder_input, outputs=encoder_output, name='encoder')

    encoder.summary()

    x = layers.RepeatVector(num_timesteps)(encoder_output)

    x = layers.LSTM(12, activation='relu', kernel_initializer='zeros',bias_initializer='zeros', return_sequences=True) (x)

    x = layers.LSTM(256, activation='relu', kernel_initializer='zeros',bias_initializer='zeros', return_sequences=True) (x)

    x = layers.LSTM(512, activation='relu', kernel_initializer='zeros',bias_initializer='zeros', return_sequences=True) (x)

    decoder_output = layers.TimeDistributed(layers.Dense(3, kernel_initializer='zeros',bias_initializer='zeros')) (x)

    autoencoder  = keras.Model(inputs=encoder_input, outputs=decoder_output, name='autoencoder')

    autoencoder.summary()

    return autoencoder



####################################################################################
#
#   get_path_set
#
#

def get_path_set():

    isLocal = os.name == 'nt'

    if isLocal:

        train_path = pathlib.Path("C:\\Users\\T149900\\ready_data\\train_l_mouth_p_40_p_41.npy")
        test_path = pathlib.Path("C:\\Users\\T149900\\ready_data\\test_l_mouth_p_40_p_41.npy")
        meta_path = pathlib.Path("C:\\Users\\T149900\\ready_data\\test_meta_p_40_p_41.pkl")
        log_dir_base = pathlib.Path("C:\\Users\\T149900\\log_dir")
    else:
        #train_path = pathlib.Path("/mnt/disks/tmp_mnt/data/ready_data/train_l_mouth_p_40_p_41.npy")
        #test_path = pathlib.Path("/mnt/disks/tmp_mnt/data/ready_data/test_l_mouth_p_41_p_44.npy")
        #meta_path = pathlib.Path("/mnt/disks/tmp_mnt/data/ready_data/test_meta_p_41_p_44.pkl")
        train_path = pathlib.Path("/mnt/disks/tmp_mnt/data/ready_data/train_l_mouth_p_0_p_40.npy")
        test_path = pathlib.Path("/mnt/disks/tmp_mnt/data/ready_data/test_l_mouth_p_41_p_44.npy")
        meta_path = pathlib.Path("/mnt/disks/tmp_mnt/data/ready_data/test_meta_p_41_p_44.pkl")

        log_dir_base = pathlib.Path("/mnt/disks/tmp_mnt/data/log_dir")
        

    assert train_path.is_file()
    assert test_path.is_file()
    assert meta_path.is_file()
    assert log_dir_base.is_dir()        



    log_dir = log_dir_base / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    return (train_path, test_path, meta_path, log_dir)



####################################################################################
#
#   preprocess_input
#

def preprocess_input(data):
    data = (data.astype(np.float32) - 256/2.0) / (256/2.0)
    return data


num_timesteps = 32

train_path, test_path, meta_path, log_dir = get_path_set()

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir= log_dir, histogram_freq= 1, update_freq = 'batch', profile_batch=0)

nTestLimit = 5000

anTest = np.load(test_path)
np.random.shuffle(anTest)

anTest = preprocess_input(anTest)

test_real = anTest[:, :32, :]
test_fake = anTest[:, 32:, :]


if nTestLimit > 0:
    anTest = anTest[:nTestLimit]


df_meta = pd.read_pickle(meta_path)

m = get_model(num_timesteps)

m.compile(loss = keras.losses.mse, optimizer = keras.optimizers.Adam(),metrics = ['mse'])

l_train_path = [train_path]

for iEpoch in range(20):
    for train_path in l_train_path:
        print(f"Loading {train_path}...")
        anTrain = np.load(train_path)
        anTrain = preprocess_input(anTrain)

        train_real = anTrain[:, :32, :]
        train_fake = anTrain[:, 32:, :]

        history = m.fit(train_fake, train_real, batch_size= 128, epochs= 1, validation_split= 0.2, callbacks=[tensorboard_callback])

        print("Checkpoint. Save model. Progress in epoch, progress on train file.")



