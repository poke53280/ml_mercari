
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

from sklearn.metrics import mean_squared_error

from mp4_frames import get_ready_data_dir
from mp4_frames import get_log_dir
from mp4_frames import get_model_dir
from mp4_frames import get_tst_vid


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
#   preprocess_input
#

def preprocess_input(data):
    data = (data.astype(np.float32) - 256/2.0) / (256/2.0)
    return data


####################################################################################
#
#   get_train_files
#

def get_train_files():
    data_dir = get_ready_data_dir()

    l_files = list (data_dir.iterdir())
    l_files = [x for x in l_files if x.stem.startswith("tr_")]

    return l_files


####################################################################################
#
#   predict_on_vid
#

def predict_on_vid(m, anTest):

    anTest_p = m.predict(anTest)

    mse = mean_squared_error(anTest_p.reshape(-1), anTest.reshape(-1))

    diff = (anTest_p - anTest).reshape(-1, 32*3)

    sqr_diff = (diff * diff)

    sum_diff = np.sum(sqr_diff, axis = 1)

    mse_row = sum_diff/ (32*3)

    mse095 = np.quantile(mse_row, 0.95)
    mse005 = np.quantile(mse_row, 0.05)

    return (mse, mse005, mse095)


log_dir = get_log_dir() / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

print(f"log dir: {str(log_dir)}")


model_dir = get_model_dir()


file_writer = tf.summary.create_file_writer(str(log_dir))
file_writer.set_as_default()



num_timesteps = 32


# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir= log_dir, histogram_freq= 1, update_freq = 'epoch', profile_batch=0)

nTrainLimit = 0


l_TestReal = [np.load(get_tst_vid() / "te_41_44_0001_real.npy"), np.load(get_tst_vid() / "te_41_44_0025_real.npy"), np.load(get_tst_vid() / "te_41_44_0044_real.npy")]
l_TestFake = [np.load(get_tst_vid() / "te_41_44_0009_fake.npy"), np.load(get_tst_vid() / "te_41_44_0029_fake.npy"), np.load(get_tst_vid() / "te_41_44_0034_fake.npy")]


l_TestReal = [preprocess_input(x) for x in l_TestReal]
l_TestFake = [preprocess_input(x) for x in l_TestFake]


m = get_model(num_timesteps)

m.compile(loss = keras.losses.mse, optimizer = keras.optimizers.Adam(),metrics = ['mse'])

l_train_path = get_train_files()

num_epochs = 1

log_step = 0

for iEpoch in range(num_epochs):
    for train_path in l_train_path:
        print(f"Loading {train_path}...")
        anTrain = np.load(train_path)

        if nTrainLimit > 0:
            np.random.shuffle(anTrain)
            anTrain = anTrain[:nTrainLimit]

        anTrain = preprocess_input(anTrain)

        num_rows = anTrain.shape[0]

        max_rows_per_run = 10000

        num_splits = int (1 + num_rows / max_rows_per_run)

        l_train = np.array_split(anTrain, num_splits)

        for ichunk, train_chunk in enumerate(l_train):

            print(f"Chunk {ichunk + 1}/ {len(l_train)}")

            train_real = train_chunk[:, :32, :]
            train_fake = train_chunk[:, 32:, :]

            history = m.fit(train_fake, train_real, batch_size = 128, epochs= 1)

            train_real_p = m.predict(train_fake)

            mse_train = mean_squared_error(train_real_p.reshape(-1), train_fake.reshape(-1))

            print(f"mse_train = {mse_train}")

            tf.summary.scalar('mse_train', data=mse_train, step=log_step)

            for ix, x in enumerate(l_TestReal):
                (mse, mse005, mse095) = predict_on_vid(m, x)
                tf.summary.scalar(f"mse_real_{ix}", data = mse, step = log_step)
                tf.summary.scalar(f"mse005_real_{ix}", data = mse005, step = log_step)
                tf.summary.scalar(f"mse095_real_{ix}", data = mse095, step = log_step)


            for ix, x in enumerate(l_TestFake): 
                (mse, mse005, mse095) = predict_on_vid(m, x)
                tf.summary.scalar(f"mse_fake_{ix}", data = mse, step = log_step)
                tf.summary.scalar(f"mse005_fake_{ix}", data = mse005, step = log_step)
                tf.summary.scalar(f"mse095_fake_{ix}", data = mse095, step = log_step)

            log_step = log_step + 1


m.save(model_dir / "common_fr.h5")

