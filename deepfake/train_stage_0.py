
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
from mp4_frames import get_chunk_dir




####################################################################################
#
#   get_model
#


def get_model(num_timesteps):

    encoder_input = keras.Input(shape=(num_timesteps, 3))

    x = layers.Bidirectional(layers.LSTM(256, activation='relu', kernel_initializer='zeros',bias_initializer='zeros', return_sequences=True))(encoder_input)

    x = layers.Bidirectional(layers.LSTM(256, activation='relu', kernel_initializer='zeros',bias_initializer='zeros', return_sequences=True)) (x)

    encoder_output = layers.Bidirectional(layers.LSTM(12, activation='relu', kernel_initializer='zeros',bias_initializer='zeros',)) (x)

    encoder = keras.Model(inputs=encoder_input, outputs=encoder_output, name='encoder')

    encoder.summary()

    x = layers.RepeatVector(num_timesteps)(encoder_output)

    x = layers.Bidirectional(layers.LSTM(12, activation='relu', kernel_initializer='zeros',bias_initializer='zeros', return_sequences=True)) (x)

    x = layers.Bidirectional(layers.LSTM(256, activation='relu', kernel_initializer='zeros',bias_initializer='zeros', return_sequences=True)) (x)

    x = layers.Bidirectional(layers.LSTM(256, activation='relu', kernel_initializer='zeros',bias_initializer='zeros', return_sequences=True)) (x)

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
    data_dir = get_chunk_dir()

    l_files = list (data_dir.iterdir())
    l_files = [x for x in l_files if x.stem.startswith("tr_")]

    return l_files


####################################################################################
#
#   get_test_files
#

def get_test_files():
    data_dir = get_chunk_dir()

    l_files = list (data_dir.iterdir())
    l_files = [x for x in l_files if x.stem.startswith("te_")]

    l_target = [str(x.stem).split("_")[4] for x in l_files]

    filetuple = zip (l_files, l_target)

    return filetuple




zTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

log_dir_real = get_log_dir() / (zTime + "_real")
log_dir_fake = get_log_dir() / (zTime + "_fake")


#file_writer_real = tf.summary.create_file_writer(str(log_dir_real))
#file_writer_fake = tf.summary.create_file_writer(str(log_dir_fake))


model_dir = get_model_dir()


num_timesteps = 32


nTrainLimit = 0


class TestHub:

    def __init__(self, l_test_path):
        self.l_test_file = [x[0] for x in l_test_path]
        self.l_test_target = [x[1] for x in l_test_path]

        for x in self.l_test_file:
            assert x.is_file()

        s = pd.Series(self.l_test_target)

        m_real = s == 'real'
        m_fake = s == 'fake'

        assert (m_real ^ m_fake).all()

        self.idx_real = np.where(m_real)[0]
        self.idx_fake = np.where(m_fake)[0]
       

    def predict(self, idx, m, num_lines):

        # print(f"Predicting {idx}...")
        anTest = np.load(self.l_test_file[idx])

        if num_lines > 0:
            np.random.shuffle(anTest)
            anTest = anTest[:num_lines]

        anTest = preprocess_input(anTest)

        

        anTest_p = m.predict(anTest)

        mse = mean_squared_error(anTest_p.reshape(-1), anTest.reshape(-1))

        diff = (anTest_p - anTest).reshape(-1, 32*3)

        sqr_diff = (diff * diff)

        sum_diff = np.sum(sqr_diff, axis = 1)

        mse_row = sum_diff/ (32*3)

        mse095 = np.quantile(mse_row, 0.95)
        mse005 = np.quantile(mse_row, 0.05)

        return (mse, mse005, mse095)

    def get_first_real(self, num):
        assert num <= len (self.idx_real)
        return self.idx_real[:num]

    def get_first_fake(self, num):
        assert num <= len (self.idx_fake)
        return self.idx_fake[:num]



m = get_model(num_timesteps)

m.compile(loss = keras.losses.mse, optimizer = keras.optimizers.Adam(),metrics = ['mse'])

l_train_path = get_train_files()

print(f"Found {len(l_train_path)} train file(s)")

l_test_path = list(get_test_files())



num_epochs = 1

log_step = 0

for iEpoch in range(num_epochs):
    for train_path in l_train_path:
        print(f"Training on {train_path}...")
        anTrain = np.load(train_path)

        if nTrainLimit > 0:
            np.random.shuffle(anTrain)
            anTrain = anTrain[:nTrainLimit]

        anTrain = preprocess_input(anTrain)

        num_rows = anTrain.shape[0]

        max_rows_per_run = 200000

        num_splits = int (1 + num_rows / max_rows_per_run)

        l_train = np.array_split(anTrain, num_splits)

        for ichunk, train_chunk in enumerate(l_train):

            print(f"Chunk {ichunk + 1}/ {len(l_train)}")

            train_real = train_chunk[:, :32, :]
            train_fake = train_chunk[:, 32:, :]

            #
            # Todo: create noised train set with bits from fake (param).
            # train_fake.shape  (199474, 32, 3)
            # 

            history = m.fit(train_real, train_real, batch_size = 512, epochs= 1)

            #train_real_p = m.predict(train_fake)
            #mse_train = mean_squared_error(train_real_p.reshape(-1), train_fake.reshape(-1))
            #print(f"mse_train = {mse_train}")

            #with file_writer_real.as_default():
            #    tf.summary.scalar('mse_train', data = mse_train, step = log_step)

            #with file_writer_fake.as_default():
            #    tf.summary.scalar('mse_train', data = mse_train, step = log_step)


            test_hub = TestHub(l_test_path)

            num_sample_pairs = 20

            idx_real = test_hub.get_first_real(num_sample_pairs)
            idx_fake = test_hub.get_first_fake(num_sample_pairs)

            l_mse_real = []
            l_mse_005_real = []
            l_mse_095_real = []

            l_mse_fake = []
            l_mse_005_fake = []
            l_mse_095_fake = []

            for i in range (num_sample_pairs):
                (mse, mse005, mse095) = test_hub.predict(idx_real[i], m, 200)

                l_mse_real.append(mse)
                l_mse_005_real.append(mse005)
                l_mse_095_real.append(mse095)

                (mse, mse005, mse095) = test_hub.predict(idx_fake[i], m, 200)

                l_mse_fake.append(mse)
                l_mse_005_fake.append(mse005)
                l_mse_095_fake.append(mse095)

            mse_real_005_mean = np.array(l_mse_005_real).mean()
            mse_real_005_std = np.array(l_mse_005_real).std()

            mse_real_095_mean = np.array(l_mse_095_real).mean()
            mse_real_095_std = np.array(l_mse_095_real).std()

            print(f"mean r:{np.array(l_mse_real).mean():.3} f:{np.array(l_mse_fake).mean():.3}")
            print(f"lo   r:{np.array(l_mse_005_real).mean():.3} f:{np.array(l_mse_005_fake).mean():.3}")
            print(f"hi   r:{np.array(l_mse_095_real).mean():.3} f:{np.array(l_mse_095_fake).mean():.3}")


            #with file_writer_real.as_default():
            #    tf.summary.scalar(f"mean", data = np.array(l_mse_real).mean(), step = log_step)
            #    tf.summary.scalar(f"std", data = np.array(l_mse_real).std(), step = log_step)

            #    tf.summary.scalar(f"lo_mean", data = np.array(l_mse_005_real).mean(), step = log_step)
            #    tf.summary.scalar(f"lo_std", data = np.array(l_mse_005_real).std(), step = log_step)

            #    tf.summary.scalar(f"hi_mean", data = np.array(l_mse_095_real).mean(), step = log_step)
            #    tf.summary.scalar(f"hi_std", data = np.array(l_mse_095_real).std(), step = log_step)

            #with file_writer_fake.as_default():

            #    tf.summary.scalar(f"mean", data = np.array(l_mse_fake).mean(), step = log_step)
            #    tf.summary.scalar(f"std", data = np.array(l_mse_fake).std(), step = log_step)

            #    tf.summary.scalar(f"lo_mean", data = np.array(l_mse_005_fake).mean(), step = log_step)
            #    tf.summary.scalar(f"lo_std", data = np.array(l_mse_005_fake).std(), step = log_step)

            #    tf.summary.scalar(f"hi_mean", data = np.array(l_mse_095_fake).mean(), step = log_step)
            #    tf.summary.scalar(f"hi_std", data = np.array(l_mse_095_fake).std(), step = log_step)

            log_step = log_step + 1


m.save(model_dir / "common_fr.h5")

