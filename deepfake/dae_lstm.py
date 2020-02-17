

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


from featureline import get_feature_converter

from mp4_frames import get_model_filepath
from mp4_frames import get_test_filepath
from mp4_frames import get_train_filepath
from mp4_frames import get_meta_test_filepath
from mp4_frames import get_pred0_filepath



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

def train(zModel_type, train_file_path, test_file_path, test_meta_file, model_path, limit):
    

    anTrain = np.load(train_file_path)
    anTest = np.load(test_file_path)

    np.random.shuffle(anTrain)

    if limit > 0:
        anTrain = anTrain[:limit]
        anTest = anTest[:limit]

    anTrain = preprocess_input(anTrain)
    anTest = preprocess_input(anTest)

    test_real = anTest[:, :32, :]
    test_fake = anTest[:, 32:, :]

    num_train = anTrain.shape[0]
    num_test = anTest.shape[0]

    num_timesteps = 32

    model = get_model(num_timesteps)

    # For 'fr':
    
    for iEpoch in range(6):
        np.random.shuffle(anTrain)
        train_real = anTrain[:, :32, :]
        train_fake = anTrain[:, 32:, :]

        # Todo provide new real /fake<n> set for each epoch.

        model.fit(train_fake, train_real, epochs=1, batch_size=256, verbose=1)

        #data_p = model.predict(test_real)
        #rms = mean_squared_error(data_p.reshape(-1), test_real.reshape(-1))
        #print(f"Reconstuction error rms = {rms}")


    model.save(model_path)
   


#################################################################################
#
#   main_get_art_arg
#

def main_get_art_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_min", "-tr_min", help="train pair feature dataset, min part", required = True)
    parser.add_argument("--train_max", "-tr_max", help="train pair feature dataset, max part", required = True)

    parser.add_argument("--test_min", "-te_min", help="test feature dataset, min part", required = True)
    parser.add_argument("--test_max", "-te_max", help="test feature dataset, max part", required = True)

    parser.add_argument("--feature", "-f", help="face feature", required = True)

    parser.add_argument("--model_type", "-mtype", help="model type {rr, ff, fr, rf}", required = True)

    parser.add_argument("--limit", "-l", help="data cap. 0: no limit", required = True)

    args = parser.parse_args()

    iTrainPartMin = int (args.train_min)
    iTrainPartMax = int (args.train_max)
    assert iTrainPartMax > iTrainPartMin

    iTestPartMin = int (args.test_min)
    iTestPartMax = int (args.test_max)
    assert iTestPartMax > iTestPartMin

    zFeature = args.feature

    l_features = list (get_feature_converter().keys())

    assert zFeature in l_features

    zModel_type = args.model_type

    assert zModel_type in ['rr', 'ff' ,'rf', 'fr']

    limit = int(args.limit)

    
    train_file_path = get_train_filepath(zFeature, iTrainPartMin, iTrainPartMax, True)
    test_file_path = get_test_filepath(zFeature, iTestPartMin, iTestPartMax, True)
    test_meta_file = get_meta_test_filepath(iTestPartMin, iTestPartMax, True)

    model_path = get_model_filepath(zFeature, zModel_type, False)


    return (zModel_type, train_file_path, test_file_path, test_meta_file, model_path, limit)


def set_test_values():
    iTrainPartMin = 40
    iTrainPartMax = 41

    iTestPartMin = 41
    iTestPartMax = 44

    limit = 10000

    zFeature = 'l_mouth'

    zModel_type = 'rr'


#################################################################################
#
#   __main__
#

if __name__ == '__main__':
    zModel_type, train_file_path, test_file_path, test_meta_file, model_path, limit = main_get_art_arg()

    train(zModel_type, train_file_path, test_file_path, test_meta_file, model_path, limit)





