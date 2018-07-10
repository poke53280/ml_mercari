
#
# https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
#

from dae.SwapNoise import add_swap_noise
import dae.RankGauss
import numpy as np
import pandas as pd
import scipy 
from scipy.sparse import vstack
from keras.layers import Input, Dense
from keras.models import Model

from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers import Input, SpatialDropout1D,Dropout, GlobalAveragePooling1D, CuDNNGRU, Bidirectional, Dense, Embedding
from keras.layers import Conv1D
from keras.layers import MaxPool1D, Flatten
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


import time
import gc


########################################################################
#
#    trainCV
#
#

def trainCV(model, y, num_epochs, noise_factor, num_folds, n_batchsize, num_sleep_secs, model_file):

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=22)

    lKF = list (enumerate(kf.split(y)))

    lRMS = []

    while len(lKF) > 0:
        iLoop, (train_index, test_index) = lKF.pop(0)

        print(f"--- Fold: {iLoop + 1}/ {num_folds} ---")

        # Clear model to random/init
        model.load_weights(model_file)
   

        for i in range(num_epochs):
            print(f"Epoch {i + 1} / {num_epochs}...")

            y_train = y [train_index]
            y_valid = y[test_index]    

            X_train = y[train_index]
            X_valid = y[test_index]

            X_train = add_swap_noise(X_train, y, noise_factor)
            X_valid = add_swap_noise(X_valid, y, noise_factor)

            p_t = np.random.permutation(len(y_train))
            p_v = np.random.permutation(len(y_valid))

            X_train = X_train[p_t]
            y_train = y_train[p_t]

            X_valid = X_valid[p_v]
            y_valid = y_valid[p_v]

            if i > 0:
                # GPU cooling
                time.sleep(num_sleep_secs)

            h = model.fit(x=X_train, y=y_train, batch_size=n_batchsize, epochs=1, verbose=1, validation_data = (X_valid, y_valid))
            y_p = model.predict(X_valid)

            mse_error = mean_squared_error(y_p, y_valid)

            print(f"Epoch finished. MSE = {mse_error}.")
    
        lRMS.append(mse_error)

    anRMS = np.array(lRMS)

    return anRMS
    
"""c"""

DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE

def loadY():

    train = pd.read_csv(DATA_DIR + 'train.csv')
    train = train.drop(['target', 'ID'], axis = 1)

    test = pd.read_csv(DATA_DIR + 'test.csv')
    test = test.drop(['ID'], axis = 1)

    df = pd.concat([train, test], axis = 0)

    df = df.apply(dae.RankGauss.gauss_rank_transform, axis = 0, raw = True)

    Y = np.array(df)

    return Y


Y = loadY()

gc.collect()

num_features = Y.shape[1]

input_user = Input(shape=(num_features,))

out = Dense(1500, activation='relu') (input_user)
out = Dense(1500, activation='relu') (out)
out = Dense(1500, activation='relu') (out)
out = Dense(num_features, activation='linear') (out)

model = Model(input_user, out)

model.compile(loss='mean_squared_error', optimizer=Adam(lr = 0.003, decay = 0.995))

model.summary()

model_file_name = DATA_DIR + 'init_model3.h5'

model.save_weights(model_file_name)        

anMSE = trainCV(model, Y, 80, 0.11, 7, 128, 15.0, model_file_name)

MSEmean = anMSE.mean()
MSEstd  = anMSE.std()
    
print(f"  ==> MSE = {MSEmean} +/- {MSEstd}")

#
#
#
# swap 0.11 epoch 4 -   0.0604 - val_loss: 0.0611
#
#
#
#
# swa0 0.15
# Working: 
# epoch 80: loss: 0.0608 - val_loss: 0.0615
#
#
# Adam. lr = 0.003, decay = 0.995
#input_user = Input(shape=(num_features,))
#
#out = Dense(1500, activation='relu') (input_user)
#out = Dense(1500, activation='relu') (out)
#out = Dense(1500, activation='relu') (out)
#out = Dense(num_features, activation='linear') (out)
# X_train = add_swap_noise(X_train, y, 0.15)
#  h = model.fit(x=X_train, y=y_train, batch_size=128, epochs=1, verbose=1, validation_data = (X_valid, y_valid))
#
#
# OOM on biggest example
#
#
