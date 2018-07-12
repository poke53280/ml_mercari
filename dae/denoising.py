
#
# Several ideas from https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
#

import numpy as np
import pandas as pd
import scipy 
from keras.layers import Input, Dense
from keras.models import Model

from keras.optimizers import Adam
from keras.layers import Input, Dense
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from keras import backend as K


import time
import gc

from scipy.special import erfinv



DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
DATA_DIR_AWS = "./"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_AWS


######################################################################
#
#      add_swap_noise
#
#

def add_swap_noise(X_batch, X_clean, p):

    nNumRowsBatch = X_batch.shape[0]
    nNumRowsSource = X_clean.shape[0]

    print(f"Adding {p * 100.0}% noise to {nNumRowsBatch} row(s) from noise pool of {nNumRowsSource} row(s).")

    print(f"   Creating noise source indices")
    aiNoiseIndex = np.random.randint(nNumRowsSource, size=nNumRowsBatch)

    print(f"   Allocating noise source")
    X_noise = X_clean[aiNoiseIndex]

    print(f"   Allocating noise mask")
    X_mask = np.random.rand(X_batch.shape[0], X_batch.shape[1])

    print(f"   Applying noise")
    m = X_mask < p

    X_batch[m] = 0
    X_noise[~m] = 0

    X_batch = X_noise + X_batch

    return X_batch
"""c"""

######################################################################
#
#      rank_gauss
#
#

def rank_gauss(x):
    # x is numpy vector
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2 # rank_x.max(), rank_x.min() should be in (-1, 1)
    efi_x = erfinv(rank_x) # np.sqrt(2)*erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x


######################################################################
#
#      gauss_rank_transform
#

def gauss_rank_transform(x):
    return pd.Series(rank_gauss(x))


######################################################################
#
#      loadY
#

def loadY():

    train = pd.read_csv(DATA_DIR + 'train.csv')
    print(f"Loading train data...")
    train = train.drop(['target', 'ID'], axis = 1)

    print(f"Loading test data...")
    test = pd.read_csv(DATA_DIR + 'test.csv')
    test = test.drop(['ID'], axis = 1)

    df = pd.concat([train, test], axis = 0)

    df = df.apply(gauss_rank_transform, axis = 0, raw = True)

    Y = np.array(df, dtype='float32')  #todo  - run with float64, possible loss in accuracy.

    return Y

########################################################################
#
#    trainCV
#
#

def trainCV(y, num_epochs, noise_factor, num_folds, lRunFolds, n_batchsize, num_neurons):

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=22)   

    lKF = list (enumerate(kf.split(y)))

    lRMS = []

    for iFold in lRunFolds:
        assert iFold >= 0 and iFold < num_folds

        iLoop, (train_index, test_index) = lKF[iFold]

        print(f" Running fold {iLoop +1}/ {num_folds}")

        # Create model for each fold
        num_features = y.shape[1]

        model = create_model(num_features, num_neurons)

        y_train_const = y[train_index]
        y_valid_const = y[test_index]    

        X_train_const = y[train_index]
        X_valid_const = y[test_index]

        for i in range(num_epochs):
            print(f"Fold {iLoop + 1}/ {num_folds} Epoch {i + 1}/ {num_epochs}:")

            y_train = y_train_const.copy()
            y_valid = y_valid_const.copy()

            X_train = X_train_const.copy()
            X_valid = X_valid_const.copy()
            

            X_train = add_swap_noise(X_train, y, noise_factor)
            X_valid = add_swap_noise(X_valid, y, noise_factor)

            p_t = np.random.permutation(len(y_train))
            p_v = np.random.permutation(len(y_valid))

            X_train = X_train[p_t]
            y_train = y_train[p_t]

            X_valid = X_valid[p_v]
            y_valid = y_valid[p_v]


            h = model.fit(x=X_train, y=y_train, batch_size=n_batchsize, epochs=1, verbose=1, validation_data = (X_valid, y_valid))
            
            y_p = model.predict(X_valid)

            mse_error = mean_squared_error(y_p, y_valid)

            print(f"Fold {iLoop + 1}/ {num_folds} Epoch {i + 1}/ {num_epochs} finished. MSE = {mse_error}.")
    
        lRMS.append(mse_error)
        del model
        
        K.clear_session()
        gc.collect()

    anRMS = np.array(lRMS)

    return anRMS
    
"""c"""


########################################################################
#
#    create_model
#
#

def create_model(num_features, num_neurons):

    input_user = Input(shape=(num_features,))

    encoded_l_0 = Dense(num_neurons, activation='relu') (input_user)
    encoded_l_1 = Dense(num_neurons, activation='relu') (encoded_l_0)
    encoded_l_2 = Dense(num_neurons, activation='relu') (encoded_l_1)

    decoded = Dense(num_features, activation='linear') (encoded_l_2)

    autoencoder = Model(input_user, decoded)

    autoencoder.compile(loss='mean_squared_error', optimizer=Adam(lr = 0.003, decay = 0.995))

    autoencoder.summary()
    
    return autoencoder

########################################################################
#
#    create_model_large
#
#

def create_model_large(num_features):

    input_user = Input(shape=(num_features,))

    x = Dense(15000, activation='relu') (input_user)
    x = Dense(15000, activation='relu') (x)
    x = Dense(3000, activation='linear') (x)
    x = Dense(15000, activation='relu') (x)
    x = Dense(15000, activation='relu') (x)

    decoded = Dense(num_features, activation='linear') (x)

    autoencoder = Model(input_user, decoded)

    autoencoder.compile(loss='mean_squared_error', optimizer=Adam(lr = 0.001, decay = 0.995))

    autoencoder.summary()
    
    return autoencoder


########################################################################
#
#    main
#
#

def main():
    Y = loadY()
   
    # lRunFolds = list (range(9))

    lRunFolds = [2]

    anMSE = trainCV(Y, 1000, 0.11, 9, lRunFolds, 128, 5000)

    MSEmean = anMSE.mean()
    MSEstd  = anMSE.std()
    
    print(f"  ==> MSE = {MSEmean} +/- {MSEstd}")


if __name__ == "__main__": main()  

# Fold 3/ 9 Epoch 3/ 3 finished. MSE = 0.06062798947095871. 1500
# Fold 3/ 9 Epoch 3/ 3 finished. MSE = 0.06034927815198898. 2500
# Fold 3/ 9 Epoch 3/ 3 finished. MSE = 0.06111340969800949. 3500


# at 1500n: Fold 3/ 9 Epoch 109/ 1000 finished. MSE = 0.05556689202785492.
# Fold 3/ 9 Epoch 4/ 1000 finished. MSE = 0.06124715879559517 8500

# plateau abpit 0.055

# running 5000
