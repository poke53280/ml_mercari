
#
# Several ideas from https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
#

import numpy as np
import pandas as pd
import scipy 
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from keras import backend as K
import time
import gc
from scipy.special import erfinv
import random

from keras import regularizers


DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
DATA_DIR_AWS = "./"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE

def keras_root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 


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

    print(f"Loading train data...")
    train = pd.read_csv(DATA_DIR + 'train.csv')
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
#    trainAll
#
#

def trainAll(Y, config):
    num_features = Y.shape[1]
    num_rows = Y.shape[0]

    models = create_dae_model(num_features, config)

    s = SwapNoise()

    for i in range(config['num_epochs']):

        print(f"Epoch {i + 1}/ {config['num_epochs']}:")

        y_train = Y.copy()
        X_train = Y.copy()

        X_train = s.add_swap_noise(X_train, Y, config['noise_factor'], config['verbose'])

        p_t = np.random.permutation(len(y_train))
      
        X_train = X_train[p_t]
        y_train = y_train[p_t]

        h = models['autoencoder'].fit(x=X_train, y=y_train, batch_size=config['mini_batch_size'], epochs=1, verbose=config['verbose'])

    return models


########################################################################
#
#    trainCV
#
#

def trainCV(Y, num_folds, lRunFolds, config):

    num_features = Y.shape[1]

    num_rows = Y.shape[0]

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=22)   

    lKF = list (enumerate(kf.split(Y)))

    lRMS = []

    E0 = np.zeros((num_rows, config['num_neurons']), dtype='float32') # Can be empty
    E1 = np.zeros((num_rows, config['num_neurons']), dtype='float32') 
    E2 = np.zeros((num_rows, config['num_neurons']), dtype='float32') 

    for iFold in lRunFolds:
        assert iFold >= 0 and iFold < num_folds

        iLoop, (train_index, test_index) = lKF[iFold]

        print(f" Running fold {iLoop +1}/ {num_folds}")

        models = create_dae_model(num_features, config)

        s = SwapNoise()

        y_train_const = Y[train_index]
        y_valid_const = Y[test_index]    

        for i in range(config['num_epochs']):
            print(f"Fold {iLoop + 1}/ {num_folds} Epoch {i + 1}/ {config['num_epochs']}:")

            y_train = y_train_const.copy()
            y_valid = y_valid_const.copy()

            X_train = y_train_const.copy()
            X_valid = y_valid_const.copy()

            X_train = s.add_swap_noise(X_train, Y, config['noise_factor'], config['verbose'])
            X_valid = s.add_swap_noise(X_valid, Y, config['noise_factor'], config['verbose'])

            p_t = np.random.permutation(len(y_train))
            p_v = np.random.permutation(len(y_valid))

            X_train = X_train[p_t]
            y_train = y_train[p_t]

            X_valid = X_valid[p_v]
            y_valid = y_valid[p_v]

            h = models['autoencoder'].fit(x=X_train, y=y_train, batch_size=config['mini_batch_size'], epochs=1, verbose=config['verbose'], validation_data = (X_valid, y_valid))
            
            y_p = models['autoencoder'].predict(X_valid)

            mse_error = mean_squared_error(y_p, y_valid)

            print(f"Fold {iLoop + 1}/ {num_folds} Epoch {i + 1}/ {config['num_epochs']} finished. MSE = {mse_error}.")
   
        lRMS.append(mse_error)
        
        K.clear_session()
        gc.collect()

    anRMS = np.array(lRMS)

    return anRMS
    
"""c"""

########################################################################
#
#    create_stage2_model
#
#

def create_stage2_model(config):

    num_features = config['num_neurons'] * 3
    
    input_user = Input(shape=(num_features,))

    x = Dropout(0.1) (input_user)
    x = Dense(500, activation='relu', kernel_regularizer=regularizers.l2(0.05)) (x)
    x = Dropout(0.5) (x)
    x = Dense(500, activation='relu' , kernel_regularizer=regularizers.l2(0.05)) (x)
    x = Dropout(0.5) (x)
    x = Dense(500, activation='relu', kernel_regularizer=regularizers.l2(0.05)) (x)
    x = Dropout(0.5) (x)
    out = Dense(1, activation='linear') (x)
    model = Model(input_user, out)
    model.compile(loss = keras_root_mean_squared_error, optimizer=Adam(lr = 0.01, decay = 0.995))

    return model


########################################################################
#
#    create_stage2_model_no_regularization
#
#

def create_stage2_model_no_regularization(config):

    num_features = config['num_neurons'] * 3
    
    input_user = Input(shape=(num_features,))

    x = Dense(500, activation='relu') (input_user)
    x = Dense(500, activation='relu') (x)
    x = Dense(500, activation='relu') (x)
    out = Dense(1, activation='linear') (x)
    model = Model(input_user, out)

    model.compile(loss = keras_root_mean_squared_error, optimizer=Adam(lr = 0.01, decay = 0.995))

    return model

########################################################################
#
#    create_dae_model
#
#

def create_dae_model(num_features, config):

    models = {}

    input_user = Input(shape=(num_features,))

    encoded_l_0 = Dense(config['num_neurons'], activation='relu') (input_user)
    encoder_0 = Model(input_user, encoded_l_0)


    encoded_l_1 = Dense(config['num_neurons'], activation='relu') (encoded_l_0)
    encoder_1   = Model(input_user, encoded_l_1)

    encoded_l_2 = Dense(config['num_neurons'], activation='relu') (encoded_l_1)
    encoder_2   = Model(input_user, encoded_l_2)

    decoded = Dense(num_features, activation='linear') (encoded_l_2)

    autoencoder = Model(input_user, decoded)

    autoencoder.compile(loss='mean_squared_error', optimizer=Adam(lr = config['r_learning_rate'], decay = config['r_decay']))

    models['autoencoder'] = autoencoder

    models['encoder_0'] = encoder_0
    models['encoder_1'] = encoder_1
    models['encoder_2'] = encoder_2
    
    return models


class Configurator:
    _c = {}

    def __init__(self):
        self._c['num_neurons']       = 1500
        self._c['r_learning_rate']   = 0.003
        self._c['r_decay']           = 0.995
        self._c['mini_batch_size']   = 128
        self._c['noise_factor']      = 0.11
        self._c['num_epochs']        = 4
        self._c['verbose']           = 1

    def get_configuration(self):
        return self._c

    def randomize(self):
        self._c['num_neurons']       = random.choice([1500, 2500, 3500])
        self._c['r_learning_rate']   = random.choice([0.003, 0.001, 0.002])
        self._c['r_decay']           = 0.995
        self._c['mini_batch_size']   = random.choice([128, 32, 256])
        self._c['noise_factor']      = random.choice([0.11, 0.15, 0.12])
        self._c['num_epochs']        = 5
        self._c['verbose']           = 1

"""c"""

########################################################################
#
#    stage2
#
#

def stage2(config):
    
    num_folds = 9
    lRunFolds = list (range(num_folds))
    
    df = pd.read_csv(DATA_DIR + 'train.csv')

    train_id = df.ID

    y = df.target.values
    
    y = np.log1p(y)
    y = y.astype(np.float32)    

    df = df.drop(['target', 'ID'], axis = 1)

    df = df.apply(gauss_rank_transform, axis = 0, raw = True)

    X = np.array(df, dtype='float32')

    num_features = X.shape[1]

    configurator = Configurator()

    config = configurator.get_configuration()

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=22)   

    lKF = list (enumerate(kf.split(X)))

    lRMSLE = []

    for iFold in lRunFolds:
        assert iFold >= 0 and iFold < num_folds

        iLoop, (train_index, test_index) = lKF[iFold]

        print(f" Running fold {iLoop +1}/ {num_folds}")


        enc = create_dae_model(num_features, c)

        del enc['autoencoder']
        enc['encoder_0'].load_weights(DATA_DIR + "encoder_0")
        enc['encoder_1'].load_weights(DATA_DIR + "encoder_1")
        enc['encoder_2'].load_weights(DATA_DIR + "encoder_2")

        m = create_stage2_model(c)

        s = SwapNoise()

        x_train_const = X[train_index]
        x_valid_const = X[test_index]    

        y_train_const = y[train_index]
        y_valid_const = y[test_index]

        for i in range(config['num_epochs']):
            print(f"Fold {iLoop + 1}/ {num_folds} Epoch {i + 1}/ {config['num_epochs']}:")

            y_train = y_train_const.copy()
            y_valid = y_valid_const.copy()

            X_train = x_train_const.copy()
            X_valid = x_valid_const.copy()

            X_train = s.add_swap_noise(X_train, X, 0.0, config['verbose'])
            X_valid = s.add_swap_noise(X_valid, X, 0.0, config['verbose'])

            p_t = np.random.permutation(len(y_train))
            p_v = np.random.permutation(len(y_valid))

            X_train = X_train[p_t]
            y_train = y_train[p_t]

            X_valid = X_valid[p_v]
            y_valid = y_valid[p_v]

            E0_train = enc['encoder_0'].predict(X_train)               
            E1_train = enc['encoder_1'].predict(X_train) 
            E2_train = enc['encoder_2'].predict(X_train) 

            E0_valid = enc['encoder_0'].predict(X_valid)               
            E1_valid = enc['encoder_1'].predict(X_valid) 
            E2_valid = enc['encoder_2'].predict(X_valid)

            E_train = np.hstack((E0_train, E1_train, E2_train))
            E_valid = np.hstack((E0_valid, E1_valid, E2_valid))

            h = m.fit(x=E_train, y=y_train, batch_size=128, epochs=1, verbose=config['verbose'], validation_data = (E_valid, y_valid))
            
            y_p = m.predict(E_valid)

            rmsle_error = np.sqrt(mean_squared_error(y_valid, y_p))

            print(f"Fold {iLoop + 1}/ {num_folds} Epoch {i + 1}/ {config['num_epochs']} finished. RMSLE = {rmsle_error}.")

        lRMSLE.append(rmsle_error)
        
        K.clear_session()
        gc.collect()

    anRMS = np.array(lRMSLE)
    return anRMS

"""c"""


########################################################################
#
#    main
#
#

def main():
    
    Y = loadY()

    #num_folds = 9
    #lRunFolds = [2]
    # lRunFolds = list (range(num_folds))

    configurator = Configurator()

    c = configurator.get_configuration()

    c['num_epochs'] = 45

    info = f"Starting: n{c['num_neurons']}lr{c['r_learning_rate']}d{c['r_decay']}b{c['mini_batch_size']}sw{c['noise_factor']}e{c['num_epochs']} "
    print (info)


    m = trainAll(Y, c)

    m['encoder_0'].save_weights(DATA_DIR + "encoder_0")
    m['encoder_1'].save_weights(DATA_DIR + "encoder_1")
    m['encoder_2'].save_weights(DATA_DIR + "encoder_2")

    del m['autoencoder']
    
    info = f"Is completed: n{c['num_neurons']}lr{c['r_learning_rate']}d{c['r_decay']}b{c['mini_batch_size']}sw{c['noise_factor']}e{c['num_epochs']} "

    print (info)




    #anMSE, E = trainCV(Y, num_folds, lRunFolds, c)

    #MSEmean = anMSE.mean()
    #MSEstd  = anMSE.std()

    #info = f"Is completed: n{c['num_neurons']}lr{c['r_learning_rate']}d{c['r_decay']}b{c['mini_batch_size']}sw{c['noise_factor']}e{c['num_epochs']} ==> {MSEmean}+/-{MSEstd}"

    #print(info)
    #np.save(DATA_DIR + "EMatrix", E)

#
#    if len(lRunFolds) == num_folds:
#        print("Saving " + info + "...")
#
#       np.save(DATA_DIR + "EMatrix", E)
#        F = np.load(DATA_DIR + "EMatrix.npy")
#        if (E == F).all():
#            print("Matrix saved and verified OK")
#    else:
#        print("Not all folds run, no predictions saved")


if __name__ == "__main__": main()  

# 0.11, 256 (?)
# at 1500, maxes at 0.0490 after about 100 epochs

# at 3500, plataeu at 0.0480 - 0.0486 from 300 -490
# at 3500, plataeu at 0.0480 - 0.0486 from 600 as well.


# 2500, 32, noise 0.2  37 mill params
# NO, stops at 0.0520 early , 40 epochs.



#
#
# Fold 3/ 9 Epoch 3/ 3 finished. MSE = 0.06062798947095871. 1500
# Fold 3/ 9 Epoch 3/ 3 finished. MSE = 0.06034927815198898. 2500
# Fold 3/ 9 Epoch 3/ 3 finished. MSE = 0.06111340969800949. 3500
#
# at 1500n: Fold 3/ 9 Epoch 109/ 1000 finished. MSE = 0.05556689202785492.
# Fold 3/ 9 Epoch 4/ 1000 finished. MSE = 0.06124715879559517 8500
#
# plateau abpit 0.055
#
# running 5000
# not moving a lot at: Fold 3/ 9 Epoch 180/ 1000 finished. MSE = 0.054184440523386.
#
#



