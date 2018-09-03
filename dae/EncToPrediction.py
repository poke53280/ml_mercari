
import numpy as np
import pandas as pd
import gc
from keras import backend as K

from keras.regularizers import l2 # L2-regularisation

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


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 


DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE

def getXY():
   
    train = pd.read_csv(DATA_DIR + 'train.csv')

    Y = train.target.values
    X = np.load(DATA_DIR + "EMatrix.npy")

    X_train = X[:len(Y)]
    X_test  = X[len(Y):]
    return X_train, X_test, Y

"""c"""


X, _, Y = getXY()


Y = Y / 1000.0
Y = np.log1p(Y)

# LGBM: 1.72
# SVR: 1.75A

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from scipy.special import erfinv



df = pd.DataFrame(X)

df = df.apply(gauss_rank_transform, axis = 0, raw = True)

X = df.values

num_features = X.shape[1]

input_user = Input(shape=(num_features,))

x = Dropout(0.1) (input_user)

x = Dense(500, activation='relu', kernel_regularizer=l2(0.05)) (x)

x = Dropout(0.5) (x)

x = Dense(500, activation='relu' , kernel_regularizer=l2(0.05)) (x)

x = Dropout(0.5) (x)


x = Dense(500, activation='relu', kernel_regularizer=l2(0.05)) (x)

x = Dropout(0.5) (x)

out = Dense(1, activation='linear') (x)

model = Model(input_user, out)

model.compile(loss = root_mean_squared_error, optimizer=Adam(lr = 0.01, decay = 0.995))


X_train = X[:4200]
Y_train = Y[:4200]

X_valid = X[4200:]
Y_valud = X[4200:]


h = model.fit(x=X_train, y=Y_train, batch_size=128, epochs=35, verbose=1, validation_data = (X_valid, Y_valud))


# Test model. save. load.

y_p_test = model.predict(X_valid)

print(y_p_test[0], y_p_test[9], y_p_test[200])

# ==> [6.5986834] [8.930857] [7.621281]

model.save_weights(DATA_DIR + "weight_test")

# Reset, start with mode definition but no training.


model.load_weights(DATA_DIR + "weight_test")

y_p_test = model.predict(X_valid)

print(y_p_test[0], y_p_test[9], y_p_test[200])

# ==> works


loss_stat = h.history['loss']
val_loss_stat = h.history['val_loss']


loss_stat = loss_stat[150:]
val_loss_stat = val_loss_stat[150:]

import matplotlib.pyplot as plt

plt.plot(loss_stat)
plt.plot(val_loss_stat)

plt.show()

from sklearn.svm import SVR


clf = SVR(C=1.0, epsilon=0.2)

clf.fit(X_train, y_train)


y_p = clf.predict(X_valid)


from sklearn.metrics import mean_squared_error

rmsle_error = np.sqrt(mean_squared_error(y_valid, y_p))

#

