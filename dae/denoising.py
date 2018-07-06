
#
# https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629
#

import dae.SwapNoise
import dae.RankGauss
import numpy as np
import pandas as pd
import scipy 
from scipy.sparse import vstack
from keras.layers import Input, Dense
from keras.models import Model

from keras.optimizers import Adam
from keras.layers import Input, SpatialDropout1D,Dropout, GlobalAveragePooling1D, CuDNNGRU, Bidirectional, Dense, Embedding
from keras.layers import Conv1D
from keras.layers import MaxPool1D, Flatten
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE

train = pd.read_csv(DATA_DIR + 'train.csv')
train = train.drop(['target', 'ID'], axis = 1)

test = pd.read_csv(DATA_DIR + 'test.csv')
test = test.drop(['ID'], axis = 1)

df = pd.concat([train, test], axis = 0)

df.shape


q = df.apply(dae.RankGauss.gauss_rank_transform, axis = 0, raw = True)

q.describe()

y = np.array(q)
X = np.array(q)


y_acc = None
X_acc = None



for i in range(1):

    X0 = dae.SwapNoise.swap_rows(X, 0.15)
    mse = ((y - X0) ** 2).mean(axis=None)
    print(f"idx = {i}, mse = {mse}")

    X_acc = vstack([X_acc, X0])
    y_acc = vstack([y_acc, y])



# Todo: Corrupt at each batch
# 'Each batch samples new noise from the complete dataset. Complete dataset is train+test features.'
# Copy the features before to use as target

#######################################################################

# SERIALIZATION GATE


scipy.sparse.save_npz(DATA_DIR + 'X_acc.npz', X_acc)
scipy.sparse.save_npz(DATA_DIR + 'y_acc.npz', y_acc)


X_acc = scipy.sparse.load_npz(DATA_DIR + 'X_acc.npz')
y_acc = scipy.sparse.load_npz(DATA_DIR + 'y_acc.npz')


print(X_acc.shape)
print(y_acc.shape)

#######################################################################

num_features = X_acc.shape[1]

input_user = Input(shape=(num_features,))

out = Dense(256, activation='relu') (input_user)
out = Dense(256, activation='relu') (out)
out = Dense(256, activation='relu') (out)
out = Dense(num_features, activation='linear') (out)

model = Model(input_user, out)

model.compile(loss='mean_squared_error', optimizer=Adam(lr = 0.003))

model.summary()

model.save_weights(DATA_DIR + 'init_model.h5')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       


NUM_FOLDS = 7

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=22)

lKF = list (enumerate(kf.split(X_acc)))

lRMS = []

while len(lKF) > 0:
    iLoop, (train_index, test_index) = lKF.pop(0)

    print(f"--- Fold: {iLoop}/ {NUM_FOLDS} ---")

    X_train = X_acc.todense()[train_index]
    y_train = y_acc.todense()[train_index]
    
    X_valid = X_acc.todense()[test_index]
    y_valid = y_acc.todense()[test_index]

    model.load_weights(DATA_DIR + 'init_model.h5')

    for i in range(1):
        print(f"Epoch {i + 1}...")
        h = model.fit(x=X_train, y=y_train, batch_size=128, epochs=1, verbose=1, validation_data = (X_valid, y_valid))
        y_p = model.predict(X_valid)
        mse_error = mean_squared_error(y_p, y_valid)
        print (mse_error)
    
    lRMS.append(mse_error)
    
"""c"""
anRMS = np.array(lRMS)

RMSLEmean = anRMS.mean()
RMSLEstd  = anRMS.std()

print(f"  ==> RMSLE = {RMSLEmean} +/- {RMSLEstd}")


# One pre-generated set. ==> RMSLE = 0.43662030341314306 +/- 0.153089487053149