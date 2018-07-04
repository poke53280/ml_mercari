




import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

DATA_DIR_PORTABLE = "C:\\santander_3_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE

train_CONST = pd.read_csv(DATA_DIR + 'train.csv')

y_trainFull = np.log(train_CONST.target)
train_id = train_CONST.ID

train_CONST = train_CONST.drop(['target', 'ID'], axis = 1)


train = train_CONST


def create_nn_sequence(x):

    an = np.array(x)
    m = (an != 0)
    nz = np.where(m)
    nz = nz[0]

    an_out = np.zeros(shape= (len(x)), dtype = np.float32)

    for index, n in enumerate(nz):

        value = an[n]

        #Split value over all preceding zeros and self.

        offset_start = 0

        if index > 0:
            former_non_null = nz[index - 1]
            offset_start = former_non_null + 1

        else:
            offset_start = 0

        nElements = n + 1 - offset_start
        value = value / nElements

        an_out[offset_start: n +1 ] = value


    return pd.Series (list (an_out))

"""c"""

q = train.apply(create_nn_sequence, axis = 1)

X = np.array(q)

non_zero_cols = np.count_nonzero(X, axis = 0)

m = non_zero_cols > 0

X = X[:, m]

# So most populous to the left
X_nz = np.count_nonzero(X, axis = 0)
idx = (-X_nz).argsort()
X = X[:, idx]



from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()

X = scaler.fit_transform(X)

# X = np.expand_dims(X, axis=2)



from keras.layers import Input, Dense
from keras.models import Model

from keras.optimizers import Adam

from keras.layers import Input, SpatialDropout1D,Dropout, GlobalAveragePooling1D, CuDNNGRU, Bidirectional, Dense, Embedding

from keras.layers import Conv1D

from keras.layers import MaxPool1D, Flatten

input_user = Input(shape=(X.shape[1],))

out = Dense(192) (input_user)
out = Dropout(0.1) (out)
out = Dense(32) (out)
out = Dense(64) (out)
out = Dense(1)(out)

model = Model(input_user, out)

model.compile(loss='mean_squared_error', optimizer=Adam(lr = 0.001))

model.summary()

model.save_weights(DATA_DIR + 'init_model.h5')



NUM_FOLDS = 7

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=22)

lKF = list (enumerate(kf.split(X)))

lRMS = []
   
while len(lKF) > 0:
    iLoop, (train_index, test_index) = lKF.pop(0)

    print(f"--- Fold: {iLoop +1}/ {NUM_FOLDS} ---")

    X_train = X[train_index]
    y_train = y_trainFull[train_index]
    
    X_valid = X[test_index]
    y_valid = y_trainFull[test_index]

    model.load_weights(DATA_DIR + 'init_model.h5')

    for i in range(6):
        print(f"Epoch {i + 1}...")
        h = model.fit(x=X_train, y=y_train, batch_size=4, epochs=10, verbose=1, validation_data = (X_valid, y_valid))
        y_p = model.predict(X_valid)
        rmsle_error = np.sqrt(mean_squared_error(y_p, y_valid))
        print (rmsle_error)
    
    lRMS.append(rmsle_error)
    
"""c"""
anRMS = np.array(lRMS)

RMSLEmean = anRMS.mean()
RMSLEstd  = anRMS.std()

print(f"  ==> RMSLE = {RMSLEmean} +/- {RMSLEstd}")




#0.1 - MLP 64 - 0.1 - 32
# RMSLE = 1.6569574771917635 +/- 0.05794541105025524

#
# convnet 20 , 20, maxpool 12 flatten. dense (1)
# 
# ==> RMSLE = 1.734177937270466 +/- 0.03589383368555738
#
# convnet 200, 40, 12, flatten, dense.
# Won't converge beyond 3
#
# convnet 100 40, 12, flatten, dense

# converge to about 2.6 or
# try with sorted features.
#
#
# => NO DIFF



