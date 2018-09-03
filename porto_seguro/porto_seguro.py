

import pandas as pd
import numpy as np


from keras.layers import Input, Dense
from keras.models import Model

from keras.optimizers import Adam
from sklearn.model_selection import KFold

from dae.SwapNoise import add_swap_noise


DATA_DIR_PORTABLE = "C:\\porto_seguro_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE


train = pd.read_csv(DATA_DIR + 'train.csv')

train = train.drop(['id', 'target'], axis = 1)

test = pd.read_csv(DATA_DIR + 'test.csv')
test = test.drop(['id'], axis = 1)

df = pd.concat([train, test], axis = 0)

# Drop all 'calc' features
#
all_c = df.columns


drop_c = []

for c in all_c:
    if 'calc' in c:
        drop_c.append(c)        

df = df.drop(drop_c, axis = 1)

all_c = df.columns

# 37 columns

cat_c = []

for c in all_c:
    if 'cat' in c:
        cat_c.append(c)        

"""c"""

for c in cat_c:

    p = f"{c}_"

    q = pd.get_dummies(df[c], prefix = p)
    df = pd.concat([df, q], axis = 1)

"""c"""


import dae.RankGauss

all_c = df.columns

non_binary_c = []

for c in all_c:
    min = df[c].min()
    max = df[c].max()

    if min == 0 and max == 1:
        pass
    else:
        non_binary_c.append(c)


df[non_binary_c] = df[non_binary_c].apply(dae.RankGauss.gauss_rank_transform, axis = 0, raw = True)


Y = np.array(df, dtype = np.float)


num_features = Y.shape[1]

input_user = Input(shape=(num_features,))


out = Dense(1500, activation='relu') (input_user)
out = Dense(1500, activation='relu') (out)
out = Dense(1500, activation='relu') (out)
out = Dense(num_features, activation='linear') (out)

model = Model(input_user, out)

model.compile(loss='mean_squared_error', optimizer=Adam(lr = 0.003, decay = 0.995))

model.summary()

model.save_weights(DATA_DIR + 'init_model.h5')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       

NUM_FOLDS = 7

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=22)

lKF = list (enumerate(kf.split(Y)))

lRMS = []

while len(lKF) > 0:
    iLoop, (train_index, test_index) = lKF.pop(0)

    print(f"--- Fold: {iLoop + 1}/ {NUM_FOLDS} ---")

    # Clear model to random/init
    model.load_weights(DATA_DIR + 'init_model.h5')

    for i in range(250):
        print(f"Epoch {i + 1}...")

        y_train = Y [train_index]
        y_valid = Y [test_index] 

        X_train = Y[train_index]
        X_valid = Y[test_index]

        X_train = add_swap_noise(X_train, Y, 0.15)
        X_valid = add_swap_noise(X_valid, Y, 0.15)

        # Shuffle
        assert len(y_train) == len(X_train)
        
        p = np.random.permutation(len(y_train))

        X_train = X_train[p]
        y_train = y_train[p]

        assert len(y_valid) == len (X_valid)

        p_v = np.random.permutation(len(y_valid))

        X_valid = X_valid[p_v]
        y_valid = y_valid[p_v]

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


